#!/usr/bin/env python3
import sys, os
import shutil
import collections
import time
import subprocess as sp
from glob import glob
import re
from multiprocessing import Pool
import pickle as cp
from base64 import b64encode, b64decode
import socket
import numpy as np
import tables as tb

os.environ['THEANO_FLAGS'] = ''  # 'cxx=,optimizer=fast_compile'
is_worker = __name__ == '__main__' and sys.argv[1] == 'worker'

try:
    upside_path = os.getenv("UPSIDE_HOME")
    py_path = upside_path + 'py/'
    if py_path not in sys.path:
        sys.path.append(py_path)
    if not is_worker:
        import rotamer_parameter_estimation as rp
    import run_upside as ru
    import upside_engine as ue
    import mdtraj_upside as mu
    import mdtraj as md
    from scale_params import *
except:
    raise RuntimeError(f'Error importing upside utils from {py_path}')

np.set_printoptions(precision=2, suppress=True)

## Important parameters
n_threads = 4 # this is per protein, must be >3
native_restraint_strength = 1. / 3. ** 2  # weak restraints aiming to hold at about 1.0A RMSD
rmsd_k = 1  # number of atoms to cut from each end of the protein for RMSD calculation. May cause type errors if too low.
minibatch_size = 4
scale_factor = 0.0
balance_target = 0.0
n_frame = 100.
sim_time = 1000.
alpha = 0.5

resnames = ['ALA', 'ARG', 'ASN', 'ASP',
            'CYS', 'GLN', 'GLU', 'GLY',
            'HIS', 'ILE', 'LEU', 'LYS',
            'MET', 'PHE', 'PRO', 'SER',
            'THR', 'TRP', 'TYR', 'VAL']

hydrophobicity_order = ['ASP', 'GLU', 'LYS', 'HIS',
                        'ARG', 'GLY', 'ASN', 'GLN',
                        'ALA', 'SER', 'THR', 'PRO',
                        'CYS', 'VAL', 'MET', 'TYR',
                        'ILE', 'LEU', 'PHE', 'TRP']

Target = collections.namedtuple('Target', 'fasta native native_path init_path n_res chi'.split())
UpdateBase = collections.namedtuple('UpdateBase',
                                    'enve envc envs envw bbenve bbenvc bbenvs bbenvw  cov rot hyd hb dhb sheet'.split())


class Update(UpdateBase):

    def _do_binary(self, other, op):
        try:
            len(other)
            is_seq = True
        except TypeError:
            is_seq = False

        if is_seq:
            ret = []
            assert len(self) == len(other)
            for a, b in zip(self, other):
                if a is None or b is None:  # treat None as an non-existent value and propagate
                    ret.append(None)
                else:
                    ret.append(op(a, b))
        else:
            ret = [None if a is None else op(a, other) for a in self]
        return Update(*ret)

    def __add__(self, other):
        return self._do_binary(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._do_binary(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._do_binary(other, lambda a, b: a * b)

    def __truediv__(self, other):
        return self._do_binary(other, lambda a, b: a / b)


def gen_swap_set(st, n_rep):
    a = ""
    for i in range(st, n_rep - 1, 2):
        a += '{}-{},'.format(i, i + 1)
    b = ""
    for i in range(st + 1, n_rep - 1, 2):
        b += '{}-{},'.format(i, i + 1)
    return a[:-1], b[:-1]


def print_param(param):
    print('hb ', param.hb[0], param.hb[1], param.hb[2], param.dhb[0])
    print('sheet %.3f' % np.mean(param.sheet))

    print('env')
    env_dict = dict(zip(resnames, np.vstack([param.enve, param.envc, param.envs, param.envw[:20]]).transpose()))
    for r in hydrophobicity_order:  # easier to understand in hydrophobicity order
        print('   ', r, env_dict[r])

    print('bb env')
    print(param.bbenve, param.bbenvc, param.bbenvs, param.bbenvw)


def get_d_obj():
    import theano.tensor as T
    import theano

    cov_wide = rp.unpack_cov_expr[:, :, 2 * rp.n_knot_angular:2 * rp.n_knot_angular + rp.n_knot_hb]
    cov_narrow = rp.unpack_cov_expr[:, :, 2 * rp.n_knot_angular + rp.n_knot_hb:]
    cov_rHN = rp.unpack_cov_expr[:, :, :rp.n_knot_angular]
    cov_rSC = rp.unpack_cov_expr[:, :, rp.n_knot_angular:2 * rp.n_knot_angular]

    hyd_wide = rp.unpack_hyd_expr[:, :, 2 * rp.n_knot_angular:2 * rp.n_knot_angular + rp.n_knot_hb]
    hyd_narrow = rp.unpack_hyd_expr[:, :, 2 * rp.n_knot_angular + rp.n_knot_hb:]
    hyd_rHN = rp.unpack_hyd_expr[:, :, :rp.n_knot_angular]
    hyd_rSC = rp.unpack_hyd_expr[:, :, rp.n_knot_angular:2 * rp.n_knot_angular]

    rot_wide = rp.unpack_rot_expr[:, :, 2 * rp.n_knot_angular:2 * rp.n_knot_angular + rp.n_knot_sc]
    rot_narrow = rp.unpack_rot_expr[:, :, 2 * rp.n_knot_angular + rp.n_knot_sc:]
    rot_rSC1 = rp.unpack_rot_expr[:, :, :rp.n_knot_angular]
    rot_rSC2 = rp.unpack_rot_expr[:, :, rp.n_knot_angular:2 * rp.n_knot_angular]

    def student_t_neglog(t, scale, nu):
        return 0.5 * (nu + 1.) * T.log(1. + (1. / (nu * scale ** 2)) * t ** 2)

    hb_n_knot = (rp.n_knot_angular, rp.n_knot_angular, rp.n_knot_hb, rp.n_knot_hb)
    sc_n_knot = (rp.n_knot_angular, rp.n_knot_angular, rp.n_knot_sc, rp.n_knot_sc)
    cov_energy = rp.quadspline_energy(rp.unpack_cov_expr, hb_n_knot)
    rot_energy = rp.quadspline_energy(rp.unpack_rot_expr, sc_n_knot)
    hyd_energy = rp.quadspline_energy(rp.unpack_hyd_expr, hb_n_knot)

    r_for_cov_energy = np.arange(1, rp.n_knot_hb - 1) * rp.hb_dr
    r_for_rot_energy = np.arange(1, rp.n_knot_sc - 1) * rp.sc_dr
    r_for_hyd_energy = np.arange(1, rp.n_knot_hb - 1) * rp.hb_dr

    def cutoff_func(x, scale, module=np):
        return 1. / (1. + module.exp(x / scale))

    def lower_bound(x, lb):
        return T.where(x < lb, 1e0 * (x - lb) ** 2, 0. * x)

    cov_expect = 5. * cutoff_func(r_for_cov_energy - 2., 0.2)
    rot_expect = 5. * cutoff_func(r_for_rot_energy - 2., 0.2)
    hyd_expect = 5. * cutoff_func(r_for_hyd_energy - 1., 0.2)

    reg_scale = T.dscalar('reg_scale')
    cov_reg = student_t_neglog(cov_energy - cov_expect[None, None, :, None, None], 200., 3.).sum() / reg_scale
    rot_reg = student_t_neglog(rot_energy - rot_expect[None, None, :, None, None], 200., 3.).sum() / reg_scale
    hyd_reg = student_t_neglog(hyd_energy - hyd_expect[None, None, :, None, None], 200., 3.).sum() / reg_scale

    reg_expr = (
            cov_reg +
            hyd_reg +
            rot_reg +

            # allow no energies below -5. to avoid weird pathologies
            lower_bound(cov_energy, -6.).sum() +
            lower_bound(rot_energy, -6.).sum() +
            lower_bound(hyd_energy, -6.).sum() +

            0.  # 8
    )  # 6

    # For ease of use (and avoiding double evaluation of simulations for derivs), we don't want
    # to make Upside run within the theano evaulation.  We still want theano to handle the deriv
    # propagation for us.  The way to do this is to define an auxiliary objective f2 instead of the
    # real objective f using the identity (d/dx)(f(g(x))) = (d/dx)(c*g(x)) if c == f'(g(x)).  I call
    # the c the "coupling".
    rot_coupling = T.dtensor3()
    cov_coupling = T.dtensor3()
    hyd_coupling = T.dtensor3()

    obj_expr = ((rot_coupling * rp.unpack_rot_expr).sum() +
                (cov_coupling * rp.unpack_cov_expr).sum() +
                (hyd_coupling * rp.unpack_hyd_expr).sum() +
                reg_expr)

    d_obj = theano.function([rp.lparam, rot_coupling, cov_coupling, hyd_coupling, reg_scale],
                            T.grad(obj_expr, rp.lparam))
    return d_obj


if not is_worker:
    d_obj = get_d_obj()


    def get_init_param(init_dir):
        init_param_files = dict(
            env=os.path.join(init_dir, 'environment.h5'),
            bbenv=os.path.join(init_dir, 'bb_env.dat'),
            rot=os.path.join(init_dir, 'sidechain.h5'),
            hb=os.path.join(init_dir, 'hbond.h5'),
            #nc=os.path.join(init_dir, 'mid_nc.h5'),
            sheet=os.path.join(init_dir, 'sheet'))

        with tb.open_file(init_param_files['rot']) as t:
            rotp = t.root.pair_interaction[:]
            covp = t.root.coverage_interaction[:]
            hydp = t.root.hydrophobe_interaction[:]
            hydplp = t.root.hydrophobe_placement[:]
            rotposp = t.root.rotamer_center_fixed[:]
            rotscalarp = np.zeros((rotposp.shape[0],))  # t.root.rotamer_prob_fixed[:]

        with tb.open_file(init_param_files['env']) as t:
            enve = t.root.scale[:]
            envc = t.root.center[:]
            envs = t.root.sharpness[:]
            envw = t.root.weights[:]

        bb_env_data = np.loadtxt(init_param_files['bbenv'])
        bbenve = bb_env_data[0]
        bbenvc = bb_env_data[1]
        bbenvs = bb_env_data[2]
        bbenvw = bb_env_data[3]

        with tb.open_file(init_param_files['hb']) as t:
            hb = t.root.parameter[list(range(3)) + list(range(4, 12))]
            dhb = t.root.parameter[3:4]

        # with tb.open_file(init_param_files['nc']) as t:
        #     nc = t.root.interaction_param[:]

        sheet = np.loadtxt(init_param_files['sheet'])

        param = Update(*([None] * 14))._replace(
            enve=enve,
            envc=envc,
            envs=envs,
            envw=envw,
            bbenve=bbenve,
            bbenvc=bbenvc,
            bbenvs=bbenvs,
            bbenvw=bbenvw,
            rot=rp.pack_param(rotp, covp, hydp, hydplp, rotposp, rotscalarp[:, None]),
            hb=hb,
            dhb=dhb,
            sheet=sheet)
        return param, init_param_files


    def expand_param(params, orig_param_files, new_param_files):
        rotp, covp, hydp, hydplp, rotposp, rotscalarp = rp.unpack_params(params.rot)

        shutil.copyfile(orig_param_files['rot'], new_param_files['rot'])  # both rot and cov are in the same file
        with tb.open_file(new_param_files['rot'], 'a') as t:
            t.root.pair_interaction[:] = rotp
            t.root.coverage_interaction[:] = covp
            t.root.hydrophobe_interaction[:] = hydp
            t.root.hydrophobe_placement[:] = hydplp
            t.root.rotamer_center_fixed[:] = rotposp
            # t.root.rotamer_prob_fixed[:]     = rotscalarp

        # read sheet and hb from file here and update
        shutil.copyfile(orig_param_files['hb'], new_param_files['hb'])
        with tb.open_file(new_param_files['hb'], 'a') as t:
            param_hb = np.zeros(12)
            param_hb[:3] = params.hb[:3]
            param_hb[4:] = params.hb[3:]
            param_hb[3] = params.dhb[0]
            t.root.parameter[:] = param_hb

        # shutil.copyfile(orig_param_files['nc'], new_param_files['nc'])  # both rot and cov are in the same file
        # with tb.open_file(new_param_files['nc'], 'a') as t:
        #     t.root.interaction_param[:] = params.nc

        np.savetxt(new_param_files['sheet'], params.sheet)

        bb_env_data = np.zeros(4)
        bb_env_data[0] = params.bbenve
        bb_env_data[1] = params.bbenvc
        bb_env_data[2] = params.bbenvs
        bb_env_data[3] = params.bbenvw
        np.savetxt(new_param_files['bbenv'], bb_env_data)

        shutil.copyfile(orig_param_files['env'], new_param_files['env'])
        with tb.open_file(new_param_files['env'], 'a') as t:
            t.root.scale[:] = params.enve
            t.root.center[:] = params.envc
            t.root.sharpness[:] = params.envs
            t.root.weights[:] = params.envw


    def backprop_deriv(param, deriv_update, reg_scale):
        # place all of the derivatives in rot for good measure
        # envd = deriv_update.env[:,:-1].copy()
        # envd[:,-2] += deriv_update.env[:,-1]   # from zero clamp condition
        # hb = deriv_update.hb.copy()
        # hb[:2] -= hb[0]

        return deriv_update._replace(
            rot=d_obj(param.rot, deriv_update.rot, deriv_update.cov, deriv_update.hyd, reg_scale),
            cov=0.,
            hyd=0.)


def run_minibatch(worker_path, param, initial_param_files, direc, minibatch, solver, reg_scale, sim_time):
    if not os.path.exists(direc): os.mkdir(direc)
    print(direc)
    print()

    d_obj_param_files = dict([(k, os.path.join(direc, 'nesterov_temp__' + os.path.basename(x)))
                              for k, x in initial_param_files.items()])
    expand_param(param + solver.update_for_d_obj(), initial_param_files, d_obj_param_files)

    with open(os.path.join(direc, 'sim_time'), 'w') as f:
        print(sim_time, file=f)

    jobs = collections.OrderedDict()
    for nm, t in minibatch[::-1]:
        pickled_params = b64encode(cp.dumps(d_obj_param_files)).decode('ascii')
        args = [worker_path, 'worker', nm, direc, t.fasta, t.init_path, str(t.n_res), t.chi,
                pickled_params, str(sim_time)]

        jobs[nm] = sp.Popen(args, close_fds=True)

    rmsd = dict()
    change = []
    for nm, j in jobs.items():
        if j.wait() != 0:
            print(nm, 'WORKER_FAIL')
            continue
        with open('%s/%s.divergence.pkl' % (direc, nm), 'rb') as f:
            divergence = cp.load(f)
            rmsd[nm] = (divergence['rmsd_restrain'], divergence['rmsd'])
            change.append(divergence['contrast'])
    if not change:
        raise RuntimeError('All jobs failed')

    d_param = backprop_deriv(param, Update(*[np.sum(x, axis=0) for x in zip(*change)]), reg_scale)

    with open('%s/rmsd.pkl' % (direc,), 'wb') as f:
        cp.dump(rmsd, f, -1)
    print()
    print('Median RMSD %.2f %.2f' % tuple(np.median(np.array(list(rmsd.values())), axis=0)))

    ## Update the parameters
    new_param_files = dict([(k, os.path.join(direc, os.path.basename(x))) for k, x in initial_param_files.items()])
    new_param = param + solver.update_step(d_param)
    # solver.update_step(d_param)
    expand_param(new_param, initial_param_files, new_param_files)
    solver.log_state(direc)

    print_param(new_param)

    return new_param


def compute_divergence(config_base, pos, mode=0):
    try:
        with tb.open_file(config_base) as t:
            seq_raw = t.root.input.sequence[:]
            restype_raw = t.root.input.potential.rama_map_pot._v_attrs.restype
            
            if isinstance(seq_raw[0], bytes):
                seq = [s.decode('utf-8') for s in seq_raw]
            else:
                seq = seq_raw
            for i in range(len(seq)):
                if seq[i] == "CPR":
                    seq[i] = "PRO"
            if isinstance(restype_raw[0], bytes):
                restype = [r.decode('utf-8') for r in restype_raw]
            else:
                restype = restype_raw
            
            uniq_seq = np.unique(seq)
            ridx_dict = dict([(x, i) for i, x in enumerate(restype)])

            eps = t.root.input.potential.rama_map_pot._v_attrs.sheet_eps
            sheet_scale = 1. / (2. * eps)
            more_sheet = [None] * uniq_seq.size
            less_sheet = [None] * uniq_seq.size

            for i, residue in enumerate(uniq_seq):
                more_sheet[i] = t.root.input.potential.rama_map_pot['more_sheet_rama_pot_' + residue][:]
                less_sheet[i] = t.root.input.potential.rama_map_pot['less_sheet_rama_pot_' + residue][:]

            # hb_strength = t.root.input.potential.hbond_energy._v_attrs.protein_hbond_energy
            hb_param_shape = t.root.input.potential.hbond_energy.parameters.shape
            # nc_param_shape = t.root.input.potential.NC_pair.interaction_param.shape
            rot_param_shape = t.root.input.potential.rotamer.pair_interaction.interaction_param.shape
            cov_param_shape = t.root.input.potential.hbond_coverage.interaction_param.shape
            hyd_param_shape = t.root.input.potential.hbond_coverage_hydrophobe.interaction_param.shape

            enve_param_size = t.root.input.potential.sigmoid_coupling_environment.scale.shape[0]
            envc_param_size = t.root.input.potential.sigmoid_coupling_environment.center.shape[0]
            envs_param_size = t.root.input.potential.sigmoid_coupling_environment.sharpness.shape[0]
            envw_param_size = t.root.input.potential.sigmoid_coupling_environment.weights.shape[0]
            env_param_shape = (enve_param_size + envc_param_size + envs_param_size + envw_param_size,)

    except Exception as e:
        print(os.path.basename(config_base)[:5], 'ANALYSIS_FAIL', e)
        raise

    # engine = ue.Upside(pos.shape[1], config_base)
    engine = ue.Upside(config_base)
    contrast = Update([], [], [], [], [], [], [], [], [], [], [], [], [], [])
    n_restype = len(restype)
    n_res = len(resnames)
    n_res2 = n_res * n_res

    for i in range(pos.shape[0]):
        engine.energy(pos[i])

        # logarithmic derivative of scale factor
        contrast.cov.append(engine.get_param_deriv(cov_param_shape, 'hbond_coverage'))
        contrast.rot.append(engine.get_param_deriv(rot_param_shape, 'rotamer'))
        contrast.hyd.append(engine.get_param_deriv(hyd_param_shape, 'hbond_coverage_hydrophobe'))

        env = engine.get_param_deriv(env_param_shape, 'sigmoid_coupling_environment')
        contrast.enve.append(env[:enve_param_size])
        contrast.envc.append(env[enve_param_size:enve_param_size + envc_param_size])
        contrast.envs.append(env[enve_param_size + envc_param_size:enve_param_size + envc_param_size + envs_param_size])
        contrast.envw.append(env[enve_param_size + envc_param_size + envs_param_size:])

        bbenv = engine.get_param_deriv((4,), 'bb_sigmoid_coupling_environment')
        contrast.bbenve.append(bbenv[0])
        contrast.bbenvc.append(bbenv[1])
        contrast.bbenvs.append(bbenv[2])
        contrast.bbenvw.append(bbenv[3])

        hb = engine.get_param_deriv(hb_param_shape, 'hbond_energy')
        contrast.hb.append(hb[list(range(3)) + list(range(4, 12))])
        contrast.dhb.append(hb[3:4])
        # contrast.nc.append(engine.get_param_deriv(nc_param_shape, 'NC_pair'))

        contrast.sheet.append(np.zeros(n_restype))

    if mode >= 1:
        for j, r in enumerate(uniq_seq):
            rid = ridx_dict[r]
            engine.set_param(more_sheet[j], 'rama_map_pot')
            for i in range(pos.shape[0]):
                engine.energy(pos[i])
                contrast.sheet[i][rid] = engine.get_output('rama_map_pot')[0, 0] * sheet_scale
            engine.set_param(less_sheet[j], 'rama_map_pot')
            for i in range(pos.shape[0]):
                engine.energy(pos[i])
                contrast.sheet[i][rid] -= engine.get_output('rama_map_pot')[0, 0] * sheet_scale

    # convert to numpy arrays
    contrast = Update(*[np.array(x) for x in contrast])
    return contrast


def compute_frame_properties(argv):
    path_code = argv[0]
    config_base = argv[1]
    trajs = argv[2]
    rep_id = argv[3]
    start = argv[4]

    ## calculate the rmsd
    # get native structure
    fn = mu.load_upside_ref(config_base, add_atoms=False)
    top = fn.top
    ca = top.select("name CA")
    fn_N = fn.atom_slice(ca)
    # get traj
    fn = mu.load_upside_traj(trajs, add_atoms=False)
    xyz = fn.xyz[:] * 10.0
    fn_S = fn.atom_slice(ca)
    SN = fn.n_frames
    rmsd = ru.traj_rmsd(fn_S.xyz[:, rmsd_k:-rmsd_k, :] * 10., fn_N.xyz[0, rmsd_k:-rmsd_k, :] * 10.)
    rg = md.compute_rg(fn_S)

    ## calculate the potential
    pot = ["rotamer", "sigmoid_coupling_environment", "bb_sigmoid_coupling_environment", "hbond_energy", "rama_map_pot"]
    n_pot = len(pot)
    enregies = np.zeros((n_pot + 1, SN))
    engine = ue.Upside(config_base)
    for i in range(SN):
        enregies[0, i] = engine.energy(xyz[i])
        for j in range(n_pot):
            enregies[j + 1, i] = engine.get_output(pot[j])[0, 0]

    return [rmsd, enregies, rg]


def compute_frame_divergence(argv):
    config_base = argv[0]
    trajs = argv[1]
    mode = argv[2]
    start = argv[3]
    with tb.open_file(trajs) as t:
        o = t.root.output
        xyz = o.pos[start:, 0]
    div = compute_divergence(config_base, xyz, mode)
    div = [x for x in div]
    return div


def main_worker():
    tstart = time.time()
    assert is_worker
    code = sys.argv[2]
    direc = sys.argv[3]
    fasta = sys.argv[4]
    init_path_npy = sys.argv[5]
    n_res = int(float(sys.argv[6]))
    chi = sys.argv[7]
    param_files = cp.loads(b64decode(sys.argv[8].encode('ascii')))
    sim_time = float(sys.argv[9])

    init_data = np.load(init_path_npy)
    init_path = os.path.join(direc, code + '.initial.npy')
    np.save(init_path, init_data)

    frame_interval = int(sim_time / n_frame)

    path_code = '{}/{}'.format(direc, code)

    # with open(param_files['hb'])    as f: hb        = float(f.read())
    # with open(param_files['sheet']) as f: sheet_mix = float(f.read())

    rama_lib_path=upside_path + "parameters/common/rama.dat"
    rama_ref_path=upside_path + "parameters/common/rama_reference.pkl"
    assert os.path.exists(rama_lib_path), f"Error: The file '{rama_lib_path}' does not exist."
    assert os.path.exists(rama_ref_path), f"Error: The file '{rama_ref_path}' does not exist."

    ## Configure the files
    kwargs = dict(
        environment_potential=param_files['env'],
        environment_potential_type=1,
        bb_environment_potential=param_files['bbenv'],
        rotamer_interaction=param_files['rot'],
        rotamer_placement=param_files['rot'],  # placement file is currently the same as the interaction file
        dynamic_rotamer_1body=True,
        hbond_energy=param_files['hb'],
        # mid_NC_dist_energy=param_files['nc'],
        rama_sheet_mix_energy=param_files['sheet'],
        rama_param_deriv=True,
        rama_library=rama_lib_path,
        reference_state_rama=rama_ref_path,
    )

    T = 0.85 * (1. + (150. / n_res) ** 0.05 * 0.014 * np.arange(n_threads - 2)) ** 2
    # double the first temperature for the restrained replica
    # double the last temperature for the unfolded state
    T = np.concatenate([T[0:1], T, T[-1:]])

    try:
        config_base = '%s/%s.base.up' % (direc, code)
        configs = [re.sub(r'\.base\.up', '.run.%i.up' % i_rs, config_base) for i_rs in range(len(T))]

        # for unfolding target
        ru.upside_config(fasta, configs[-1], **kwargs)
        shutil.copyfile(configs[-1], configs[-2])
        apply_param_scale(configs[-1], hb_scale=scale_factor, env_scale=scale_factor, rot_scale=scale_factor)

        # for free simulations starting from native
        kwargs['initial_structure'] = init_path
        ru.upside_config(fasta, config_base, **kwargs)
        for i in range(1, n_threads - 2):  # just copy to get different temperatures
            shutil.copyfile(config_base, configs[i])

        # now create the near-native simulation config
        ru.upside_config(fasta, configs[0], **kwargs)
        # apply spring restraints using advanced_config
        ru.advanced_config(configs[0], restraint_groups=['0-%i' % (n_res - 1)], restraint_spring_constant=native_restraint_strength)

    except tb.NoSuchNodeError:
        raise RuntimeError('CONFIG_FAIL')

    ## Launch the jobs
    set1, set2 = ru.swap_table2d(n_threads-1, 1)
    swap_sets = [set1, set2]
    j = ru.run_upside('', configs, sim_time, frame_interval, n_threads=n_threads, temperature=T * 0.05,
                      swap_sets=swap_sets, mc_interval=5., replica_interval=5., time_step=0.015, anneal_factor=20.,
                      anneal_start=96., anneal_end=400.)
    if j.job.wait() != 0: raise RuntimeError('RUN_FAIL')

    divergence = dict()

    equil_fraction = 0.5
    start = int(equil_fraction * n_frame)

    argv = [(path_code, configs[1], configs[i], i, start) for i in range(n_threads)]
    pool = Pool(processes=n_threads)
    traj_data = pool.map(compute_frame_properties, argv)

    # output the TM and RMSD of the first replica
    divergence['rmsd_restrain'] = np.mean(traj_data[0][0])
    divergence['rmsd'] = np.mean(traj_data[1][0])

    # output the trajs data
    Rmsd = []
    energy = []
    dE = []
    Rg = []
    for i in range(n_threads):
        Rmsd.append(traj_data[i][0])
        Rg.append(traj_data[i][2])
        energy.append(traj_data[i][1])
        if i > 0 and i < 4:
            dE.append(energy[i][0][start:])
    with open('%s/%s.Rmsd.pkl' % (direc, code), 'wb') as f:
        cp.dump(Rmsd, f)
    with open('%s/%s.Energy.pkl' % (direc, code), 'wb') as f:
        cp.dump(energy, f)

    # estimate melting temperature
    mRg = []
    for i in range(1, n_threads - 1):
        mRg.append(np.mean(Rg[i][start:]))
    mRg = np.array(mRg)
    mid_Rg = (mRg[0] + mRg[-1]) * 0.5
    left_Rg = mRg[0] * 0.67 + mRg[-1] * 0.33
    right_Rg = mRg[0] * 0.33 + mRg[-1] * 0.67

    rid_left = np.where(mRg < mid_Rg)[0][-1] + 1
    rid_right = rid_left + 1

    print(rid_left, rid_right, mRg[0], mRg[-1], mRg[rid_left - 1], mRg[rid_left])

    argv = [(configs[1], configs[i], 1, start) for i in range(4)]
    argv.append((configs[1], configs[rid_left], 0, start))
    argv.append((configs[1], configs[rid_right], 0, start))
    argv.append((configs[1], configs[-1], 0, start))
    pool = Pool(processes=7)

    condiv_data = pool.map(compute_frame_divergence, argv)

    # with open('%s/%s.dparam.pkl' % (direc,code),'wb') as f:
    #    cp.dump(condiv_data, f)

    # target 1 for native
    # weights of different replica
    dE = np.array(dE)
    for i in range(1, 4):
        dE[i - 1] *= (T[0] - T[i]) / T[i]
    dE[dE < -200.] = -200.
    weight = np.exp(dE)
    weight[0] /= np.sum(weight[0])
    weight[1] /= np.sum(weight[1])
    weight[2] /= np.sum(weight[2])
    weight[0] *= 0.60
    weight[1] *= 0.30
    weight[2] *= 0.10
    weight = weight.flatten()

    Div0 = [x.mean(axis=0) for x in condiv_data[0]]
    Div1 = condiv_data[1]
    for i in range(2, 4):
        div = condiv_data[i]
        for j, data in enumerate(div):
            Div1[j] = np.concatenate([Div1[j], data], axis=0)

    Div1 = [np.average(x, axis=0, weights=weight) for x in Div1]
    dalpha1 = np.array([x - Div1[i] for i, x in enumerate(Div0)])

    # target 2 for unfolding
    Div2 = [x.mean(axis=0) for x in condiv_data[-1]]

    ndx1 = np.where(Rg[rid_left][start:] > left_Rg)[0]
    ndx2 = np.where(Rg[rid_right][start:] > left_Rg)[0]
    Div3 = condiv_data[-3]
    Div4 = condiv_data[-2]

    if ndx1.size > 0:
        Div5 = [np.concatenate([x[ndx1], Div4[i][ndx2]]) for i, x in enumerate(Div3)]
        Div5 = [x.mean(axis=0) for x in Div5]
    else:
        Div5 = [x[ndx2].mean(axis=0) for x in Div4]
    dalpha2 = np.array([x - Div5[i] for i, x in enumerate(Div2)])

    if energy[-2][0][-1] < 1000.:
        dalpha = dalpha1 + balance_target * dalpha2
    else:
        dalpha = dalpha1

    dalphas_update = Update(*[x for x in dalpha])
    divergence['contrast'] = dalphas_update
    divergence['walltime'] = time.time() - tstart

    ## Cleanup files on success (let's just leave for triage on failure)
    for fn in [config_base] + configs:
        os.remove(fn)

    ## Write output
    with open('%s/%s.divergence.pkl' % (direc, code), 'wb') as f:
        cp.dump(divergence, f, -1)


def main_loop(state_str, max_iter):
    def main_loop_iteration(state):
        print('#########################################')
        print('####      EPOCH %2i MINIBATCH %2i      ####' % (state['epoch'], state['i_mb']))
        print('#########################################')
        print()
        # decay = lambda x: 1./(1.+0.5*x)
        # epoch_real = state['epoch']*1. + state['i_mb']*(1./len(state['minibatches']))
        # print('alpha_scale %.2f'%decay(epoch_real))
        # state['solver'].alpha = state['initial_alpha'] * decay(epoch_real)
        sys.stdout.flush()

        tstart = time.time()
        state['mb_direc'] = os.path.join(state['base_dir'],
                                         'epoch_%02i_minibatch_%02i' % (state['epoch'], state['i_mb']))
        if os.path.exists(state['mb_direc']):  # possibly cleanup an earlier failed execution
            shutil.rmtree(state['mb_direc'], ignore_errors=True)
        state['param'] = run_minibatch(state['worker_path'], state['param'], state['init_param_files'],
                                       state['mb_direc'], state['minibatches'][state['i_mb']],
                                       state['solver'], state['n_prot'] * 1., state['sim_time'])
        print()
        print('%.0f seconds elapsed this minibatch' % (time.time() - tstart))
        print()
        sys.stdout.flush()

        # increment counters for next minibatch
        state['i_mb'] += 1
        if state['i_mb'] >= len(state['minibatches']):
            state['i_mb'] = 0
            state['epoch'] += 1
            # state['sim_time'] *= 2 # np.sqrt(3.)  # progressively lengthen simulations to explore new areas of phase space

    for i in range(max_iter):
        # Load the state from a string on every iteration to ensure that we are pickling
        #  everything correctly and thus checkpointing is equivalent to not stopping
        state = cp.loads(state_str)
        main_loop_iteration(state)  # mutates state
        state_str = cp.dumps(state, -1)
        with open(os.path.join(state['mb_direc'], 'checkpoint.pkl'), 'wb') as f:
            f.write(state_str)


def main_initialize(args):
    state = dict()
    state['init_dir'], state['protein_dir'], protein_list, state['base_dir'], = args
    if not os.path.exists(state['base_dir']): os.mkdir(state['base_dir'])

    # Copy this file to the working directory to ensure that it is unchanged during worker invocations
    if not __file__: raise RuntimeError('No file name available')
    state['worker_path'] = os.path.join(state['base_dir'], 'condiv2.py')
    shutil.copy(__file__, state['worker_path'])  # copy preserve execute permission
    os.chmod(state['worker_path'], 0o755)  # make it executable

    ## Read training set
    if protein_list != 'cached':
        print('Reading training set')
        with open(protein_list) as f:
            protein_names = [x.split()[0] for x in f]
            assert protein_names[0] == 'prot'
            protein_names = protein_names[1:]

        training_set = dict()
        excluded_prot = []
        for code in sorted(protein_names):
            base = os.path.join(state['protein_dir'], code)

            init_npy_path = base + '.initial.npy'
            native_pos_raw = np.load(init_npy_path)
            native_pos = native_pos_raw
            n_res = len(native_pos) / 3

            max_sep = np.sqrt(np.sum(np.diff(native_pos, axis=0) ** 2, axis=-1)).max()
            if max_sep < 2.:  # no chain breaks
                training_set[code] = Target(base + '.fasta', native_pos, init_npy_path,
                                            init_npy_path, n_res, base + '.chi')
            else:
                excluded_prot.append(code)
                print(code)

        print('Excluded %i proteins due to chain breaks' % len(excluded_prot))
        with open(os.path.join(state['base_dir'], 'cd_training.pkl'), 'wb') as f:
            cp.dump(training_set, f, -1)
    else:
        with open(os.path.join(state['base_dir'], 'cd_training.pkl'), 'rb') as f:
            training_set = cp.load(f)

    ## Construct minibatches
    # ensure each minibatch has roughly the same mix of protein sizes for variance reduction
    training_list = sorted(training_set.items(), key=lambda x: (x[1].n_res, x[0]))
    np.random.shuffle(training_list)

    minibatch_excess = len(training_list) % minibatch_size
    if minibatch_excess: training_list = training_list[:-minibatch_excess]
    n_minibatch = len(training_list) // minibatch_size
    state['minibatches'] = [training_list[i::n_minibatch] for i in range(n_minibatch)]
    state['n_prot'] = n_minibatch * minibatch_size
    print('Constructed %i minibatches of size %i (%i proteins)' % (n_minibatch, minibatch_size, state['n_prot']))

    if state['init_dir'] != 'cached':
        print('about to get init')
        state['param'], state['init_param_files'] = get_init_param(state['init_dir'])
        print('found init')
        with open(os.path.join(state['base_dir'], 'condiv_init.pkl'), 'wb') as f:
            cp.dump((state['init_dir'], state['param'], state['init_param_files']), f, -1)
    else:
        with open(os.path.join(state['base_dir'], 'condiv_init.pkl'), 'rb') as f:
            state['init_dir'], state['param'], state['init_param_files'] = cp.load(f)

    state['initial_alpha'] = Update(*[
        0.10,  # enve
        0.05,  # envc
        0.02,  # envs
        0.10,  # envw
        0.05,  # bbenve
        0.00,  # bbenvc
        0.00,  # bbenvs
        0.00,  # bbenvw
        0.,  # cov (not needed during backprop because it is in rot)
        0.25,  # rot
        0.,  # hyd (part of hyd)
        0.02,  # hb
        0.01,  # dhb
        0.03])  # sheet
    state['initial_alpha'] = state['initial_alpha'] * alpha
    state['solver'] = rp.AdamSolver(len(state['initial_alpha']), alpha=state['initial_alpha'])
    state['sim_time'] = sim_time  # about 0.5 hour

    print()
    print('Optimizing with solver', state['solver'])
    print()
    print()
    print_param(state['param'])
    print()

    state['epoch'] = 0
    state['i_mb'] = 0
    return state


if __name__ == '__main__':
    if sys.argv[1] == 'worker':
        main_worker()

    elif sys.argv[1] == 'restart':
        assert len(sys.argv[1:]) == 3
        print('Running as PID %i on host %s' % (os.getpid(), socket.gethostname()))
        with open(sys.argv[2], 'rb') as f:
            state_str = f.read()
        main_loop(state_str, int(sys.argv[3]))  # first is checkpoint, 2nd is max_iter

    elif sys.argv[1] == 'initialize':
        initial_state = main_initialize(sys.argv[2:])  # this will dump an initial state file in the working directory
        with open(os.path.join(initial_state['base_dir'], 'initial_checkpoint.pkl'), 'wb') as f:
            cp.dump(initial_state, f, -1)

    else:
        raise RuntimeError('Illegal mode %s.  Please see condiv2.py for details' % sys.argv[1])