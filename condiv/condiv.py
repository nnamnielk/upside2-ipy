#!/usr/bin/env python3
import sys,os

upside_home = os.getenv('UPSIDE_HOME')
sys.path.append(f"{upside_home}/py/")
os.environ['THEANO_FLAGS'] = '' # 'cxx=,optimizer=fast_compile'
is_worker = __name__ == '__main__' and sys.argv[1]=='worker'
if not is_worker:
    # This script depends on 'rotamer_parameter_estimation', 'run_upside', and 'upside_engine'
    # which are assumed to be Python 3 compatible.
    import rotamer_parameter_estimation as rp
    import run_upside as ru
    import upside_engine as ue

import numpy as np
import subprocess as sp
import tables as tb
import pickle as cp
from glob import glob
import re
import shutil
import collections
import time
import socket

# These modules are only imported here to avoid breaking the script if they are not available
# when only the main logic is being examined.
if is_worker:
    import run_upside as ru
    import upside_engine as ue


np.set_printoptions(precision=2, suppress=True)

## Important parameters (ARGUMENTS)
n_threads = 6
native_restraint_strength = 1.0/3.0**2  # Use float literals for clarity
rmsd_k = 0   # ATOMS TO CUT FROM EACH END FOR THE LOSS FUNCTION
minibatch_size = 6 # NUMBER OF PROTEINS PER BATCH

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

static_rama_lib_path = f"{upside_home}/parameters/common/rama.dat"
static_ref_state_rama = f"{upside_home}/parameters/common/rama_reference.pkl"

Target = collections.namedtuple('Target', 'fasta native native_path init_path n_res chi'.split())
UpdateBase = collections.namedtuple('UpdateBase', 'env cov rot hyd hb sheet'.split())
class Update(UpdateBase):
    # def __init__(self, *args):
    #     super(Update, self).__init__(*args)

    def _do_binary(self, other, op):
        try:
            len(other)
            is_seq = True
        except TypeError:
            is_seq = False

        if is_seq:
            ret = []
            assert len(self) == len(other)
            for a,b in zip(self,other):
                if a is None or b is None:   # treat None as an non-existent value and propagate
                    ret.append(None)
                else:
                    ret.append(op(a,b))
        else:
            ret = [None if a is None else op(a,other) for a in self]
        return Update(*ret)

    def __add__(self, other):
        return self._do_binary(other, lambda a,b: a+b)
    def __sub__(self, other):
        return self._do_binary(other, lambda a,b: a-b)
    def __mul__(self, other):
        return self._do_binary(other, lambda a,b: a*b)
    def __truediv__(self, other): # Changed from __div__ for Python 3
        return self._do_binary(other, lambda a,b: a/b)


def print_param(param):
    # Changed to print() function
    print('hb    %.3f'% param.hb)
    print('sheet %.3f'% param.sheet)
    
    print('env')
    env_dict = dict(zip(resnames, param.env[:,1::2]))
    for r in hydrophobicity_order:   # easier to understand in hydrophobicity order
        print('   ',r,env_dict[r])

def get_d_obj():
    # Note: Theano is no longer actively maintained. This code assumes a working Python 3 compatible
    # installation of Theano or a replacement like Aesara.
    import theano.tensor as T
    import theano

    cov_wide   = rp.unpack_cov_expr[:,:,2*rp.n_knot_angular:2*rp.n_knot_angular+rp.n_knot_hb]
    cov_narrow = rp.unpack_cov_expr[:,:,2*rp.n_knot_angular+rp.n_knot_hb:]
    cov_rHN    = rp.unpack_cov_expr[:,:,:rp.n_knot_angular]
    cov_rSC    = rp.unpack_cov_expr[:,:,rp.n_knot_angular:2*rp.n_knot_angular]
    
    hyd_wide   = rp.unpack_hyd_expr[:,:,2*rp.n_knot_angular:2*rp.n_knot_angular+rp.n_knot_hb]
    hyd_narrow = rp.unpack_hyd_expr[:,:,2*rp.n_knot_angular+rp.n_knot_hb:]
    hyd_rHN    = rp.unpack_hyd_expr[:,:,:rp.n_knot_angular]
    hyd_rSC    = rp.unpack_hyd_expr[:,:,rp.n_knot_angular:2*rp.n_knot_angular]
    
    rot_wide   = rp.unpack_rot_expr[:,:,2*rp.n_knot_angular:2*rp.n_knot_angular+rp.n_knot_sc]
    rot_narrow = rp.unpack_rot_expr[:,:,2*rp.n_knot_angular+rp.n_knot_sc:]
    rot_rSC1   = rp.unpack_rot_expr[:,:,:rp.n_knot_angular]
    rot_rSC2   = rp.unpack_rot_expr[:,:,rp.n_knot_angular:2*rp.n_knot_angular]

    def student_t_neglog(t, scale, nu):
        return 0.5*(nu+1.)*T.log(1.+(1./(nu*scale**2)) * t**2)

    hb_n_knot   = (rp.n_knot_angular,rp.n_knot_angular, rp.n_knot_hb, rp.n_knot_hb)
    sc_n_knot   = (rp.n_knot_angular,rp.n_knot_angular, rp.n_knot_sc, rp.n_knot_sc)
    cov_energy  = rp.quadspline_energy(rp.unpack_cov_expr, hb_n_knot)
    rot_energy  = rp.quadspline_energy(rp.unpack_rot_expr, sc_n_knot)
    hyd_energy  = rp.quadspline_energy(rp.unpack_hyd_expr, hb_n_knot)

    r_for_cov_energy = np.arange(1,rp.n_knot_hb-1)*rp.hb_dr
    r_for_rot_energy = np.arange(1,rp.n_knot_sc-1)*rp.sc_dr
    r_for_hyd_energy = np.arange(1,rp.n_knot_hb-1)*rp.hb_dr
    
    def cutoff_func(x, scale, module=np):
        return 1./(1.+module.exp(x/scale))
    
    def lower_bound(x, lb):
        return T.where(x<lb, 1e0*(x-lb)**2, 0.*x)
    
    cov_expect = 5.*cutoff_func(r_for_cov_energy-2., 0.2)
    rot_expect = 5.*cutoff_func(r_for_rot_energy-2., 0.2)
    hyd_expect = 5.*cutoff_func(r_for_hyd_energy-1., 0.2)
    
    reg_scale = T.dscalar('reg_scale')
    cov_reg = student_t_neglog(cov_energy - cov_expect[None,None,:,None,None], 200., 3.).sum()/reg_scale
    rot_reg = student_t_neglog(rot_energy - rot_expect[None,None,:,None,None], 200., 3.).sum()/reg_scale
    hyd_reg = student_t_neglog(hyd_energy - hyd_expect[None,None,:,None,None], 200., 3.).sum()/reg_scale
    
    reg_expr = (
            cov_reg + 
            hyd_reg +
            rot_reg + 
            lower_bound(cov_energy, -6.).sum() + 
            lower_bound(rot_energy, -6.).sum() + 
            lower_bound(hyd_energy, -6.).sum() + 
            0.
            )

    rot_coupling = T.dtensor3()
    cov_coupling = T.dtensor3()
    hyd_coupling = T.dtensor3()

    # ARGUMENT: THE LOSS FUNCTION
    obj_expr = ((rot_coupling*rp.unpack_rot_expr).sum() +
                (cov_coupling*rp.unpack_cov_expr).sum() +
                (hyd_coupling*rp.unpack_hyd_expr).sum() +
                reg_expr)

    d_obj = theano.function([rp.lparam, rot_coupling, cov_coupling, hyd_coupling, reg_scale],
            T.grad(obj_expr, rp.lparam))
    return d_obj


if not is_worker:
    d_obj = get_d_obj()

    def get_init_param(init_dir):
        init_param_files = dict(
                env = os.path.join(init_dir, 'environment.h5'),
                rot = os.path.join(init_dir, 'sidechain.h5'),
                hb  = os.path.join(init_dir, 'hbond'),
                sheet  = os.path.join(init_dir, 'sheet'))

        with tb.open_file(init_param_files['rot']) as t:
            rotp = t.root.pair_interaction[:]
            covp = t.root.coverage_interaction[:]
            hydp = t.root.hydrophobe_interaction[:]
            hydplp = t.root.hydrophobe_placement[:]
            rotposp = t.root.rotamer_center_fixed[:]
            rotscalarp = np.zeros((rotposp.shape[0],))

        with tb.open_file(init_param_files['env']) as t:
            env = t.root.energies[:,:-1]

        with open(init_param_files['hb']) as f: hb = float(f.read())
        with open(init_param_files['sheet']) as f: sheet = float(f.read())

        param = Update(*([None]*6))._replace(
                env = env,
                rot = rp.pack_param(rotp, covp, hydp, hydplp, rotposp, rotscalarp[:,None]),
                hb  = hb,
                sheet = sheet)
        return param, init_param_files


    def expand_param(params, orig_param_files, new_param_files):
        rotp, covp, hydp, hydplp, rotposp, rotscalarp = rp.unpack_params(params.rot)

        shutil.copyfile(orig_param_files['rot'], new_param_files['rot'])
        with tb.open_file(new_param_files['rot'],'a') as t:
            t.root.pair_interaction[:]       = rotp
            t.root.coverage_interaction[:]   = covp
            t.root.hydrophobe_interaction[:] = hydp
            t.root.hydrophobe_placement[:]   = hydplp
            t.root.rotamer_center_fixed[:]   = rotposp

        # Changed print >> f to print(..., file=f)
        with open(new_param_files['hb'],'w') as f:    print(params.hb, file=f)
        with open(new_param_files['sheet'],'w') as f: print(params.sheet, file=f)

        shutil.copyfile(orig_param_files['env'], new_param_files['env'])
        with tb.open_file(new_param_files['env'],'a') as t:
            tmp = np.zeros(t.root.energies.shape)
            tmp[:,:-1] = params.env
            tmp[:,-1]  = tmp[:,-3]
            t.root.energies[:] = tmp


    def backprop_deriv(param, deriv_update, reg_scale):
        envd = deriv_update.env[:,:-1].copy()
        envd[:,-2] += deriv_update.env[:,-1]
        return deriv_update._replace(
                rot = d_obj(param.rot, deriv_update.rot, deriv_update.cov, deriv_update.hyd, reg_scale),
                cov = 0.,
                hyd = 0.,
                env = envd)


def run_minibatch(worker_path, param, initial_param_files, direc, minibatch, solver, reg_scale, sim_time):
    if not os.path.exists(direc): os.mkdir(direc)
    print(direc)
    print()

    d_obj_param_files = dict([(k,os.path.join(direc, 'nesterov_temp__'+os.path.basename(x))) 
        for k,x in initial_param_files.items()])
    expand_param(param+solver.update_for_d_obj(), initial_param_files, d_obj_param_files)

    with open(os.path.join(direc,'sim_time'),'w') as f:
            print(sim_time, file=f)

    jobs = collections.OrderedDict()
    for nm,t in minibatch[::-1]:
        # Write the parameters to a temporary file instead of passing as a string
        param_pickle_path = os.path.join(direc, '%s.param.pkl' % nm)
        with open(param_pickle_path, 'wb') as f:
            cp.dump(d_obj_param_files, f, -1)

        #IF ON SLURM RUN THIS

        # args = ['srun', '--nodes=1', '--ntasks=1', '--cpus-per-task=%i'%n_threads, '--slurmd-debug=0',
        #         '--output=%s/%s.output_worker'%(direc,nm),
        #         worker_path,
        #         'worker', nm, direc, t.fasta, t.init_path, str(t.n_res), t.chi,
        #         param_pickle_path, str(sim_time)]

        #OTHERWISE RUN THIS

        python_executable = sys.executable
        args = [python_executable,
                worker_path,
                'worker', nm, direc, t.fasta, t.init_path, str(t.n_res), t.chi,
                param_pickle_path, str(sim_time)]

        #ENDIF

        jobs[nm] = sp.Popen(args, close_fds=True)

    rmsd = dict()
    change = []
    for nm,j in jobs.items():
        if j.wait() != 0: 
            print(nm, 'WORKER_FAIL')
            continue
        # Open pickle files in binary mode ('rb')
        with open('%s/%s.divergence.pkl'%(direc,nm), 'rb') as f:
            divergence = cp.load(f)
            rmsd[nm] = (divergence['rmsd_restrain'], divergence['rmsd'])
            change.append(divergence['contrast'])
    if not change:
        raise RuntimeError('All jobs failed')

    d_param = backprop_deriv(param, Update(*[np.sum(x,axis=0) for x in zip(*change)]), reg_scale)

    # Open pickle files in binary mode ('wb')
    with open('%s/rmsd.pkl' % (direc,),'wb') as f:
        cp.dump(rmsd, f, -1)
    print()
    # .values() returns a view in Py3, convert to list for numpy
    print('Median RMSD %.2f %.2f' % tuple(np.median(np.array(list(rmsd.values())), axis=0)))

    new_param_files = dict([(k,os.path.join(direc, os.path.basename(x))) for k,x in initial_param_files.items()])
    new_param = param+solver.update_step(d_param)
    expand_param(new_param, initial_param_files, new_param_files)
    solver.log_state(direc)

    print()
    print_param(new_param)

    return new_param


def compute_divergence(config_base, pos):
    try:
        with tb.open_file(config_base) as t:
            eps        = t.root.input.potential.rama_map_pot._v_attrs.sheet_eps
            sheet_scale = 1.0/(2.0*eps)
            hb_strength = t.root.input.potential.hbond_energy._v_attrs.protein_hbond_energy
            rot_param_shape = t.root.input.potential.rotamer.pair_interaction.interaction_param.shape
            cov_param_shape = t.root.input.potential.hbond_coverage.interaction_param.shape
            hyd_param_shape = t.root.input.potential.hbond_coverage_hydrophobe.interaction_param.shape
            env_param_shape = t.root.input.potential.nonlinear_coupling_environment.coeff.shape
            more_sheet = t.root.input.potential.rama_map_pot.more_sheet_rama_pot[:]
            less_sheet = t.root.input.potential.rama_map_pot.less_sheet_rama_pot[:]
    except Exception as e:
        print(os.path.basename(config_base)[:5],'ANALYSIS_FAIL',e)
        return None

    engine = ue.Upside(config_base)
    contrast = Update([],[],[],[],[],[])

    engine.set_param(more_sheet, 'rama_map_pot')
    for i in range(pos.shape[0]):
        engine.energy(pos[i])
        contrast.cov.append(engine.get_param_deriv(cov_param_shape, 'hbond_coverage'))
        contrast.rot.append(engine.get_param_deriv(rot_param_shape, 'rotamer'))
        contrast.hyd.append(engine.get_param_deriv(hyd_param_shape, 'hbond_coverage_hydrophobe'))
        contrast.hb.append(engine.get_output('hbond_energy')[0,0]/hb_strength)
        contrast.sheet.append(engine.get_output('rama_map_pot')[0,0]*sheet_scale)
        contrast.env.append(engine.get_param_deriv(env_param_shape, 'nonlinear_coupling_environment'))

    engine.set_param(less_sheet, 'rama_map_pot')
    for i in range(pos.shape[0]):
        engine.energy(pos[i])
        contrast.sheet[i] -= engine.get_output('rama_map_pot')[0,0]*sheet_scale

    contrast = Update(*[np.array(x) for x in contrast])
    return contrast


def main_worker():
    tstart = time.time()
    assert is_worker
    code        = sys.argv[2]
    direc       = sys.argv[3]
    fasta       = sys.argv[4]
    init_path   = sys.argv[5]
    n_res       = int(sys.argv[6])
    chi         = sys.argv[7]
    # Decode the command line argument from string back to bytes for unpickling
    # Load the parameters from the pickle file path provided as an argument
    param_pickle_path = sys.argv[8]
    with open(param_pickle_path, 'rb') as f:
        param_files = cp.load(f)
    sim_time    = float(sys.argv[9])

    n_frame = 250.0
    frame_interval = int(sim_time / n_frame)

    with open(param_files['hb'])    as f: hb        = float(f.read())
    with open(param_files['sheet']) as f: sheet_mix = float(f.read())
    
    kwargs = dict(
            environment_potential               = param_files['env'],
            rotamer_interaction                 = param_files['rot'],
            rotamer_placement                   = param_files['rot'],
            initial_structure                   = init_path,
            hbond_energy                        = param_files['hb'],
            dynamic_rotamer_1body               = True,
            rama_sheet_mix_energy               = param_files['sheet'],
            rama_library                        = static_rama_lib_path,
            reference_state_rama                = static_ref_state_rama,)
    
    T = 0.80 * (1. + np.sqrt(100./n_res)*0.020*np.arange(n_threads-1))**2
    T = np.concatenate((T[0:1],T))

    try:
        config_base = '%s/%s.base.h5' % (direc,code)
        print(f"Running... {ru.upside_config(fasta, config_base, **kwargs)}")
        ru.upside_config(fasta, config_base, **kwargs)
        configs = [re.sub(r'\.base\.h5','.run.%i.h5'%i_rs, config_base) for i_rs in range(len(T))]
        for i in range(1,len(T)):
            shutil.copyfile(config_base, configs[i])

        kwargs['restraint_groups'] = ['0-%i'%(n_res-1)]
        kwargs['restraint_spring'] = native_restraint_strength
        ru.upside_config(fasta, configs[0], **kwargs)
    except tb.NoSuchNodeError:
        raise RuntimeError('CONFIG_FAIL')
    
    j = ru.run_upside('', configs, sim_time, frame_interval, n_threads=n_threads, temperature=T,
            swap_sets=ru.swap_table2d(1,len(T)), mc_interval=5., replica_interval=10.)
    if j.job.wait() != 0: raise RuntimeError('RUN_FAIL')

    swap_stats = []
    with tb.open_file(configs[0]) as t:
        target = t.root.input.pos[:,:,0]
        o = t.root.output
        pos_restrain = o.pos[int(n_frame/2):,0]

    with tb.open_file(configs[1]) as t:
        o = t.root.output
        pos_free     = o.pos[int(n_frame/2):,0]
        swap_stats.extend(o.replica_cumulative_swaps[-1] - o.replica_cumulative_swaps[int(n_frame/2)])

    for nrep in range(3,len(T),2):
        with tb.open_file(configs[nrep]) as t:
            o = t.root.output
            swap_stats.extend(o.replica_cumulative_swaps[-1] - o.replica_cumulative_swaps[int(n_frame/2)])

    divergence = dict()
    alldiv = compute_divergence(config_base, np.concatenate([pos_restrain,pos_free],axis=0))
    if alldiv:
        divergence['contrast'] = Update(*[x[:len(pos_restrain)].mean(axis=0) -
                                          x[len(pos_restrain):].mean(axis=0) for x in alldiv])
        divergence['rmsd_restrain']= ru.traj_rmsd(pos_restrain[:,rmsd_k:-rmsd_k], target[rmsd_k:-rmsd_k]).mean()
        divergence['rmsd']         = ru.traj_rmsd(pos_free    [:,rmsd_k:-rmsd_k], target[rmsd_k:-rmsd_k]).mean()
    else:
        # Handle case where compute_divergence failed
        divergence['contrast'] = Update(*([None]*6)) 
        divergence['rmsd_restrain'] = -1.0
        divergence['rmsd'] = -1.0

    divergence['walltime'] = time.time()-tstart
    divergence['swap_stats'] = np.array(swap_stats)

    for fn in [config_base] + configs:
        os.remove(fn)

    with open('%s/%s.divergence.pkl' % (direc,code),'wb') as f:
        cp.dump(divergence, f, -1)


def main_loop(state_str, max_iter):
    def main_loop_iteration(state):
        print('#########################################')
        print('####      EPOCH %2i MINIBATCH %2i      ####' % (state['epoch'],state['i_mb']))
        print('#########################################')
        print()
        sys.stdout.flush()
    
        tstart = time.time()
        state['mb_direc'] = os.path.join(state['base_dir'],'epoch_%02i_minibatch_%02i'%(state['epoch'],state['i_mb']))
        if os.path.exists(state['mb_direc']):
            shutil.rmtree(state['mb_direc'], ignore_errors=True)
        state['param'] = run_minibatch(state['worker_path'], state['param'], state['init_param_files'],
                state['mb_direc'], state['minibatches'][state['i_mb']],
                state['solver'], state['n_prot']*1., state['sim_time'])
        print()
        print('%.0f seconds elapsed this minibatch'%(time.time()-tstart))
        print()
        sys.stdout.flush()

        state['i_mb'] += 1
        if state['i_mb'] >= len(state['minibatches']):
            state['i_mb'] = 0
            state['epoch'] += 1

    for i in range(max_iter):
        # state_str is bytes, load directly
        state = cp.loads(state_str)
        main_loop_iteration(state)
        state_str = cp.dumps(state,-1)
        # Write checkpoint in binary mode
        with open(os.path.join(state['mb_direc'], 'checkpoint.pkl'),'wb') as f:
            f.write(state_str)


def main_initialize(args):
    state = dict()
    state['init_dir'], state['protein_dir'], protein_list, state['base_dir'], = args
    if not os.path.exists(state['base_dir']): os.mkdir(state['base_dir'])

    if not __file__: raise RuntimeError('No file name available')
    state['worker_path'] = os.path.join(state['base_dir'],'ConDiv.py')
    shutil.copy(__file__, state['worker_path'])

    if protein_list != 'cached':
        print('Reading training set')
        with open(protein_list) as f:
            protein_names = [x.split()[0] for x in f.readlines()]
            assert protein_names[0]=='prot'
            protein_names = protein_names[1:]
                
        training_set = dict()
        excluded_prot = []
        for code in sorted(protein_names):
            base = os.path.join(state['protein_dir'], code)

            # UPDATED: Load .npy file instead of .pkl
            native_pos = np.load(base + '.initial.npy')
            n_res = len(native_pos) // 3

            max_sep = np.sqrt(np.sum(np.diff(native_pos,axis=0)**2,axis=-1)).max()
            if max_sep < 2.:
                # UPDATED: Use .npy path in the Target tuple
                training_set[code] = Target(base+'.fasta', native_pos, base+'.initial.npy',
                    base+'.initial.npy', n_res, base+'.chi')
            else:
                excluded_prot.append(code)
                print(code)

        print('Excluded %i proteins due to chain breaks'%len(excluded_prot))
        with open(os.path.join(state['base_dir'], 'cd_training.pkl'),'wb') as f: cp.dump(training_set, f, -1)
    else:
        with open(os.path.join(state['base_dir'], 'cd_training.pkl'), 'rb') as f:
            training_set = cp.load(f)

    training_list = sorted(training_set.items(), key=lambda x: (x[1].n_res,x[0]))
    np.random.shuffle(training_list)

    minibatch_excess = len(training_list)%minibatch_size
    if minibatch_excess: training_list = training_list[:-minibatch_excess]
    n_minibatch = len(training_list)//minibatch_size
    state['minibatches'] = [training_list[i::n_minibatch] for i in range(n_minibatch)]
    state['n_prot'] = n_minibatch*minibatch_size
    print('Constructed %i minibatches of size %i (%i proteins)' % (n_minibatch, minibatch_size, state['n_prot']))

    if state['init_dir'] != 'cached':
        print('about to get init')
        state['param'], state['init_param_files'] = get_init_param(state['init_dir'])
        print('found init')
        with open(os.path.join(state['base_dir'], 'condiv_init.pkl'),'wb') as f:
            cp.dump((state['init_dir'],state['param'],state['init_param_files']),f,-1)
    else:
        with open(os.path.join(state['base_dir'], 'condiv_init.pkl'), 'rb') as f:
            state['init_dir'],state['param'],state['init_param_files'] = cp.load(f)

    state['initial_alpha'] = Update(*[0.1, 0., 0.5, 0., 0.02, 0.03]) #ARGUMENT: Learning rate
    state['initial_alpha'] = state['initial_alpha'] * 0.25
    state['solver'] = rp.AdamSolver(len(state['initial_alpha']), alpha=state['initial_alpha']) 
    state['sim_time'] = 1000.*4

    print()
    print('Optimizing with solver', state['solver'])
    print()
    print_param(state['param'])
    print()

    state['epoch'] = 0
    state['i_mb'] = 0
    return state

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'worker':
            main_worker()
        elif sys.argv[1] == 'restart':
            assert len(sys.argv) == 4
            print('Running as PID %i on host %s' % (os.getpid(), socket.gethostname()))
            # Open checkpoint file in binary mode to get bytes
            with open(sys.argv[2], 'rb') as f:
                state_str = f.read()
            main_loop(state_str, int(sys.argv[3]))
        elif sys.argv[1] == 'initialize':
            assert len(sys.argv) == 6
            initial_state = main_initialize(sys.argv[2:])
            with open(os.path.join(initial_state['base_dir'],'initial_checkpoint.pkl'),'wb') as f:
                cp.dump(initial_state, f, -1)
        else:
            raise RuntimeError('Illegal mode %s. See script for details.' % sys.argv[1])
    else:
        print("Usage: ./condiv3.py [initialize|restart|worker] [args...]")