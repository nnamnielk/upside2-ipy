''' Very opinionated convenience script for running upside jobs on midway '''
import os, sys
import collections
import subprocess as sp
import numpy as np
import tables as tb
import json,uuid
import time

from upside_config import chain_endpts

import upside_engine as ue

# FIXME This assumes that upside-parameters is a sibling of upside in the 
# directory structure.  Later, I will move the parameter directory into the
# upside tree.
py_source_dir = os.path.dirname(__file__)
obj_dir = os.path.join(py_source_dir,'..','obj')

def stop_upside_gently(process, allowed_termination_seconds=60.):
    # Upside checks for sigterm periodically, then writes buffered frames and exits with a 
    # success code.  We give time for a well-behaving Upside process to stop before issuing
    # a sigkill.

    # Python raises an OSError if terminate is called on an already finished process.  This creates
    # a number of possible race conditions in this code.  We will avoid these problems by simply
    # trapping the resulting exceptions, rather than trying to lock against them

    try:
        if process.poll() is not None:
            return

        process.terminate()

        # make sure we can break out of the termination process early
        poll_interval = allowed_termination_seconds/60.
        for poll_index in range(60):
            time.sleep(poll_interval)
            if process.poll() is not None:
                return  # we have succeeded in stopping the job

        process.kill()
    except OSError:
        pass


def upside_config(fasta,
                  output,
                  backbone=True,
                  target='',
                  hbond=None,
                  mid_NC=None,
                  long_hbond=None,
                  sheet_mix_energy=None,
                  init='', 
                  rama_pot=None,
                  rama_param_deriv = False,
                  fix_rotamer = '',
                  dynamic_1body = False,
                  environment='',
                  environment_type=0,
                  environment_weights_number=20,
                  bb_environment='',
                  com_environment='',
                  long_hydrophobic=None,
                  long_hydrophobic_left=None,
                  long_hydrophobic_right=None,
                  amino_environment = False,
                  amino_sc_env = False,
                  cter_hbond = False,
                  placement='',
                  reference_rama='',
                  restraint_groups=[],
                  restraint_spring=None,
                  offset_spring='',
                  wall_spring='',
                  rotamer_interaction_param='',
                  contacts='',
                  cooperation_contacts='',
                  plumed='',
                  plumed_cb='',
                  print_hbond = '',
                  print_cter_hbond = '',
                  print_NH_env = '',
                  external_table_potential='',
                  external_pairs_type='',
                  external_pairs_used='',
                  spherical_well='',
                  secstr_bias='',
                  chain_break_from_file='',
                  apply_restraint_group_to_each_chain=False,
                  cavity_radius=0.,
                  heuristic_cavity_radius=None,
                  cavity_radius_from_config='',
                  make_unbound=False,
                  membrane_potential='',
                  membrane_thickness=0.,
                  membrane_exclude_residues=[]):
    
    args = [os.path.join(py_source_dir, 'upside_config.py'), '--fasta=%s'%fasta, '--output=%s'%output]

    if init:
        args.append('--initial-structure=%s'%init)
    if target:
        args.append('--target-structure=%s'%target)
    if rama_pot is not None:
        args.append('--rama-library=%s'%rama_pot)
    if sheet_mix_energy is not None:
        args.append('--rama-sheet-mixing-energy=%s'%sheet_mix_energy)
    if rama_param_deriv:
        args.append('--rama-param-deriv')
    if not backbone:
        args.append('--no-backbone')
    if hbond:
        args.append('--hbond-energy=%s'%hbond)
    if long_hbond:
        args.append('--long-hbond-energy=%s'%long_hbond)
    if mid_NC:
        args.append('--mid-NC-dist-energy=%s'%mid_NC)
    if reference_rama:
        args.append('--reference-state-rama=%s'%reference_rama)
    for rg in restraint_groups:
        args.append('--restraint-group=%s'%rg)
    if restraint_spring is not None:
        args.append('--restraint-spring-constant=%f'%restraint_spring)
    if offset_spring:
        args.append('--offset-spring=%s'%offset_spring)
    if wall_spring:
        args.append('--wall-spring=%s'%wall_spring)
        
    if rotamer_interaction_param:
        args.append('--rotamer-placement=%s'%placement)
    if dynamic_1body:
        args.append('--dynamic-rotamer-1body')
    if rotamer_interaction_param:
        args.append('--rotamer-interaction=%s'%rotamer_interaction_param)
    if fix_rotamer:
        args.append('--fix-rotamer=%s'%fix_rotamer)

    if environment:
        args.append('--environment-potential=%s'%environment)
    if environment_type:
        args.append('--environment-potential-type=%s'%environment_type)
    if environment_weights_number:
        args.append('--environment-weights-number=%s'%environment_weights_number)

    if bb_environment:
        args.append('--bb-environment-potential=%s'%bb_environment)
    if com_environment:
        args.append('--COM-environment-potential=%s'%com_environment)


    if long_hydrophobic:
        args.append('--long-hydrophobic-energy=%f'%long_hydrophobic)
    if long_hydrophobic_left:
        args.append('--long-hydrophobic-left=%f'%long_hydrophobic_left)
    if long_hydrophobic_right:
        args.append('--long-hydrophobic-right=%f'%long_hydrophobic_right)

    if amino_environment:
        args.append('--amino-environment')
    if amino_sc_env:
        args.append('--amino-sc-env')
    if cter_hbond:
        args.append('--count-cter-hbond')
    
    if secstr_bias:
        args.append('--secstr-bias=%s'%secstr_bias)
    if contacts:
        args.append('--contact-energies=%s'%contacts)
    if cooperation_contacts:
        args.append('--cooperation-contacts=%s'%cooperation_contacts)
    if plumed:
        args.append('--plumed=%s'%plumed)
    if plumed_cb:
        args.append('--plumed-only-cb=%s'%plumed_cb)
    if print_hbond:
        args.append('--print-hbond=%s'%print_hbond)
    if print_cter_hbond:
        args.append('--print-cter-hbond=%s'%print_cter_hbond)
    if print_NH_env:
        args.append('--print-NH-env=%s'%print_NH_env)

    if external_table_potential:
        args.append('--external-pairs-table-potential=%s'%external_table_potential)
    if external_pairs_type:
        args.append('--external-pairs-type=%s'%external_pairs_type)
    if external_pairs_used:
        args.append('--external-pairs-used-percent=%s'%external_pairs_used)
    if spherical_well:
        args.append('--spherical-well=%s'%spherical_well)

    if chain_break_from_file:
        args.append('--chain-break-from-file=%s'%chain_break_from_file)
    if apply_restraint_group_to_each_chain:
        args.append('--apply-restraint-group-to-each-chain')
    if cavity_radius:
        args.append('--cavity-radius=%f'%cavity_radius)
    if heuristic_cavity_radius:
        args.append('--heuristic-cavity-radius=%f'%heuristic_cavity_radius)
    if cavity_radius_from_config:
        args.append('--cavity-radius-from-config=%s'%cavity_radius_from_config)
    if make_unbound:
        args.append('--make-unbound')

    if membrane_potential and not membrane_thickness:
        raise ValueError('Must supply membrane_thickness if using membrane_potential')
    else:
        args.append('--membrane-potential=%s'%membrane_potential)
        args.append('--membrane-thickness=%f'%membrane_thickness)
        for ex_res in membrane_exclude_residues:
            args.append('--membrane-exclude-residues=%s'%ex_res)
        
    return ' '.join(args) + '\n' + sp.check_output(args)


def compile():
    return sp.check_output(['/bin/bash', '-c', 'cd %s; make -j4'%obj_dir])


class UpsideJob(object):
    def __init__(self,job,config,output, timer_object=None):
        self.job = job
        self.config = config
        self.output = output
        self.timer_object = timer_object

    def wait(self,):
        if self.job is None: return 0  # in-process
        retcode = self.job.wait()
        if self.timer_object is not None:
            try:
                self.timer_object.cancel()
            except:  # if cancelling the timer fails, we don't care 
                pass
        return retcode

def run_upside(queue, config, duration, frame_interval, time_limit=None, n_threads=1, minutes=None, temperature=1., seed=None,
               replica_interval=None, anneal_factor=1., anneal_duration=-1., anneal_start=-1., anneal_end=-1., mc_interval=None, input_base=None, output_base=None,
               time_step = None, swap_sets = None, exchange_criterion = None,
               log_level='basic', account=None, disable_recentering=False,
               extra_args=[], verbose=True):
    if isinstance(config,str): config = [config]
    
    upside_args = [os.path.join(obj_dir,'upside'), '--duration', '%f'%duration,
            '--frame-interval', '%f'%frame_interval] + config

    try:
        upside_args.extend(['--temperature', ','.join(map(str,temperature))])
    except TypeError:  # not iterable
        upside_args.extend(['--temperature', str(temperature)])

    if time_limit is not None:
        upside_args.extend(['--time-limit', str(time_limit)])

    if replica_interval is not None:
        upside_args.extend(['--replica-interval', '%f'%replica_interval])
        for s in swap_sets:
            upside_args.extend(['--swap-set', s])
    if mc_interval is not None:
        upside_args.extend(['--monte-carlo-interval', '%f'%mc_interval])
    if exchange_criterion is not None:
        upside_args.extend(['--exchange-criterion', '%d'%exchange_criterion])
    if anneal_factor != 1.:
        upside_args.extend(['--anneal-factor', '%f'%anneal_factor])
    if anneal_duration != -1.:
        upside_args.extend(['--anneal-duration', '%f'%anneal_duration])
    if anneal_start != -1.:
        upside_args.extend(['--anneal-start', '%f'%anneal_start])
    if anneal_end != -1.:
        upside_args.extend(['--anneal-end', '%f'%anneal_end])
    upside_args.extend(['--log-level', log_level])
    
    if time_step is not None:
        upside_args.extend(['--time-step', str(time_step)])
    if disable_recentering:
        upside_args.extend(['--disable-recentering'])

    if input_base is not None:
        upside_args.extend(['--input-base', input_base])
    if output_base is not None:
        upside_args.extend(['--output-base', output_base])

    upside_args.extend(['--seed','%li'%(seed if seed is not None else np.random.randint(1<<31))])
    upside_args.extend(extra_args)
    
    output_path = config[0]+'.output'
    timer_object = None

    if queue == '': 
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(n_threads)
        output_file = open(output_path,'w')
        job = sp.Popen(upside_args, stdout=output_file, stderr=output_file)

        if minutes is not None:
            # FIXME in Python 3.3+, subprocess supports job timeout directly
            import threading
            timer_object = threading.Timer(minutes*60., stop_upside_gently, args=[job])
            timer_object.start()

    elif queue == 'srun':
        # set num threads carefully so that we don't overwrite the rest of the environment
        # setting --export on srun will blow away the rest of the environment
        # afterward, we will undo the change

        assert minutes is None  # time limits currently not supported by srun-launcher

        old_omp_num_threads = os.environ.get('OMP_NUM_THREADS', None)

        try:
            os.environ['OMP_NUM_THREADS'] = str(n_threads)
            args = ['srun', '--ntasks=1', '--nodes=1', '--cpus-per-task=%i'%n_threads, 
                    '--slurmd-debug=0', '--output=%s'%output_path] + upside_args
            job = sp.Popen(args, close_fds=True)
        finally:
            if old_omp_num_threads is None:
                del os.environ['OMP_NUM_THREADS']
            else:
                os.environ['OMP_NUM_THREADS'] = old_omp_num_threads
    elif queue == 'in_process':
        import upside_engine as ue
        if verbose: print 'args', ' '.join(upside_args)
        os.environ['OMP_NUM_THREADS'] = str(n_threads)
        job = ue.in_process_upside(upside_args[1:], verbose=verbose)
    else:
        args = ['sbatch',
                '--no-requeue', # do not restart the job on a SLURM problem
                '-p', queue, 
                '--time=%i'%(minutes if minutes is not None else 36*60),
                '--ntasks=1', 
                '--cpus-per-task=%i'%n_threads, '--export=OMP_NUM_THREADS=%i'%n_threads,
                '--output=%s'%output_path, '--parsable', '--wrap', ' '.join(upside_args)]
        if account is not None:
            args.append('--account=%s'%account)
        job = sp.check_output(args).strip()

    return UpsideJob(job,config,output_path, timer_object=timer_object)

def continue_sim(configs, partition='', duration=0, frame_interval=0, **upside_kwargs):
    upside_kwargs = dict(upside_kwargs)
    temps = []

    for fn in configs:
        with tb.open_file(fn,'a') as t:
            i = 0
            while 'output_previous_%i'%i in t.root:
                i += 1
            new_name = 'output_previous_%i'%i
            if 'output' in t.root:
                n = t.root.output
            else:
                n = t.get_node('/output_previous_%i'%(i-1))

            t.root.input.pos[:,:,0] = n.pos[-1,0]
            temps.append(n.temperature[-1,0])

            if 'output' in t.root:
                t.root.output._f_rename(new_name)
            # print fn, temps[-1]

    if partition:
        upside_kwargs['temperature'] = temps
        return run_upside(partition, configs, duration, frame_interval, **upside_kwargs)

def read_output(t, output_name, stride):
    """Read output from continued Upside h5 files."""
    def output_groups():
        i=0
        while 'output_previous_%i'%i in t.root:
            yield t.get_node('/output_previous_%i'%i)
            i += 1
        if 'output' in t.root: 
            yield t.get_node('/output')
            i += 1

    start_frame = 0
    total_frames_produced = 0
    output = []
    time = []
    for g_no, g in enumerate(output_groups()):
        # take into account that the first frame of each output is the same as the last frame before restart
        # attempt to land on the stride
        sl = slice(start_frame,None,stride)
        output.append(g._f_get_child(output_name)[sl,:])
        total_frames_produced += g._f_get_child(output_name).shape[0]-(1 if g_no else 0)  # correct for first frame
        start_frame = 1 + stride*(total_frames_produced%stride>0) - total_frames_produced%stride
    output = np.concatenate(output,axis=0)
    return output

def compute_com_dist(config_fn):
    """Compute center of mass distance between receptor and ligand."""
    with tb.open_file(config_fn) as t:
        n_res = len(t.root.input.sequence[:])
        try:
            xyz = read_output(t, "pos", 1)[:,0,:,:] 
        except ValueError:
            print "No output for %s" % config_fn
            sys.exit(1)
        else:
            chain_first_residue = t.root.input.chain_break.chain_first_residue[:]
            rl_chains = t.root.input.chain_break.rl_chains[:]

    n_chains = len(chain_first_residue) + 1

    # receptor com
    first_res = chain_endpts(n_res, chain_first_residue, 0)[0]
    next_first_res = chain_endpts(n_res, chain_first_residue, rl_chains[0]-1)[1]
    r_com = xyz[:,first_res*3:next_first_res*3].mean(axis=1)

    # ligand com
    first_res = chain_endpts(n_res, chain_first_residue, rl_chains[0])[0]
    next_first_res = chain_endpts(n_res, chain_first_residue, n_chains-1)[1]
    l_com = xyz[:,first_res*3:next_first_res*3].mean(axis=1)

    com_dist = vmag(r_com - l_com)
    return com_dist

def status(job):
    try:
        job_state = sp.check_output(['/usr/bin/env', 'squeue', '-j', job.job, '-h', '-o', '%t']).strip()
    except sp.CalledProcessError:
        job_state = 'FN'
        
    if job_state == 'PD':
        status = ''
    else:
        status = sp.check_output(['/usr/bin/env','tail','-n','%i'%1, job.output])[:-1]
    return '%s %s' % (job_state, status)


def read_hb(tr):
    n_res = tr.root.input.pos.shape[0]/3
    don_res =  tr.root.input.potential.infer_H_O.donors.id[:,1] / 3
    acc_res = (tr.root.input.potential.infer_H_O.acceptors.id[:,1]-2) / 3
    
    n_hb = tr.root.output.hbond.shape[1]
    hb_raw   = tr.root.output.hbond[:]
    hb = np.zeros((hb_raw.shape[0],n_res,2,2))

    hb[:,don_res,0,0] =    hb_raw[:,:len(don_res)]
    hb[:,don_res,0,1] = 1.-hb_raw[:,:len(don_res)]

    hb[:,acc_res,1,0] =    hb_raw[:,len(don_res):]
    hb[:,acc_res,1,1] = 1.-hb_raw[:,len(don_res):]
    
    return hb

def read_constant_hb(tr, n_res):
    don_res = tr.root.input.potential.infer_H_O.donors.residue[:]
    acc_res = tr.root.input.potential.infer_H_O.acceptors.residue[:]
    
    n_hb = tr.root.output.hbond.shape[2]
    hb_raw   = tr.root.output.hbond[:,0]

    hb = np.zeros((hb_raw.shape[0],n_res,2,3))

    hb[:,don_res,0,0] = hb_raw[:,:len(don_res),0]
    hb[:,don_res,0,1] = hb_raw[:,:len(don_res),1]
    hb[:,don_res,0,2] = 1.-hb_raw[:,:len(don_res)].sum(axis=-1)

    hb[:,acc_res,1,0] = hb_raw[:,len(don_res):,0]
    hb[:,acc_res,1,1] = hb_raw[:,len(don_res):,1]
    hb[:,acc_res,1,2] = 1.-hb_raw[:,len(don_res):].sum(axis=-1)
    
    return hb


def rmsd_transform(target, model):
    assert target.shape == model.shape == (model.shape[0],3)
    base_shift_target = target.mean(axis=0)
    base_shift_model  = model .mean(axis=0)
    
    target = target - target.mean(axis=0)
    model = model   - model .mean(axis=0)

    R = np.dot(target.T, model)
    U,S,Vt = np.linalg.svd(R)
    if np.linalg.det(np.dot(U,Vt))<0.:
        Vt[-1] *= -1.  # fix improper rotation
    rot = np.dot(U,Vt)
    shift = base_shift_target - np.dot(rot, base_shift_model)
    return rot, shift


def structure_rmsd(a,b):
    rot,trans = rmsd_transform(a,b)
    diff = a - (trans+np.dot(b,rot.T))
    return np.sqrt((diff**2).sum(axis=-1).mean(axis=-1))


def traj_rmsd(traj, native):
    return np.array([structure_rmsd(x,native) for x in traj])


def vmag(x):
    assert x.shape[-1] == 3
    return np.sqrt(x[...,0]**2 + x[...,1]**2 + x[...,2]**2)


def vhat(x):
    return x / vmag(x)[...,None]


def compact_sigmoid(x, sharpness):
    y = x*sharpness;
    result = 0.25 * (y+2) * (y-1)**2
    result = np.where((y< 1), result, np.zeros_like(result))
    result = np.where((y>-1), result, np.ones_like (result))
    return result


def compute_topology(t):
    seq = t.root.input.sequence[:]
    infer = t.root.input.potential.infer_H_O
    n_donor = infer.donors.id.shape[0]
    n_acceptor = infer.acceptors.id.shape[0]
    id = np.concatenate((infer.donors.id[:],infer.acceptors.id[:]), axis=0)
    bond_length = np.concatenate((infer.donors.bond_length[:],infer.acceptors.bond_length[:]),axis=0)
    
    def augment_pos(pos, id=id, bond_length=bond_length):
        prev = pos[id[:,0]]
        curr = pos[id[:,1]]
        nxt  = pos[id[:,2]]
        
        virtual = curr + bond_length[:,None] * vhat(vhat(curr-nxt) + vhat(curr-prev))
        new_pos = np.concatenate((pos,virtual), axis=0)
        return json.dumps([map(float,x) for x in new_pos])  # convert to json form
    
    n_atom = 3*len(seq)
    backbone_names = ['N','CA','C']
    
    backbone_atoms = [dict(name=backbone_names[i%3], residue_num=i/3, element=backbone_names[i%3][:1]) 
                      for i in range(n_atom)]
    virtual_atoms  = [dict(name=('H' if i<n_donor else 'O'), residue_num=int(id[i,1]/3), 
                           element=('H' if i<n_donor else 'O'))
                     for i in range(n_donor+n_acceptor)]
    backbone_bonds = [[i,i+1] for i in range(n_atom-1)]
    virtual_bonds  = [[int(id[i,1]), n_atom+i] for i in range(n_donor+n_acceptor)]
    
    topology = json.dumps(dict(
        residues = [dict(resname=str(s), resid=i) for i,s in enumerate(seq)],
        atoms = backbone_atoms + virtual_atoms,
        bonds = backbone_bonds + virtual_bonds,
    ))
    
    return topology, augment_pos


def display_structure(topo_aug, pos, size=(600,600)):
    import IPython.display as disp
    id_string = uuid.uuid4()
    return disp.Javascript(lib='/files/js/protein-viewer.js', 
                    data='render_structure(element, "%s", %i, %i, %s, %s);'%
                       (id_string, size[0], size[1], topo_aug[0], topo_aug[1](pos))), id_string

def swap_table2d(nx,ny):
    idx = lambda xy: xy[0]*ny + xy[1]
    good = lambda xy: (0<=xy[0]<nx and 0<=xy[1]<ny)
    swap = lambda i,j: '%i-%i'%(idx(i),idx(j)) if good(i) and good(j) else None
    horiz0 = [swap((a,b),(a+1,b)) for a in range(0,nx,2) for b in range(0,ny)]
    horiz1 = [swap((a,b),(a+1,b)) for a in range(1,nx,2) for b in range(0,ny)]
    vert0  = [swap((a,b),(a,b+1)) for a in range(0,nx)   for b in range(0,ny,2)]
    vert1  = [swap((a,b),(a,b+1)) for a in range(0,nx)   for b in range(1,ny,2)]
    sets = (horiz0,horiz1,vert0,vert1)
    sets = [[y for y in x if y is not None] for x in sets]
    return [','.join(x) for x in sets if x]
