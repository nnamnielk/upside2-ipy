import tables as tb
import numpy as np
import tensorflow as tf

def quadspline(radial_middle, angular_log):
    c0 = radial_middle[:,:,:,1:2] # left clamping condition
    cn3 = radial_middle[:,:,:,-1:]
    cn2 = -0.5*cn3
    cn1 = cn3    # these two lines ensure right clamp is at 0
    radial = tf.concat(axis=3, values=[c0,radial_middle,cn2,cn1])

    angular = tf.sigmoid(angular_log)  # bound coefficients on (0,1)
    coeff_array = tf.concat(axis=2, values=[angular[:,:,0,:], angular[:,:,1,:], radial[:,:,0,:], radial[:,:,1,:]])
    return radial, angular, coeff_array

def jointspline(radial_middle, angular_log):
    c0 = radial_middle[:,:,:,1:2] # left clamping condition
    cn3 = radial_middle[:,:,:,-1:]
    cn2 = -0.5*cn3
    cn1 = cn3    # these two lines ensure right clamp is at 0
    radial = tf.concat(axis=3, values=[c0,radial_middle,cn2,cn1])

    angular = tf.sigmoid(angular_log)  # bound coefficients on (0,1)
    coeff_array = tf.concat(axis=2, values=[
        tf.reshape(angular, (tf.shape(angular)[0], tf.shape(angular)[1], n_knot_angular**2,)),
        radial[:,:,0,:],
        radial[:,:,1,:]])
    return radial, angular, coeff_array

    # sc_radial_base  =tf.Variable(tf.truncated_normal((n_restype,n_restype,2,n_knot_sc-3),   stddev=0.1))
    # sc_angular_base =tf.Variable(tf.truncated_normal((n_restype,n_restype,n_knot_angular,n_knot_angular),
    #   stddev=0.1))

    # sc_radial, sc_angular, sc_coeff = jointspline(
    #     0.5*(sc_radial_base  + tf.transpose(sc_radial_base,(1,0,2,3))),
    #     0.5*(sc_angular_base + tf.transpose(sc_angular_base,(1,0,3,2))))
    #
    # bb_radial_base  = tf.Variable(tf.truncated_normal((2+n_fix,n_restype,2,n_knot_hb-3),   stddev=0.1))
    # bb_angular_base = tf.Variable(tf.truncated_normal((2+n_fix,n_restype,n_knot_angular,n_knot_angular),
    #    stddev=0.1))
    # bb_radial, bb_angular, bb_coeff = jointspline(bb_radial_base, bb_angular_base)

def bead_pos(bead_unnorm, n_add_zeros=0):
    pos = bead_unnorm[:,:3]
    direc_unnorm = bead_unnorm[:,3:6]
    direc = direc_unnorm / tf.sqrt(tf.reduce_sum(direc_unnorm**2,axis=1,keep_dims=True))
    coeff_array = tf.concat(axis=1, values=[pos,direc,tf.zeros((tf.shape(pos)[0],n_add_zeros))])
    return pos, direc, coeff_array


class SidechainParam(object):
    def __init__(self, sidechain_init_h5, fixed_prob=False):
        self.fixed_prob = bool(fixed_prob)

        self.placeholders = dict()
        self._read_init(sidechain_init_h5)
        self._make_param()
        self._make_regularizer()

    def static_param_copy(self):
        return dict(self.static_param.items())

    def param_copy(self):
        return dict(self.param.items())

    def _read_init(self, sidechain_init_h5):
        with tb.open_file(sidechain_init_h5) as t:
            g = t.root.input.potential
            self.bb_bead_init = g.placement_fixed_point_vector_scalar.placement_data[:].astype('f4')
            self.sc_bead_init = g.placement_fixed_point_vector_only  .placement_data[:].astype('f4')

            pi = g.rotamer.pair_interaction.interaction_param
            self.sc_init           = pi[:].astype('f4')
            self.n_knot_angular_sc = 8  # pi._v_attrs.n_knot_angular
            self.n_knot_radial_sc  = 9  # pi._v_attrs.n_knot_radial
            self.dr_sc             = 1. # pi._v_attrs.dr

            hi = g.hbond_coverage.interaction_param
            self.hb_init           = hi[:].astype('f4')
            self.n_knot_angular_hb = 8  # hi._v_attrs.n_knot_angular
            self.n_knot_radial_hb  = 7  # hi._v_attrs.n_knot_radial
            self.dr_hb             = 1. # hi._v_attrs.dr

            hi = g.hbond_coverage_hydrophobe.interaction_param
            self.hb_init           = np.concatenate([self.hb_init,hi[:].astype('f4')])
            assert self.n_knot_angular_hb == 8  # hi._v_attrs.n_knot_angular
            assert self.n_knot_radial_hb  == 7  # hi._v_attrs.n_knot_radial
            assert self.dr_hb             == 1. # hi._v_attrs.dr

            self.orig_sc_path = t.root.input.args._v_attrs.rotamer_interaction

            self.bb_bead_init = g.placement_fixed_point_vector_scalar.placement_data[:].astype('f4')
            self.sc_bead_init = g.placement_fixed_point_vector_only  .placement_data[:].astype('f4')
            self.sc_init      = g.rotamer.pair_interaction.interaction_param[:].astype('f4')

            if self.fixed_prob:
                self.sc_prob_init = g.placement_fixed_scalar.placement_data[:,0].astype('f4')

        # Copy the fields that are not affected by optimization and other information to keep for later
        s = 'bead_order rotamer_start_stop_bead rotamer_prob restype_order restype_and_chi_and_state'.split()
        with tb.open_file(self.orig_sc_path) as t:
            self.static_param = dict([(p,t.get_node('/'+p)[:]) for p in s])

        self.init_param = dict(
                rotamer = self.sc_init,
                hbond_coverage = self.hb_init[:2],
                hbond_coverage_hydrophobe = self.hb_init[2:],
                placement_fixed_point_vector_scalar = self.bb_bead_init,
                placement_fixed_point_vector_only = self.sc_bead_init,
                )

        if self.fixed_prob:
            self.init_param['placement_fixed_scalar'] = self.sc_prob_init


    def _make_param(self):
        n_fix     = self.bb_bead_init.shape[0]
        n_rotpos  = self.sc_bead_init.shape[0]
        n_restype = self.sc_init     .shape[0]
        print('fixpostype', n_fix, n_rotpos, n_restype)

        self.sc_radial_base  = tf.Variable(
                tf.random.truncated_normal((n_restype,n_restype,2,self.n_knot_radial_sc-3), stddev=0.1),
                name='sc_radial_base')
        self.sc_angular_base = tf.Variable(
                tf.random.truncated_normal((n_restype,n_restype,1,self.n_knot_angular_sc), stddev=0.1),
                name='sc_angular_base')
        self.sc_radial, self.sc_angular, self.sc_coeff = quadspline(
            0.5*(self.sc_radial_base  + tf.transpose(self.sc_radial_base,(1,0,2,3))),
            tf.concat(axis=2,values=[self.sc_angular_base, tf.transpose(self.sc_angular_base,(1,0,2,3))]))

        self.bb_radial_base  = tf.Variable(tf.random.truncated_normal((2+n_fix,n_restype,2,self.n_knot_radial_hb-3),
            stddev=0.1), name='bb_radial_base')
        self.bb_angular_base = tf.Variable(tf.random.truncated_normal((2+n_fix,n_restype,2,self.n_knot_angular_hb),
            stddev=0.1), name='bb_angular_base')
        self.bb_radial, self.bb_angular, self.bb_coeff = quadspline(self.bb_radial_base, self.bb_angular_base)

        self.bb_bead_base = tf.Variable(self.bb_bead_init[:,:6], name='bb_bead_base')
        self.bb_bead_pos, self.bb_bead_dir, self.bb_bead_coeff = bead_pos(self.bb_bead_base, n_add_zeros=1)

        self.sc_bead_base = tf.Variable(self.sc_bead_init, name='sc_bead_base')
        self.sc_bead_pos, self.sc_bead_dir, self.sc_bead_coeff = bead_pos(self.sc_bead_base)

        self.param = dict()
        def f(name, tensor):
            self.param[name] = tf.identity(tensor, name=name)  # make sure the tensor is named

        f('rotamer',                   self.sc_coeff)
        f('hbond_coverage',            self.bb_coeff[:2])
        f('hbond_coverage_hydrophobe', self.bb_coeff[2:])
        f('placement_fixed_point_vector_scalar', self.bb_bead_coeff)
        f('placement_fixed_point_vector_only',   self.sc_bead_coeff)

        if self.fixed_prob:
            self.sc_prob_base = tf.Variable(tf.random.truncated_normal((n_rotpos,), stddev=0.1))
            self.sc_prob_coeff = self.sc_prob_base
            f('placement_fixed_scalar', self.sc_prob_coeff)

        self.distance_from_init = dict((k, tf.reduce_sum((self.param[k]-self.init_param[k])**2))
                for k in self.param)

    def _make_placeholder(self,name, dtype=tf.float32, shape=()):
        self.placeholders[name] = tf.compat.v1.placeholder(dtype, shape=shape, name=name)
        return self.placeholders[name]

    def _make_regularizer(self):
        # Regularization to encourage smooth behavior, especially near origin
        self.sc_radial_laplacian = (2.*self.sc_radial[:,:,:,1:-1]
                - self.sc_radial[:,:,:,:-2] - self.sc_radial[:,:,:,2:])/self.dr_sc**2
        self.bb_radial_laplacian = (2.*self.bb_radial[:,:,:,1:-1]
                - self.bb_radial[:,:,:,:-2] - self.bb_radial[:,:,:,2:])/self.dr_hb**2

        # we just want the narrow terms to be small but the wide terms to be smooth
        self.reg_laplacian = (
                tf.reduce_sum(self.sc_radial_laplacian[:,:,0]**2) +
                tf.reduce_sum(self.bb_radial_laplacian[:,:,0]**2))
        self.reg_l2 = (
                tf.reduce_sum(self.sc_radial          [:,:,1]**2) +
                tf.reduce_sum(self.bb_radial          [:,:,1]**2) +
                tf.reduce_sum((self.sc_radial[:,:,0,0]-5.)**2)    +
                tf.reduce_sum((self.bb_radial[:,:,0,0]-5.)**2))

        self.regularizer = (
                self._make_placeholder('lambda_laplacian') * self.reg_laplacian +
                self._make_placeholder('lambda_l2')        * self.reg_l2)