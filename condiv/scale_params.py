import tables as tb

def apply_param_scale(config_fn, hb_scale=1., env_scale=1., bb_scale=1., rot_scale=1., mempot_scale=1., rama_scale=1.):
    with tb.open_file(config_fn, 'a') as t:
        pot_group = t.root.input.potential
        if hb_scale != 1.:
            pot_group.hbond_energy.parameters[:4] *= hb_scale
            # pot_group.NC_pair.interaction_param[:] *= hb_scale

            # ToDo: what about the hbond_coverage groups -> how are they coupled
            # to the energy? For env it's clear, but not so for hb

        if env_scale != 1.:
            if 'nonlinear_coupling_environment' in pot_group:
                pot_group.nonlinear_coupling_environment.coeff[:] *= env_scale
            if 'sigmoid_coupling_environment' in pot_group:
                pot_group.sigmoid_coupling_environment.scale[:] *= env_scale

        if bb_scale != 1.:
            if 'bb_sigmoid_coupling_environment' in pot_group:
                pot_group.bb_sigmoid_coupling_environment._v_attrs.scale *= bb_scale

        if rot_scale != 1.:
            pot_group.hbond_coverage.interaction_param[:] *= rot_scale
            pot_group.hbond_coverage_hydrophobe.interaction_param[:] *= rot_scale
            pot_group.rotamer.pair_interaction.interaction_param[:] *= rot_scale

        if rama_scale != 1.:
            pot_group.rama_map_pot.rama_pot[:] *= rama_scale

        if mempot_scale != 1:
            '''
            membrane_potential
                arguments := ['placement_fixed_point_only_CB' 'environment_coverage' 'protein_hbond']

                acceptor_residue_ids (180,) int64
                cb_energy            (21, 236) float64
                    z_max := 29.4
                    z_min := -29.4
                cb_index             (181,) int64
                cov_midpoint         (21,) float64
                cov_sharpness        (21,) float64
                donor_residue_ids    (175,) int64
                env_index            (181,) int64
                residue_type         (181,) int64
                uhb_energy           (2, 236) float64
                    z_max := 29.4
                    z_min := -29.4
            '''
            pot_group.membrane_potential.cb_energy[:]  *= mempot_scale
            pot_group.membrane_potential.uhb_energy[:] *= mempot_scale