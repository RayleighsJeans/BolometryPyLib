""" **************************************************************************
    so header """

import numpy as np
import read_calculation_corona as rcc
import dat_lists

import mfr2D_matrix_gridtransform as grid_transf
import phantom_methods as pm

import factors_geometry_plots as fgp

Z = np.zeros
ones = np.ones

""" end of header
************************************************************************** """


def reff_of_LOS(
        cams=['HBCm', 'VBCr', 'VBCl'],
        camera_info={'none': None},
        reffs=np.zeros((128, 16, 31, 75)),
        emissivity=np.zeros((128, 16, 31, 75)),
        position=np.zeros((128, 16, 31, 75)),
        lines=np.zeros((128, 16, 31, 75)),
        mag_ax=np.zeros((2)),
        new_type=None,
        debug=False):
    """ effective radius of line of sight
    Keyword Arguments:
        camera_info {dict} -- general camera info (default: {{'none': None}})
        reffs {dict} -- reff along LoS (default: {{'none': None}})
        emiss {dict} -- emissivity along LoS (default: {{'none': None}})
        pos_in_mesh {dict} -- position along LoS (default: {{'none': None}})
        line_seg {dict} -- line segment in meh (default: {{'none': None}})
        geomf {dict} -- geometry factors in mesh (default: {{'none': None}})
        mag_ax {dict} -- magnetic axis position (default: {{'none': None}})
        nFS {int} -- number of fluxsurfaces (default: {33})
        nL {int} -- number or poloidal sections (default: {51})
        debug {bool} -- debugging bool (default: {False})
    Returns:
        r_min_LoS {dict} -- mean, minimum and different weighted r_eff
    """
    r_min_LOS = {
        'minimum': Z((128)),
        'emiss': Z((128))}
    N, nFS, nL = np.shape(reffs)[1:]

    f = 1.
    for j, cam in enumerate(cams):
        mag_ax_r = mag_ax[0]

        nCh = [ch for ch in camera_info['channels']['eChannels'][cam]]
        for c, ch in enumerate(nCh):

            nZ = np.where(lines[ch, :, :, :] != 0.0)
            if np.shape(nZ)[1] == 0:
                continue
            nN = np.where(reffs[ch, :, :, :][nZ] != 0.0)

            if ((cam == 'HBCm') and (np.mean(
                    position[ch, :, :, 2][nZ[1], nZ[2]][nN]) < 0.0)):
                f = -1.

            elif ((cam in ['VBC', 'VBCr', 'VBCl']) and (np.mean(
                    position[ch, :, :, 3][nZ[1], nZ[2]][nN]) < mag_ax_r)):
                f = -1.

            elif (cam == 'ARTf') and (new_type == 'MIRh') and (np.mean(
                    position[ch, :, :, 2][nZ[1], nZ[2]][nN]) < 0.0):
                f = -1.

            elif (cam == 'ARTf') and (new_type == 'VBCm') and (np.mean(
                    position[ch, :, :, 3][nZ[1], nZ[2]][nN]) < mag_ax_r):
                f = -1.

            else:
                f = 1.

            r_min_LOS['minimum'][ch] = \
                f * np.min(reffs[-1, :, :, :][nZ][nN])

            tot_emiss = np.sum(emissivity[ch, :, :, :][nZ][nN])
            for S in range(nFS):
                for L in range(nL):
                    for T in range(N):

                        if (lines[ch, T, S, L] == 0.0) or (
                                reffs[ch, T, S, L] == 0.0):
                            continue

                        if np.isnan(emissivity[ch, T, S, L]) or \
                                np.isnan(tot_emiss) or \
                                np.isnan(reffs[ch, T, S, L]) or \
                                np.isnan(f) or (tot_emiss == .0):
                            continue

                        r_min_LOS['emiss'][ch] += f * (
                            emissivity[ch, T, S, L] / tot_emiss) * \
                            reffs[-1, T, S, L]

            if debug:
                print(cam, ch, f,
                      r_min_LOS['emiss'][ch],
                      r_min_LOS['minimum'][ch])
    return (r_min_LOS)


def forward_integrated_LOS(
        material='C_',
        strahl_ids=['00091', '00092', '00093', '00094'],
        labels=['f$_{rad}\\sim$0.33', 'f$_{rad}\\sim$0.66',
                'f$_{rad}\\sim$0.9', 'f$_{rad}\\sim$1.0'],
        emissivity=np.zeros((128, 64, 20, 150)),
        reff=np.zeros((128, 64, 20, 150)),
        reff_LoS={'minimum': np.zeros((128))},
        minor_radius=0.5387891957782814,
        label='EIM_beta000_sN8_30x20x150_1.3',
        plot=False,
        debug=False):
    """ forward integrate STRAHL profiles of line radiation
        for QSB lines of sight and geometry calculations
    Args:
        material (str, optional): STRAHL material. Defaults to 'C_'.
        strahl_ids (list, optional): STRAHL names.
            Defaults to ['00091', '00092', '00093', '00094'].
        labels (list, optional): STRAHL labes.
            Defaults to ['f{rad}\\sim.33', 'f{rad}\\sim.66',
            'f{rad}\\sim.9', 'f{rad}\\sim[summary].0'].
        emissivity ([type], optional): Geometry data.
            Defaults to np.zeros((128, 64, 20, 150)).
        reff ([type], optional): Effective radius per cell.
            Defaults to np.zeros((128, 64, 20, 150)).
        reff_LoS (dict, optional): Effective radius along LOS.
            Defaults to {'minimum': np.zeros((128))}.
        minor_radius (float, optional): Minor plasma radius.
            Defaults to 0.5387891957782814.
        label (str, optional): Geometry label.
            Defaults to 'EIM_beta000_sN8_30x20x150_1.3'.
        plot (bool, optional): Plotting. Defaults to False.
        debug (bool, optional): Debugging. Defaults to False.
    Returns:
        cp (list of np.ndarray): list of chordal profiles
    """
    camera_info = dat_lists.geom_dat_to_json()
    print('\t >> Get forward chordal profile, loading: ...')
    print('\t\t ... configuration: ' + label)
    nFS, nL = np.shape(emissivity)[2:]

    reform_emiss = grid_transf.emiss_matrix_transform(
        emissivity=emissivity, cams=['HBCm', 'VBCl', 'VBCr'],
        suffix='3D', full=True)

    cp = []
    for i, strahl_id in enumerate(strahl_ids):
        phantom, _ = pm.STRAHL_to_phantom(
            nFS=20, nL=150,
            reff=reff, minor_radius=minor_radius,
            strahl_id=strahl_id,
            strgrid=str(nFS) + 'x' + str(nL))
        reform_phantom = phantom.reshape(nFS * nL)
        # simple mat. product
        forward = reform_phantom.dot(reform_emiss)
        cp.append(forward * 1.e6)

    if plot:
        for j, cam in enumerate(['HBCm', 'VBC', 'VBCr', 'VBCl']):
            for R, r_label in zip(
                    [reff_LoS['minimum'], reff_LoS['emiss']],
                    ['min.', 'geom.']):
                fgp.forward_chordal_profile(
                    labels=labels, cam=cam, label=label,
                    chord=cp, camera_info=camera_info, reff=R, reff_label=r_label, m_R=minor_radius,
                    ids=strahl_ids)
    return (cp)
