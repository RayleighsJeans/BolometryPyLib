""" **************************************************************************
    so header """

import os
import numpy as np

import dat_lists as dat_lists
import mesh_2D as mesh2D
import LoS_emissivity3D as LoS3D
import LoS_volume as LoSv

import profile_to_lineofsight as profile
import mfr_plot as mfrp
import mfr2D_matrix_gridtransform as mfr_transf
import phantom_methods as pm

Z = np.zeros
ones = np.ones

""" end of header
************************************************************************** """


def ch_pow_phantoms(
        strgrid='tN2_20x20x50_1.4',
        sigma=0.2,  # in (W/m^3)/m
        x0=[5.75, 0.0],  # in m
        amplitude=1.e6,  # in W/m^3
        error=0.025,  # in % of max amplitude, W/m^3
        new_type=None,
        in_label='_EIM_beta000_sN8_50x30x100_1.4',
        save_base='../results/INVERSION/MFR/',
        add_camera=False,
        add_noise=False,
        systematic_error=False,
        debug=False):
    """ generating the phantom and calculating the forward profiles,
        emissivities, geometries and add noise maybe
    Args:
        N (int, optional): Interpolation of det/slit. Default to 8.
        nL (int, optional): Lines. Defaults to 125.
        nFS (int, optional): FS. Defaults to 31.
        cartesian (bool, optional): Cartesian grid? Defaults to False.
        add_camera (bool, optional): New camera? Defaults to False.
        new_type (str, optional): Camera type. Defaults to 'ARTh'.
        add_noise (bool, optional): Add noise? Defaults to False.
        systematic_error (bool, optional): Systematic fault. Defaults to False.
        sigma (float, optional): Width of profile. Defaults to 0.2.
        x0 (list, optional): Position of spots. Defaults to [5.0, 0.2].
        amplitude ([type], optional): Amplitude. Defaults to 6..
        error (float, optional): Error for stddiv/noise. Defaults to 0.025.
        label (str): Label of run setting
        save_base (str, optional): Defaults to '../results/INVERSION/MFR/'.
        debug (bool, optional): Defaults to False.
    Returns:
        None.
    """
    if not add_camera:
        cams = ['HBCm', 'VBCr', 'VBCl']
    else:
        if (new_type not in in_label):
            print('\\\ failed, new type of cam not in files, returning')
            return (None)
        cams = ['HBCm', 'VBCr', 'VBCl', 'ARTf']

    save_base += strgrid + '/'
    mag_config = in_label[1:12]
    camera_info = dat_lists.geom_dat_to_json()

    print('\t>> loading files ...')
    mesh = mesh2D.store_read_mesh2D(name=in_label[1:])
    lines3D = LoS3D.store_read_line_sections(name=in_label[1:])
    volume = LoSv.store_read_volume(name=in_label[1:])
    emiss3D = LoS3D.store_read_emissivity(
        name=in_label[1:], suffix='3D')[1]
    reff, position, minor_radius = profile.store_read_profile(
        name=in_label[1:])
    N, nFS, nL = np.shape(emiss3D)[1:]

    data, label = forward_calculation_phantom(
        nFS=nFS, nL=nL, amplitude=amplitude, sigma=sigma, x0=x0,
        cams=cams, new_type=new_type, save_base=save_base,
        error=error, add_noise=add_noise, systematic_error=systematic_error,
        strgrid=strgrid, in_label=in_label, mesh=mesh,
        reff=reff, emissivity=emiss3D,
        position=position, minor_radius=minor_radius)
    label = label[1:]

    if add_camera:
        reset_geometry(
            label=in_label, cams=cams, position=position,
            new_type=new_type, debug=debug, save_base=save_base,
            mesh=mesh, emissivity=emiss3D, volume=volume,
            lines=lines3D, reff=reff, minor_radius=minor_radius)

    # set up fake time from -1. to 1. with 21 points
    time = np.linspace(-1., 1., 21)

    stddiv = error * np.max(data)
    error = np.zeros((np.shape(data)))
    error[:] = stddiv
    # at U_d= 4.000e-04 V the power P= 9.599e-05 W
    # s= 1.599e-06 W; dpower= -1.787e-08 W

    print('\t>> label:', label)
    for c, cam in enumerate(cams):
        nCh = [ch for ch in camera_info['channels']['eChannels'][cam]]

        N, M = np.shape(time)[0], np.shape(nCh)[0]
        output = Z((N, 1 + M + M))

        output[:, 0] = time
        for n, ch in enumerate(nCh):
            output[:, 1 + n] = data[ch]
            output[:, 1 + M + n] = stddiv

        if cam == 'ARTf':
            cam = new_type
        f = label + '_' + mag_config + '_' + strgrid + \
            '_chnpow_' + cam + '.dat'
        for file in [  # save_base,
                '../../bolometer_mfr/chpow/' + label + '/']:

            if not os.path.exists(file):
                if debug:
                    print(file)
                os.mkdir(file)

            np.savetxt(
                file + f, output, comments='',
                delimiter='    ', fmt='%.7e')
            print('\t\t>> saved as:', file + f)

    mfrp.phantom_forward_plot(
        strgrid=strgrid, data=data, error=error, label=label,
        cams=cams, save_base=save_base, new_type=new_type)
    return


def noise(
        error=0.025,  # in % of max amplitude, W/m^3
        data=np.zeros((64, 10000)),
        label='_EIM_beta000_sN8_50x30x100_1.4',
        debug=False):
    """ Add noise to channel data
    Args:
        error (float, optional): Noise level. Defaults to 0.025.
        data ([type], optional): Channel data. Defaults to Z((64, 10000)).
        label (str, optional): Label string. Defaults to 'None'.
        systematic (bool, optional): Systematic fault? Defaults to False.
    Returns:
        data (np.ndarray): Channel power.
        label (string): Label string.
    """
    noise = np.random.normal(0, .5, 128) * error * np.max(data)
    data = np.abs(data + noise)  # W/m^3
    label += '_rns' + str(error * 100.)
    if debug:
        print('\t\tnoise:', np.shape(noise), noise.__class__)
    print('\tnoise level: sqrt(sum(sq(noise))):',
          np.sqrt(np.sum(np.square(noise))))
    return (data, label)


def reset_geometry(
        mesh=np.zeros((2, 30, 100)),
        emissivity=np.zeros((128, 16, 30, 100)),
        volume=np.zeros((128, 16, 30, 100)),
        lines=np.zeros((128, 16, 30, 100)),
        position=np.zeros((128, 16, 30, 100, 2)),
        reff=np.zeros((128, 16, 30, 100)),
        minor_radius=0.513,
        magnetic_axis={'x1': None},
        cams=['HBCm', 'VBCl', 'VBCr', 'ARTf'],
        label='_EIM_beta000_sN8_50x30x100_1.4',
        save_base='../results/INVERSION/MFR/',
        new_type=None,
        debug=False):
    """ Resetting geometryie for the chosen phantom grid and resolution
    Args:
        cartesian (bool, optional): Cartesian grid? Defaults to False.
        extrapolate (int, optional): Added nFS. Defaults to 5.
        lines (int, optional): Poloidal lines. Defaults to 50.
        cams (list, optional): Defaults to ['HBCm', 'VBCl', 'VBCr', 'ARTf'].
        new_type (str, optional): New camera. Defaults to 'ARTh'.
        save_base (str, optional): Defaults to '../results/INVERSION/MFR/'.
        file_LoS (str, optional): LoS profile source.
            Defaults to 'lofs_profile_EIM_beta000_slanted_HBCm_95_10.json'.
        file_factors (str, optional): Geometry factors source.
            Defaults to 'geom_factors_EIM_beta000_slanted_HBCm_95_10.json'.
        file_mesh (str, optional): Mesh source.
            Defaults to 'mesh_EIM_beta000_slanted_HBCm_10000_95_10.json'.
        vp_tor (float, optional): Plasma FS volume. Defaults to 1.1.
        debug (bool, optional): Defaults to False.
    """
    print('\t>> Reseting geometry because new camera...')

    mfr_transf.emiss_matrix_transform(
        emissivity=emissivity, save_base=save_base, cams=cams,
        suffix='3D', debug=debug, label=label, new_type=new_type)
    mfr_transf.grid_matrix_transform(  # grid matrix positions
        mesh=mesh, reff=reff, label=label,
        debug=debug, save_base=save_base)
    mfr_transf.kfac_volume_transform(  # k factor and volume of channel
        emissivity=emissivity, volume=volume, suffix='3D', cams=cams,
        label=label, save_base=save_base, new_type=new_type, debug=False)
    mfr_transf.effective_radius_LoS_transform(
        label=label, save_base=save_base, reff=reff, cams=cams,
        lines=lines, m_R=minor_radius, position=position,
        emissivity=emissivity, debug=debug, new_type=new_type)
    return


def forward_calculation_phantom(
        nFS=30,
        nL=100,
        sigma=0.2,  # in (W/m^3)/m
        x0=[5.75, 0.0],  # m
        amplitude=1.e6,  # in W/m^3
        error=0.025,  # in % of max amplitude
        minor_radius=0.513,
        mesh=np.zeros((2, 30, 100)),
        emissivity=np.zeros((128, 64, 30, 100)),
        reff=np.zeros((128, 64, 30, 100)),
        position=np.zeros((128, 64, 30, 100, 2)),
        strgrid='sN8_50x30x100_1.4',
        cams=['HBCm', 'VBCl', 'VBCr'],
        in_label='_EIM_beta000_sN8_50x30x100_1.4',
        save_base='../results/INVERSION/MFR/',
        add_noise=False,
        systematic_error=False,
        new_type=None,
        debug=False):
    """ Generating the phantom from a list of predefined possibilities
        that may or may not represent the most common plasma radiation sources
    Args:
        nFS ([type], optional): Fluxsurfaces. Defaults to 21+10.
        nL (int, optional): Poloidal lines. Defaults to 95.
        sigma (float, optional): Mean width. Defaults to 0.2.
        x0 (list, optional): Spot position. Defaults to [5.0, 0.2].
        amplitude ([type], optional): Amplitude. Defaults to 6..
        strgrid (str, optional): Grid string. Defaults to '31x75'.
        cams (list, optional): Defaults to ['HBCm', 'VBCl', 'VBCr'].
        new_type (str, optional): Artificial camera. Defaults to 'ARTh'.
        save_base (str, optional): Defaults to '../results/INVERSION/MFR/'.
        file_fs (str, optional): Fluxsurface file source.
            Defaults to 'EIM_beta000_slanted_HBCmfs_data.json'.
        cartesian (bool, optional): Cartesian grid? Defaults to False.
        debug (Dbool, optional): Defaults to False.
    Returns:
        foward (np.ndarray): Forward calculated channel data.
        label (string): Name of phantom.
    """
    # p1, l1 = pm.STRAHL_to_phantom(
    #     strahl_id='00091', nFS=nFS, nL=nL,
    #     reff=reff, minor_radius=minor_radius,
    #     strgrid=strgrid, debug=debug, save_base=save_base)
    p1, l1 = pm.spot_pos_mesh(
        nFS=nFS, nL=nL, sigma=sigma, x0=x0, amplitude=amplitude,
        position=position, strgrid=strgrid, debug=debug)

    R0, R1 = 1.1, 0.85  # 0.8, 1.1  # in units of r_a, minor radius
    A0, A1 = amplitude, amplitude
    # p1, l1 = pm.radial_profile(
    #     nFS=nFS, nL=nL, sigma1=0.4, sigma2=0.05, minor_radius=minor_radius,
    #     trgt_reff=R1, amplitude=amplitude, reff_in_cell=reff,
    #     debug=debug, strgrid=strgrid)
    # p0, l0 = pm.fluxsurface_reff_ring(
    #     trgt_reff=R1, strgrid=strgrid, reff_in_cell=reff,
    #     minor_radius=minor_radius, nFS=nFS, nL=nL, sigma=sigma, amplitude=A1,
    #     debug=debug, save_base=save_base)
    # p1, l1 = pm.fluxsurface_reff_ring(
    #     trgt_reff=R1, strgrid=strgrid, reff_in_cell=reff,
    #     minor_radius=minor_radius, nFS=nFS, nL=nL, sigma=sigma, amplitude=A1,
    #     debug=debug, save_base=save_base)
    # p1, l1 = pm.anisotropic_fluxsurface_ring(
    #     trgt_reff=R0, nFS=nFS, nL=nL, sigma=sigma, amplitude=A0,
    #     reff_in_cell=reff, minor_radius=minor_radius,
    #     mode_number=5, offset=.0, symmetric=True, strgrid=strgrid,
    #     save_base=save_base, debug=debug)
    # p3, l3 = pm.anisotropic_fluxsurface_ring(
    #     trgt_reff=R1, nFS=nFS, nL=nL, sigma=sigma, amplitude=A1,
    #     reff_in_cell=reff, minor_radius=minor_radius, mode_number=1,
    #     symmetric=False, strgrid=strgrid, save_base=save_base, debug=debug)
    # p1, l1 = pm.blind_test_ones(
    #     nFS=nFS, nL=nL, strgrid=strgrid, save_base=save_base,
    #     amplitude=amplitude, debug=debug)

    phantom = p1  # + p2  # / 2. + 1.5 * p1 + p2 / 2.  # + p3
    label = l1
    # label = '_dfs_R' + str(R2) + '_' + str(R1) + \
    #     '_mx' + format(amplitude, '.1e')
    # label = '_symR' + str(R2) + '_fsR' + str(R1) + \
    #     '_mx' + format(amplitude, '.1e')
    # label = '_dfs_' + str(R0) + '_' + str(R1) + \
    #     '_asymR0_m1_mx' + format(amplitude, '.1e')
    # label = '_asym_m1R' + str(R0) + '_fsR' + str(R1) + \
    #     '_off270_mx1_' + format(A1, '.1e') + '_mx0_' + format(A0, '.1e')

    if 'ARTf' in cams:
        label += '_' + new_type

    # for multiplication with geometry matrices
    phantom1D = phantom.reshape(nFS * nL)  # in W/m^3

    if systematic_error:
        in_label = '_EIM_beta000_sym_dets_aptplane_tilt_cor_' + \
            str(-1.0) + 'deg_sN8_30x20x150_1.35'
        emissivity = LoS3D.store_read_emissivity(
            name=in_label[1:], suffix='3D')[1]

        print('\t\t>> Using files as measurement error:\n\t\t', in_label)
        label += '_tilt' + str(-1.0)

    forward = phantom_forward_integral(
        phantom=phantom1D, in_label=in_label,
        emissivity=emissivity, cams=cams, new_type=new_type,
        debug=debug, save_base=save_base)

    if add_noise:
        print('\t>> Adding noise at level', 100. * error, '%')
        forward, label = noise(
            data=forward, label=label, error=error)

    mfrp.phantom_plot(
        nFS=nFS, nL=nL, phantom=phantom, label=label,
        mesh=mesh['values']['fs']['108.'], save_base=save_base,
        VMID=in_label[1:12].replace('beta', ''), strgrid=strgrid)
    pm.phantom_save(
        label=label, save_base=save_base,
        strgrid=strgrid, phantom=phantom)
    return (forward, label)


def phantom_forward_integral(
        phantom=np.zeros((30 * 100)),  # W/m^3
        emissivity=np.zeros((128, 64, 30, 100)),
        cams=['HBCm', 'VBCl', 'VBCr'],
        in_label='_EIM_beta000_sN8_50x30x100_1.4',
        save_base='../results/INVERSION/MFR/',
        new_type=None,
        debug=False):
    print('\t>> Calculate forward profile with emissivity ...')
    emissivity = mfr_transf.emiss_matrix_transform(
        emissivity=emissivity, debug=debug,
        label=in_label, cams=cams, full=True)
    forward = phantom.dot(emissivity)  # in W, W/m^3 x m^3

    if debug:
        print('phantom1D:', np.shape(phantom), phantom.__class__, '\n',
              'emissivity:', np.shape(emissivity), emissivity.__class__, '\n',
              'forward:', np.shape(forward), forward.__class__)
    return (forward)
