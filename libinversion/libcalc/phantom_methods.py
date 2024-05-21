""" **************************************************************************
    so header """

import numpy as np

import read_calculation_corona as rcc

""" end of header
************************************************************************** """


def spot_pos_mesh(
        nFS=31,
        nL=75,
        sigma=0.6,
        x0=np.zeros((2)),
        amplitude=1.e6,
        position=np.zeros((128, 16, 31, 75, 4)),
        strgrid='31x75',
        base='../results/INVERSION/MESH/',
        save_base='../results/INVERSION/MFR/',
        debug=False):
    phantom = np.zeros((nFS, nL))
    position = position[-1]

    for S in range(nFS):
        for L in range(nL):
            phantom[S, L] = amplitude * gaussian(
                x=np.array([position[S, L, 3], position[S, L, 2]]),
                x0=x0, sigma=sigma)

    label = '_pos_mesh_x' + str(x0[0]) + '_y' + str(
        x0[1]) + '_mx' + format(amplitude, '.1e')
    return (phantom, label)


def STRAHL_to_phantom(
        nFS=31,
        nL=95,
        reff=np.zeros((128, 16, 31, 75)),
        minor_radius=0.513,
        strahl_id='00094',
        strgrid='31x95',
        base='../results/INVERSION/MESH/',
        save_base='../results/INVERSION/MFR/',
        debug=False):
    phantom = np.zeros((nFS, nL))
    strahl_data, strahl_grid = rcc.scale_impurity_radiation(  # in W
        material='C_', strahl_id=strahl_id)[:2]

    strahl_grid *= minor_radius
    for S in range(nFS):
        R_low = np.min(reff[:, :, S, :][
            np.where(reff[:, :, S, :] > .0)])
        R_high = np.max(reff[:, :, S, :][
            np.where(reff[:, :, S, :] > .0)])
        N = np.where((R_low < strahl_grid) &
                     (strahl_grid < R_high))[0]

        if N.size > 0:
            f = np.mean(strahl_data[N])  # in W
            phantom[S, :] = f * 12.  # scale to 20181010.032@3.42s, in W

    label = '_strahl_ID' + strahl_id
    return (phantom, label)


def radial_profile(
        nFS=21 + 10,
        nL=95,
        sigma1=0.4,
        sigma2=0.05,
        amplitude=.5,
        trgt_reff=1.1,
        reff_in_cell=np.zeros((128, 16, 31, 75)),
        minor_radius=0.513,
        strgrid='31x75',
        base='../results/INVERSION/MESH/',
        save_base='../results/INVERSION/MFR/',
        debug=False):
    phantom = np.zeros((nFS, nL))
    amplitude_reff = trgt_reff
    reff = reff_in_cell / minor_radius

    for S in range(nFS):
        R = np.mean(reff[:, :, S, :][
            np.where(reff[:, :, S, :] > .0)])
        sigma = sigma1 if (R < trgt_reff) else sigma2

        if not np.isnan(R):
            phantom[S, :] = amplitude * gaussian(
                x=np.array([R, R]), sigma=sigma,
                x0=np.array([amplitude_reff, amplitude_reff]))

    label = '_Rprof_R' + str(amplitude_reff) + '_sig1' + str(
        sigma1) + '_sig2' + str(sigma2) + '_mx' + format(amplitude, '.1e')
    return (phantom, label)


def fluxsurface_reff_ring(
        nFS=30,
        nL=75,
        trgt_reff=1.1,
        sigma=0.2,
        amplitude=2.0,
        reff_in_cell=np.zeros((128, 16, 31, 75)),
        minor_radius=0.513,
        strgrid='31x75',
        base='../results/INVERSION/MESH/TRIAGS/',
        save_base='../results/INVERSION/MFR/',
        debug=False):
    phantom = np.zeros((nFS, nL))
    amplitude_reff = trgt_reff
    reff_in_cell = reff_in_cell / minor_radius

    for S in range(nFS):
        R = np.mean(reff_in_cell[:, :, S, :][
            np.where(reff_in_cell[:, :, S, :] > .0)])
        if not np.isnan(R):
            phantom[S, :] = amplitude * gaussian(
                x=np.array([R, R]), sigma=sigma,
                x0=np.array([amplitude_reff, amplitude_reff]))

    label = '_ring_R' + str(amplitude_reff) + \
        '_mx' + format(amplitude, '.1e')
    return (phantom, label)


def anisotropic_fluxsurface_ring(
        nFS=21 + 10,
        nL=95,
        sigma=0.2,
        amplitude=2.0,
        mode_number=5,
        offset=0.0,
        trgt_reff=1.1,
        reff_in_cell=np.zeros((128, 16, 31, 75)),
        minor_radius=0.513,
        strgrid='31x75',
        base='../results/INVERSION/MESH/',
        save_base='../results/INVERSION/MFR/',
        symmetric=False,
        debug=False):
    phantom = np.zeros((nFS, nL))
    ring_reff = trgt_reff
    reff = reff_in_cell / minor_radius

    for S in range(nFS):
        R = np.mean(reff[:, :, S, :][
            np.where(reff[:, :, S, :] > .0)])

        for L in range(1, nL + 1):
            if not symmetric:
                f = 1. + np.cos(np.deg2rad(
                    110. + L * mode_number * (360. / nL) + offset))

            else:
                if L < nL / 2:
                    if nL % 2 != 0:
                        f = 1. + np.sin(np.deg2rad(
                            (L + 1) * mode_number * (
                                360. / nL) + offset))
                    else:
                        f = 1. + np.sin(np.deg2rad(
                            L * mode_number * (
                                360. / nL) + offset))
                else:
                    f = 1. + np.sin(np.deg2rad(
                        (nL - L) * mode_number * (
                            360. / nL) + offset))

            if ~np.isnan(R):
                phantom[S, L - 1] = amplitude * f * gaussian(
                    x=np.array([R, R]), sigma=sigma,
                    x0=np.array([ring_reff, ring_reff]))

    if symmetric:
        label = '_sym'
    else:
        label = '_asym'
    label += '_R' + str(ring_reff) + '_m' + str(
        mode_number) + '_mx' + format(amplitude, '.1e')
    return (phantom, label)


def blind_test_ones(
        nFS=30,
        nL=75,
        amplitude=1.0e6,
        strgrid='tN4_30x20x75_1.3',
        save_base='../results/INVERSION/MFR/',
        debug=False):
    phantom = np.ones((nFS, nL)) * amplitude
    label = '_blind_test_ones_mx' + format(amplitude, '.1e')
    return (phantom, label)


def phantom_save(
        label='_magax_dist_x0.0_y0.0_sig0.6_32x51',
        strgrid='32x51',
        save_base='../results/INVERSION/MFR/',
        phantom=np.zeros((32, 51))):
    f = '../results/INVERSION/MFR/' + strgrid + '/phantom'
    for B in [f, save_base + 'phantom']:
        np.savetxt(f + label + '_' + strgrid + '.dat',
                   phantom, fmt='%.6e', delimiter='    ')
    return


def gaussian(
        x=np.zeros((2)),
        x0=np.zeros((2)),
        sigma=.5):
    S = np.sqrt(np.square(
        x[0] - x0[0]) + np.square(x[1] - x0[1]))
    return (
        np.exp(-1.0 * S / (2 * np.square(sigma))))
