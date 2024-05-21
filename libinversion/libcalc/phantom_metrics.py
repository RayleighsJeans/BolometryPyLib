""" **************************************************************************
    so header """

import os
import numpy as np
import json

import mClass

import dat_lists
import prad_calculation as prc

import mfr2D_matrix_gridtransform as mfr_transf
import mfr2D_accessoires as mfra
import mfr_plot as mfrp

import phantom_comparison as phc
import profile_to_lineofsight as profile

Z = np.zeros
ones = np.ones

""" end of header
************************************************************************** """


def load_phantom(
        label='pos_mesh_x5.75_y0.0_mx1.0e+06',
        strgrid='sN8_50x30x100_1.4',
        debug=False):
    phantom = np.loadtxt(
        '../results/INVERSION/MFR/' + strgrid +
        '/phantom_' + label + '_' + strgrid + '.dat')
    if debug:
        print(np.shape(phantom), phantom.__class__)
    return (phantom)


def load_tomogram(
        label='pos_mesh_x5.75_y0.0_mx1.0e+06',
        strgrid='sN8_50x30x100_1.4',
        base='_mfr1D',
        no_ani=4,
        kani=[.1, 1.],
        nigs=1,
        times=0.11,
        add_camera=False,
        new_type=None,
        reduced=True,
        cartesian=False,
        debug=False):
    if not cartesian:
        nr, nt = int(strgrid[:2]), int(strgrid[3:])
    else:
        nr = nt = int(strgrid[:2])

    tomogram, r, z, tomogram_chordal, xp_chordal, tm_hbcm, \
        tm_vbcl, tm_vbcr, tm_artf, chi2, chi_channel, \
        kani_profile, dphi_profile, xp_error = mfra.get_mfr_results(
            program=label, kani=kani, times=times, nigs=nigs,
            no_ani=no_ani, add_camera=add_camera, new_type=new_type,
            grid_nr=nr, grid_nt=nt, base=base, cartesian=cartesian,
            debug=debug, phantom=True, plot=False, reduced=reduced)
    if debug:
        print(np.shape(tomogram), np.shape(r), np.shape(z),
              tomogram.__class__, r.__class__, z.__class__)

    return (tomogram, r, z, tomogram_chordal, xp_chordal,
            tm_hbcm, tm_vbcl, tm_vbcr, tm_artf,
            chi2, chi_channel, kani_profile, phi_profile, xp_error)


def diff_tomo_phantom(
        nr=30,
        nt=100,
        nCore=21,
        r_grid=np.zeros((30, 100, 4)),
        z_grid=np.zeros((30, 100, 4)),
        phantom=np.zeros((30, 100)),
        tomogram=np.zeros((30, 100)),
        debug=False):
    area_method = False
    # MSD = (g_rec - g_phan)^2 / g_phan^2

    msd = np.zeros((np.shape(phantom)))
    area, area_core, area_SOL = 0.0, 0.0, 0.0
    for S in range(nr):
        for L in range(nt):
            msd[S, L] = np.sqrt((tomogram[S, L] - phantom[S, L])**2)

            if area_method:
                [p1, p2, p3, p4] = [
                    np.array([r, z_grid[S, L, i]])
                    for i, r in enumerate(r_grid[S, L, :])]
                A = herons_formula(p1, p2, p3) + \
                    herons_formula(p2, p3, p4)

                area += A  # weighting with pixel area and total
                if S < nCore:
                    area_core += A
                else:
                    area_SOL += A
                msd[S, L] *= A

    if area_method:
        core_msd = np.sum(msd[:nCore, :] / area_core)
        SOL_msd = np.sum(msd[nCore:, :] / area_SOL)
        total_msd = np.sum(msd / area)

        msd *= 100. / area
    else:
        core_msd = np.sum(msd[:nCore, :]) / np.sqrt(np.sum(
            phantom[:nCore, :]**2))
        SOL_msd = np.sum(msd[nCore:, :]) / np.sqrt(np.sum(
            phantom[nCore:, :]**2))
        total_msd = np.sum(msd) / np.sqrt(np.sum(phantom**2))

        msd *= 100. / np.sqrt(np.sum(phantom**2))

    if debug:
        print('core msd:', core_msd, 'SOL msd:', SOL_msd,
              'total msd:', total_msd)
    return (msd, core_msd, SOL_msd, total_msd)  # in % (x 100)


def radial_profile(
        data=np.zeros((30 * 100)),
        radius=np.zeros((128, 30, 100)),
        grid_nr=30,
        grid_nt=100,
        debug=False):
    if debug:
        print('\tRadial profile from MFR 2D reconstruction ...')
    average = np.zeros((2, grid_nr))
    for r in range(grid_nr):
        average[0, r] = np.mean(
            radius[:, r, :][radius[:, r, :] > .0])
        average[1, r] = np.mean(data[r * grid_nt:(r + 1) * grid_nt])
    average_sort = average[:, average[0, :].argsort()]
    return (average_sort)


def forward_integral(
        channels={'none': None},
        data=np.zeros((33 * 51)),
        emissivity=[np.zeros((32, 33 * 51))],
        add_camera=False,
        reduced=True,
        debug=False):
    chord = np.zeros((128))
    if add_camera:
        cams = ['HBCm', 'VBCl', 'VBCr', 'ARTf']
    else:
        cams = ['HBCm', 'VBCl', 'VBCr']

    for n, cam in enumerate(cams):
        if reduced:
            nCh = channels['reduced']['eChannels'][cam]
        else:
            nCh = channels['full']['eChannels'][cam]
        tm2D = emissivity[n]
        forward = tm2D.dot(data)

        if debug:
            print(np.shape(forward), np.shape(nCh))

        for c, ch in enumerate(nCh):
            if False:
                print('2', cam, c, ch)
            chord[ch] = forward[c]
    return (chord)


def mfr_LoS_reff(
        LoS={'none': None},
        factors={'none': None},
        add_camera=False,
        new_type=None,
        debug=False):

    cams = ['HBCm', 'VBCl', 'VBCr']
    if add_camera:
        cams.append('ARTf')
    L, F = LoS['values'], factors['values']
    mag_ax_r = L['magnetic_axis']['x1']
    camera_info = dat_lists.geom_dat_to_json()
    r_min = np.zeros((128))

    for j, cam in enumerate(cams):
        nCh = [ch for ch in camera_info['channels']['eChannels'][cam]]
        for c, ch in enumerate(nCh):

            nZ = np.where(F['line_sections'][ch, :, :] != 0.0)
            if np.shape(nZ)[1] == 0:
                continue
            nN = np.where(L['reff_in_cell'][ch, :, :][nZ] != 0.0)

            if (((cam == 'HBCm') or (add_camera and ((cam == 'ARTf') and (
                    new_type == 'ARTm')))) and (np.mean(L['pos_in_mesh'][
                    ch, :, :, 2][nZ][nN]) < 0.0)):
                f = -1.
            elif ((cam in ['VBCr', 'VBC', 'VBCl']) and (
                    np.mean(L['pos_in_mesh'][
                        ch, :, :, 3][nZ][nN]) < mag_ax_r)):
                f = -1.
            elif (add_camera and ((cam == 'ARTf') and ((
                    new_type == 'ARTh') or (new_type == 'ARTv')))):
                f = -1.
            else:
                f = 1.

            r_min[ch] = f * np.min(L['reff_in_cell'][ch, :, :][nZ][nN])

    return (r_min, L['minor_radius'], L['magnetic_axis'])


def load_LoSp_factors(
        magconf='EIM_beta000',
        strgrid='sN8_50x30x100_1.4',
        debug=False):

    label = magconf + '_' + strgrid
    factors, reff, position, minor_radius = \
        profile.store_read_profile(name=label)

    return (reff, factors)


def profile_radial_and_chord(
        phantom=np.zeros((30 * 100)),
        emiss=[np.zeros((32, 30 * 100))],
        LoS_factors={'none': None},
        geometry_factors={'none': None},
        nr=30,
        nt=100,
        magconf='EIM_beta000',
        strgrid='sN8_50x30x100_1.4',
        add_camera=False,
        new_type=None,
        reduced=True,
        debug=False):

    reff, factors = load_LoSp_factors(
        magconf=magconf, strgrid=strgrid)

    r_min, m_r, mag_ax = mfr_LoS_reff(
        LoS=LoS_factors, factors=geometry_factors,
        add_camera=add_camera, new_type=new_type)

    r_profile = radial_profile(
        data=phantom, radius=reff,
        grid_nr=nr, grid_nt=nt, debug=False)

    chordial_profile = forward_integral(
        data=phantom, channels=mfr_transf.mfr_channel_list(),
        emissivity=emiss, add_camera=add_camera,
        reduced=reduced, debug=debug)

    if debug:
        print(np.shape(r_profile), r_profile.__class__)
        print(np.shape(chordial_profile), chordial_profile.__class__)

    M_r = np.sqrt(mag_ax['x1'] ** 2 + mag_ax['x3'] ** 2)
    return (r_min, chordial_profile, r_profile, m_r, M_r)


def distance(
        p1=np.zeros((2)),
        p2=np.zeros((2))):

    if np.shape(p1)[0] == 2:
        return (np.sqrt(np.square(
            p1[0] - p2[0]) + np.square(p1[1] - p2[1])))

    elif np.shape(p1)[0] == 3:
        return (np.sqrt(np.square(
            p1[0] - p2[0]) + np.square(
                p1[1] - p2[1]) + np.square(p1[2] - p2[2])))


def herons_formula(
        p1=np.zeros((2)),
        p2=np.zeros((2)),
        p3=np.zeros((2))):

    a = distance(p1, p2)
    b = distance(p2, p3)
    c = distance(p3, p1)
    p = (a + b + c) / 2

    return (
        np.sqrt(p * (p - a) * (p - b) * (p - c)))


def power_2D_profile(
        nr=20,
        nt=75,
        minor_radius=0.5139,  # m
        major_radius=5.15,  # m
        profile=np.zeros((20, 75)),
        raw=np.zeros((20, 75)),
        radial_profile=np.zeros((2, 30)),
        r_grid=np.zeros((20, 75, 4)),
        z_grid=np.zeros((20, 75, 4)),
        fs_reff=np.zeros((20)),
        vmec_ID='EIM_000',
        debug=False):
    total, core, neg = .0, .0, .0
    # Let a,b,c be the lengths of the sides of a triangle.
    # The area is given by:
    # A = √p(p - a) * (p − b) * (p − c)
    # where p is half the perimeter, or
    # p = (a + b + c
    if nr is None or nt is None:
        nr, nt = np.shape(profile)

    area = True
    if area:
        neg_map = np.abs(raw.clip(max=.0))

        for S in range(nr):
            for L in range(nt):
                [p1, p2, p3, p4] = [
                    np.array([r, z_grid[S, L, i]])
                    for i, r in enumerate(r_grid[S, L, :])]

                A = herons_formula(p1, p2, p3) + herons_formula(p2, p3, p4)
                total += A * profile[S, L]  # m^2 * W/m^3 = W/m
                neg += A * neg_map[S, L]  # m^2 * W/m^3 = W/m

                if fs_reff[S] <= minor_radius:
                    core += A * profile[S, L]

                if debug:
                    print('area:', A)
        # ca 1.8m^2
        total = total * 2 * np.pi * major_radius  # W/m * m = W
        core = core * 2 * np.pi * major_radius  # W/m * m = W
        neg = neg * 2 * np.pi * major_radius

    else:
        # simple torus volume V = 2 * pi * major_r * pi * minor_r^2
        # volume of torus shell
        # V_out - V_in = 2 * pi * major_r * pi * (minor_out^2 - minor_in^2)

        for L in range(1, np.shape(radial_profile)[1]):
            r_in = radial_profile[0, L - 1]  # m
            r_out = radial_profile[0, L]  # m
            dV = 2 * np.pi * major_radius * np.pi * (r_out**2 - r_in**2)  # m^3
            dP = dV * radial_profile[1, L]  # m^3 * W/m^3 = W

            if debug:
                print('nL:', format(L, '.4f'),
                      ' dr:', format(dV, '.4f'),
                      ' dP:', format(dP, '.4f'))

            if fs_reff[L] <= minor_radius:
                core += dP
            total += dP

    if debug:
        print('total:', total, 'W')
        print('core:', core, 'W')
        print('neg:', neg, 'W')
    return (total, core, neg)


def fast_prad_chordal(
        chordal=np.zeros((128)),
        magconf='EIM_beta000',
        strgrid='sN8_30x20x150_1.35',
        reduced=True,
        debug=False):
    cams = ['HBCm', 'VBC', 'VBCr', 'VBCl']
    geometry = dat_lists.geom_dat_to_json(
        geom_input='self', strgrid=strgrid, magconf=magconf)
    volume_torus = 45.  # m^3

    p_rads = {}
    channels = mfr_transf.mfr_channel_list()
    for n, cam in enumerate(cams):
        if reduced:
            nCh = channels['reduced']['eChannels'][cam]
        else:
            nCh = channels['full']['eChannels'][cam]

        if debug:
            print(np.shape(nCh), np.shape(chordal),
                  np.shape(geometry['geometry']['vbolo']),
                  np.shape(geometry['geometry']['kbolo']))

        # take OG prad routine and calculate it from chordal results
        # of both tomogram and phantom
        P_rad, volume_sum = prc.calculate_prad(
            time=np.array([0.0, 1.0]), volume=geometry['geometry']['vbolo'],
            k_bolo=geometry['geometry']['kbolo'], volume_torus=volume_torus,
            channels=nCh, camera_list=nCh, date=None, shotno=None,
            brk_chan=geometry['channels']['droplist'], camera_tag=cam,
            saving=False, method=None, debug=False,
            power=np.array([chordal, chordal]).transpose())

        if False:
            print(cam, '>> p_rad:', format(P_rad[0], '.3f'),
                  'W; v_sum:', format(volume_sum, '.3f'), 'm^3')
        p_rads[cam] = P_rad[0]
    return (p_rads)


def pearson_covariance(
        tomogram1D=np.zeros((30 * 100)),
        phantom1D=np.zeros((30 * 100)),
        debug=False):
    # Pearson correlation coefficient, ρ c , which can
    # measure the correlation between two vectors. In this con-
    # text, it is defined as the covariance of the two emission source
    # vectors, inversion x inv and phantom solution x sol , divided by
    # the product of their standard deviations,
    # ρ = Cov(x inv, x sol) / [σ(x inv) * σ(x sol)]

    # Cov (X,Y) = E[ (X - E(X)) * (Y - E(Y)) ]
    # E(X) = sum X_i * p_i = sum X_i * P(X=X_i)
    # p_i = 1. / (N) for all
    # E(X) = 1 / N * sum X_i
    # Cov (X, Y) = 1 / N * sum [(X_i - µ_x) * (Y_i - µ_y)]

    if True:  # normalization
        t1D = tomogram1D / np.max(phantom1D)
        p1D = phantom1D / np.max(phantom1D)

    P_c = (np.cov(t1D, p1D)[0, 1] /
           (np.std(t1D) * np.std(p1D)))
    if debug:
        print('\tcov(tomo, phant):', np.cov(t1D, p1D)[0, 1],
              '\n\tcov(tomo):', np.cov(t1D),
              '\n\tcov(phan):', np.cov(p1D),
              '\n\tstd(tomo):', np.std(t1D),
              '\n\tstd(phan):', np.std(p1D),
              '\n\tp_c:', P_c)
    return (P_c)


def compare_tomo_phantom(
        nFS=30,
        nL=100,
        N=8,
        no_ani=4,
        kani=[.1, 1.],
        nVals=[20, 1],
        nigs=1,
        times=0.11,
        label='pos_mesh_x5.75_y0.0_mx1.0e+06',
        strgrid='sN8_50x30x100_1.4',
        base='_mfr1D',
        vmec_ID='EIM_000',
        magconf='EIM_beta000',
        RGS=False,
        add_camera=False,
        new_type=None,
        reduced=False,
        plot=False,
        saving=False,
        debug=False):

    L = label
    if RGS:
        L += '_RGS'
    if (no_ani == 1):
        L += '_no_ani'
    elif (no_ani == 0):
        L += '_kani' + str(kani[0]) + '_' + str(
            kani[1])
    elif (no_ani in [2, 3, 4, ]):
        L += '_aniM' + str(no_ani) + '_' + str(
            kani[0]) + '_' + str(kani[1])
        if (nVals is not None) and (no_ani == 4):
            L += '_nT' + str(nVals[0]) + '_nW' + str(nVals[1])

    if add_camera:
        L += '_ARTf'
    L += '_nigs' + str(nigs) + '_times' + str(times)
    if reduced:
        L += '_reduced'

    file = '../results/INVERSION/MFR/' + strgrid + '/' + \
        'comp_phantom_tomogram_' + L + '.json'

    # if os.path.isfile(file):
    #     with open(file, 'r') as infile:
    #         indict = json.load(infile)
    #     infile.close()
    #     data = mClass.dict_transf(indict, to_list=False)
    #     mfrp.tomogram_phantom_wrapper(
    #         data_object=data, strgrid=strgrid,
    #         add_camera=add_camera)
    #     return (data)
    # else:
    #     pass

    data = {'label': L, 'values': {}}
    d = data['values']
    nr, nt = nFS, nL

    tomo, r, z, fs_reff, LoS_reff, m_r, M_R, tomo_chordal, \
        phantom_chordal, tomo_radial, pRt, pHWt, pIt, \
        HWt, tomo_total, tomo_core, fPrad_tomo, fPrad_phantom, \
        xp_error, kani_profile, dphi_profile, chi2, \
        tm_hbcm, tm_vbcl, tm_vbcr, tm_artf, labelt, \
        base, nigs_real, mfr2D = mfra.get_mfr_results(
            program=label, kani=kani, times=times, nigs=nigs, nVals=nVals,
            no_ani=no_ani, RGS=RGS, add_camera=add_camera, 
            new_type=new_type, grid_nr=nr, grid_nt=nt, strgrid=strgrid,
            debug=debug, phantom=True, plot=False,
            reduced=reduced, base=base)

    tomo1D = tomo.reshape(np.shape(tomo)[0] * np.shape(tomo)[1])
    phantom = load_phantom(strgrid=strgrid, label=label)

    phantom1D = phantom.reshape(np.shape(tomo)[0] * np.shape(tomo)[1])

    phantom_radial = mfra.avg_radial_profile(
        data=phantom1D, fs_radius=fs_reff,
        grid_nr=nr, grid_nt=nt, debug=debug)
    pRp, peaks_HW, pIp, HWp = phc.fwhm_radial_profiles(
        profile=phantom_radial)

    msd = diff_tomo_phantom(
        nr=nr, nt=nt, r_grid=r, z_grid=z,
        phantom=phantom, tomogram=tomo)[0]

    p_c = pearson_covariance(
        tomogram1D=tomo1D, phantom1D=phantom1D, debug=debug)

    phantom_total, phantom_core, error2D = power_2D_profile(
        profile=phantom, radial_profile=phantom_radial,
        r_grid=r, z_grid=z, vmec_ID=vmec_ID,
        fs_reff=fs_reff, minor_radius=m_r, raw=mfr2D,
        nr=nr, nt=nt, major_radius=M_R, debug=debug)

    if debug:
        print('tomo:', np.shape(tomo), '\n',
              'tomo1D:', np.shape(tomo1D), '\n',
              'tomo chordal:', np.shape(tomo_chordal),
              tomo_chordal.__class__, '\n',
              'phantom chordal:', np.shape(phantom_chordal),
              phantom_chordal.__class__, '\n',
              'tomo radial:', np.shape(tomo_radial), '\n',
              'phantom radial:', np.shape(tomo_radial))

    # basic info on the results and phantom image
    d['phantom'] = phantom
    d['tomogram'] = tomo
    d['orig_tomogram'] = mfr2D
    d['r_grid'] = r
    d['z_grid'] = z
    d['nigs'] = nigs
    d['kani'] = kani
    d['kani_profile'] = kani_profile
    d['dphi_profile'] = dphi_profile
    d['times'] = times
    d['minor_radius'] = m_r
    d['major_radius'] = M_R
    d['VMEC_ID'] = vmec_ID

    # difference between tomogram and phantom image
    d['difference'] = {}
    d['difference']['mean_square_deviation'] = msd
    d['difference']['pearson_coefficient'] = p_c
    d['difference']['chi2'] = chi2

    # radial and chordial profile from both
    # the tomogram and phantom image
    d['profiles'] = {}
    d['profiles']['radial_tomogram'] = tomo_radial
    # np.array([tomo_radial[0], tomo_radial[1]])
    d['profiles']['chordal_tomogram'] = tomo_chordal
    d['profiles']['radial_phantom'] = phantom_radial
    d['profiles']['chordal_phantom'] = phantom_chordal

    # errorbars
    d['profiles']['radial_error'] = tomo_radial[0, 1] - tomo_radial[0, 0]
    d['profiles']['2D_error'] = error2D
    d['profiles']['abs_error'] = np.abs(np.min(mfr2D)) / 2.
    d['profiles']['xp_error'] = xp_error

    d['peaks'] = {}
    d['peaks']['half_widths'] = {}
    d['peaks']['half_widths']['phantom'] = HWp
    d['peaks']['half_widths']['tomogram'] = HWt
    d['peaks']['radial_pos'] = {}
    d['peaks']['radial_pos']['phantom'] = pRp
    d['peaks']['radial_pos']['tomogram'] = pRt
    d['peaks']['index_pos'] = {}
    d['peaks']['index_pos']['phantom'] = pIp
    d['peaks']['index_pos']['tomogram'] = pIt

    # total radiated power from profiles
    d['P_rad'] = {}
    d['P_rad']['tomogram'] = fPrad_tomo
    d['P_rad']['phantom'] = fPrad_phantom
    d['total_power'] = {}
    d['total_power']['phantom'] = phantom_total
    d['total_power']['tomogram'] = tomo_total
    d['core_power'] = {}
    d['core_power']['phantom'] = phantom_core
    d['core_power']['tomogram'] = tomo_core

    if plot:
        mfrp.tomogram_phantom_wrapper(
            data_object=data, strgrid=strgrid,
            debug=debug, add_camera=add_camera)

    if saving:
        print('\t\tWriting to', file)
        outdict = mClass.dict_transf(data, to_list=True)
        with open(file, 'w') as outfile:
            json.dump(outdict, outfile, indent=4, sort_keys=False)
        outfile.close()
        data = mClass.dict_transf(outdict, to_list=False)

    return (data)
