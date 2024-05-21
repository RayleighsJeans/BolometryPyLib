""" **************************************************************************
    so header """

import numpy as np
# import json

# import mClass
import main as prad_main

import dat_lists as dat_lists

import mesh_2D as mesh2D
import LoS_emissivity3D as LoS3D
import LoS_emissivity2D as LoS2D
import LoS_volume as LoSv

import profile_to_lineofsight as profile
import profile_to_lint as profile_LoS

import factors_geometry_plots as fgp
import mfr_plot as mfrp

Z = np.zeros
ones = np.ones

""" end of header
************************************************************************** """


def prepare_geometry_power(
        program='20181010.032',
        label='_EIM_beta000_tN8_50x30x100_1.4',
        strgrid='tN8_50x30x100_1.4',
        cams=['HBCm', 'VBCr', 'VBCl'],
        new_type=None,
        debug=False):
    save_base = '../results/INVERSION/MFR/' + strgrid + '/'
    mag_config = label[1:12]

    mesh = mesh2D.store_read_mesh2D(name=label[1:])
    # lines2D = LoS2D.store_read_line_sections(name=label[1:])
    # emiss2D = LoS3D.store_read_emissivity(name=label[1:], suffix='2D')[1]
    lines3D = LoS3D.store_read_line_sections(name=label[1:])
    emiss3D = LoS3D.store_read_emissivity(name=label[1:], suffix='3D')[1]
    volume = LoSv.store_read_volume(name=label[1:], typ='')
    reff, position, minor_radius = profile.store_read_profile(name=label[1:])
    N, nFS, nL = np.shape(emiss3D)[1:]

    # emiss_matrix_transform(
    #     emissivity=emiss2D, save_base=save_base,
    #     suffix='2D', debug=debug, label=label)
    emiss_matrix_transform(
        emissivity=emiss3D, save_base=save_base,
        cams=cams, new_type=new_type,
        suffix='3D', debug=debug, label=label)
    grid_matrix_transform(  # grid matrix positions
        mesh=mesh, reff=reff, label=label, strgrid=strgrid,
        debug=debug, save_base=save_base)
    # kfac_volume_transform(  # k factor and volume of channel
    #     emissivity=emiss2D, volume=volume, suffix='2D',
    #     label=label, save_base=save_base, debug=False)
    kfac_volume_transform(  # k factor and volume of channel
        emissivity=emiss3D, volume=volume, suffix='3D',
        cams=cams, new_type=new_type,
        label=label, save_base=save_base, debug=False)
    effective_radius_LoS_transform(
        label=label, save_base=save_base, reff=reff,
        lines=lines3D, m_R=minor_radius, position=position,
        cams=cams, new_type=new_type,
        emissivity=emiss3D, debug=debug,
        mag_ax=mesh['values']['fs']['108.'][:, 0, 0])

    if program == 'phantom':
        return  # end
    ch_pow_mfr(
        program=program, strgrid=strgrid,
        mag_config=mag_config, debug=False)
    return


def reverse_refactor_matrix(
        matrix=Z((16, 31, 51)),
        n1=33,
        n2=51,
        debug=False):
    reform = Z((n1 * n2))
    if debug:
        print(n1, n2, np.shape(matrix))

    for i in range(n1):
        for j in range(n2):
            if debug:
                print((i * n2) + j)
            reform[(i * n2) + j] = np.sum(matrix[:, i, j])

    return (reform)


def emiss_matrix_transform(
        emissivity=np.zeros((128, 64, 30, 100)),
        cams=['HBCm', 'VBCl', 'VBCr'],
        label='_EIM_beta000_sN8_50x30x100_1.4',
        save_base='../results/INVERSION/MFR/',
        suffix='3D',
        new_type=None,
        full=False,
        debug=False):
    mag_config = label[1:12] + '_' + suffix
    camera_info = dat_lists.geom_dat_to_json()

    n1, n2 = np.shape(emissivity)[2:]  # nFS, nL
    if full:
        reform_emiss = np.zeros((n1 * n2, 128))

    for c, cam in enumerate(cams):
        nCh = [ch for ch in camera_info['channels']['eChannels'][cam]]

        if not full:
            reform_emiss = Z((n1 * n2, np.shape(nCh)[0]))
        for n, ch in enumerate(nCh):
            if False:  # debug:
                print(n, '/', np.shape(nCh)[0])

            if not full:
                reform_emiss[:, n] = reverse_refactor_matrix(  # m^3
                    matrix=emissivity[ch, :, :, :], n1=n1, n2=n2)
            else:
                reform_emiss[:, ch] = reverse_refactor_matrix(  # m^3
                    matrix=emissivity[ch, :, :, :], n1=n1, n2=n2)

        if not full:
            if cam == 'ARTf':
                cam = new_type
            np.savetxt(
                save_base + mag_config + '_cam' + cam + '_tm2D.dat',
                reform_emiss, delimiter='    ', fmt='%.6e', comments='')

    if debug:
        og_emiss = np.loadtxt(
            '../../bolometer_mfr/standard/14x100/standard_camHBCm_tm2D.dat')
        print('og_emiss:', np.shape(og_emiss), 'reform_emiss:',
              np.shape(reform_emiss))

    if full:
        return (reform_emiss)  # in m^3
    else:
        return


def reverse_grid_transform(
        cartesian=False,
        mesh=Z((2, 30, 100)),
        n1=32,
        n2=50,
        debug=False):

    if debug:
        print(n1, n2, np.shape(mesh))

    grid = Z((n1 * n2, 8))
    for i in range(n1):
        for j in range(n2):
            if debug:
                print((i * n2) + j)

            if not cartesian:
                p1, p2, p3, p4 = fgp.poly_from_mesh(
                    m=mesh, S=i, L=j, nL=n2, nFS=n1)
            elif cartesian:
                p1, p2, p3, p4 = fgp.square_from_mesh(
                    m=mesh, S=i, L=j, nL=n2, nFS=n1)

            grid[(i * n2) + j, :] = [
                p1[0], p1[1], p2[0], p2[1],
                p3[0], p3[1], p4[0], p4[1]]

    return (grid)


def mean_effective_radius_FS(
        reffs=Z((30, 100)),
        nFS=31,
        debug=False):
    means = Z((nFS))
    for S in range(nFS):
        means[S] = np.mean(
            reffs[S, :][
                np.where(reffs[S, :] != 0.0)])
    return (means)


def grid_matrix_transform(
        mesh={'108.': np.zeros((2, 30, 100))},
        reff=np.zeros((128, 64, 30, 100)),
        label='_EIM_beta000_sN8_50x30x100_1.4',
        strgrid='sN8_50x30x100_1.4',
        save_base='../results/INVERSION/MFR/',
        debug=False):
    mag_config = label[1:12]
    n0, n1, n2 = np.shape(reff)[1:]
    n0 = int(np.sqrt(n0))

    reform_grid = reverse_grid_transform(
        n1=n1, n2=n2, debug=debug,
        mesh=mesh['values']['fs']['108.'])
    mr_eff = mean_effective_radius_FS(
        reffs=reff[-1, 0], nFS=n1)

    header = str(mr_eff).replace('[', '').replace(
        ']', '').replace(',', '').replace('\n', '') + '\n' + \
        str(n1 - 2) + '    ' + '2' + '    ' + str(n2 + 1)  # + '\n'

    # standard_grids2D_14x100_phi108.dat
    f = save_base + mag_config + '_grids2D_' + strgrid + '.dat'
    np.savetxt(
        f, reform_grid, delimiter='    ',
        header=header, fmt='%.6f', comments='')
    if debug:
        og_grid = np.loadtxt(
            '../../bolometer_mfr/tmw7x/' +
            'grids2D/standard_grids2D_14x100_phi108.dat',
            skiprows=2)
        print('og_grid:', np.shape(og_grid), 'reform_emiss:',
              np.shape(reform_grid))

    return


def kfac_volume_transform(
        emissivity=np.zeros((128, 64, 30, 100)),
        volume=np.zeros((128, 2)),
        cams=['HBCm', 'VBCl', 'VBCr'],
        label='_EIM_beta000_sN8_50x30x100_1.4',
        save_base='../results/INVERSION/MFR/',
        new_type=None,
        suffix='3D',
        saving=True,
        debug=False):
    mag_config = label[1:12] + '_' + suffix
    camera_info = dat_lists.geom_dat_to_json()

    kbolo = np.zeros((128))
    for n, cam in enumerate(cams):
        nCh = [ch for ch in camera_info['channels']['eChannels'][cam]]

        if saving:
            A = np.zeros((np.shape(nCh)[0], 2))
        for c, ch in enumerate(nCh):
            if saving:
                A[c, 0] = kbolo[ch] = np.sum(emissivity[ch])
                A[c, 1] = np.sum(volume[ch])

        if saving:
            if cam == 'ARTf':
                cam = new_type
            np.savetxt(
                save_base + mag_config + '_cam' + cam +
                '_kbolott_and_volume_PV_pih.dat',
                A, delimiter='\t', fmt='%.6e')

    if not saving:
        return (kbolo, volume)
    else:
        return


def effective_radius_LoS_transform(
        reff=np.zeros((128, 64, 30, 100)),
        position=np.zeros((128, 64, 30, 100, 4)),
        lines=np.zeros((128, 64, 30, 100)),
        emissivity=np.zeros((128, 64, 30, 100)),
        mag_ax=np.zeros((2)),
        m_R=0.513,
        cams=['HBCm', 'VBCr', 'VBCl'],
        label='_EIM_beta000_tN4_50x30x125_1.4',
        save_base='../results/INVERSION/MFR/',
        new_type=None,
        saving=True,
        debug=False):
    mag_config = label[1:12]
    camera_info = dat_lists.geom_dat_to_json()
    n0, n1, n2 = np.shape(reff)[1:]

    R = profile_LoS.reff_of_LOS(
        camera_info=camera_info, cams=cams, reffs=reff,
        position=position, lines=lines, emissivity=emissivity,
        mag_ax=mag_ax, new_type=new_type)['minimum']

    if not saving:
        reff, rho = Z((128)), Z((128))
    header = '# channel number    reff m    roh reff/a'
    for c, cam in enumerate(cams):
        nCh = [ch for ch in camera_info['channels']['eChannels'][cam]]
        #     if ch not in camera_info['channels']['droplist']]

        if saving:
            reff_LoS = Z((np.shape(nCh)[0], 3))
        for n, ch in enumerate(nCh):
            if saving:
                reff_LoS[n, 0] = n + 1
                reff_LoS[n, 1] = R[ch]
                reff_LoS[n, 2] = R[ch] / m_R

            if not saving:
                reff[ch] = R[ch]
                rho[ch] = R[ch] / m_R

        if saving:
            if cam == 'ARTf':
                cam = new_type
            # print(save_base + mag_config + '_cam' + cam + '_reffLoS.dat')
            np.savetxt(
                save_base + mag_config + '_cam' + cam + '_reffLoS.dat',
                reff_LoS, header=header, comments='', delimiter='    ',
                fmt='%.7f')

    if not saving:
        return (reff, rho)

    else:
        return


def ch_pow_mfr(
        program='20181010.032',
        strgrid='sN8_50x30x100_1.4',
        mag_config='EIM_beta000',
        save_base='../results/INVERSION/MFR/',
        debug=False):
    camera_info = dat_lists.geom_dat_to_json()
    power_object = prad_main.main(
        program=program, return_pow=True,
        filter_method='mean', plot=False,
        geom_input='self',
        magconf=mag_config, strgrid=strgrid)

    time = power_object['time']  # in s
    channel_power = power_object['power']  # in W
    power_stddiv = power_object['power_stddiv'] + 1.5e-6  # in W
    # at U_d= 4.000e-04 V the power P= 9.599e-05 W
    # s= 1.599e-06 W  dpower= -1.787e-08 W

    print('\t>> saving: ' + program)
    for c, cam in enumerate(['HBCm', 'VBCr', 'VBCl']):
        nCh = [ch for ch in camera_info['channels']['eChannels'][cam]]

        N, M = np.shape(time)[0], np.shape(nCh)[0]
        output = Z((N, 1 + M + M))

        output[:, 0] = time
        for n, ch in enumerate(nCh):
            output[:, 1 + n] = channel_power[ch]  # in W
            output[:, 1 + M + n] = power_stddiv[ch]  # in W

        f = program + '_' + mag_config + '_' + strgrid + \
            '_chnpow_' + cam + '.dat'
        print('\t\t>> store at: ' + '../../bolometer_mfr/chpow/' +
              program + '/' + f)
        for file in [save_base + strgrid + '/',
                     '../../bolometer_mfr/chpow/' + program + '/']:
            np.savetxt(
                file + f, output, comments='',
                delimiter='    ', fmt='%.7e')

    if program == '20181010.032':
        trgt = 3.42
    elif program == '20180725.044':
        trgt = 3.05
    else:
        trgt = 1.0

    mfrp.pow_forward_plot(
        cams=['HBCm', 'VBCl', 'VBCr'], camera_info=camera_info,
        power=channel_power, error=power_stddiv, target=trgt,
        time=time, debug=debug, program=program)
    return


def check_geometry_grid_emiss(
        base='../results/INVERSION/MFR/',
        nr=30,
        nt=100,
        tm2D='EIM_beta000_3D_camHBCm_tm2D.dat',
        grid='EIM_beta000_grids2D_sN8_50x30x100_1.4.dat',
        debug=False):
    f = base + str(nr) + 'x' + str(nt)
    f_tm = f + '/' + tm2D
    f_grid = f + '/' + grid

    grd = np.loadtxt(f_grid, skiprows=2)
    tm = np.loadtxt(f_tm)

    if debug:
        print(np.shape(tm), np.shape(grd))
    mfrp.check_reverse_plot(
        save_base=f + '/', grid=grd, emiss=tm)

    return


def mfr_channel_list():
    channels = {
        'full': {
            'eChannels': {
                'HBCm': [
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                    28, 29, 30, 31],
                'nHBCm': 32,
                'VBCr': [
                    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
                    78, 79, 80, 81, 82, 83, 84, 85, 86],
                'nVBCr': 22,
                'VBCl': [
                    48, 50, 51, 52,
                    54, 56, 57, 58, 59, 60, 61, 62, 63],
                'nVBCl': 13,
                'VBC': [
                    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
                    79, 80, 81, 82, 83, 84, 85, 86, 87, 54, 55, 56, 57, 58, 59,
                    60, 61, 62, 63],
                'nVBC': 34,
                'ARTf': [
                    113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
                    123, 124, 125, 126, 127],
                'nARTf': 15
            }, 'gChannels': {
                'HBCm': [
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                    18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                    28, 29, 30, 31, 32],
                'VBCr': [
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15,
                    16, 17, 18, 19, 20, 21, 22, 23],
                'VBCl': [
                    9, 11, 12, 13,
                    15, 17, 18, 19, 20, 21, 22, 23, 24],
                'VBC': [
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                    33, 34],
                'ARTf': [
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            }}, 'reduced': {'eChannels': {
                'HBCm': [
                    0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                    16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                'nHBCm': 30,
                'VBCr': [
                    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                    76, 78, 79, 80, 81, 82, 83, 84, 85, 86],
                'nVBCr': 22,
                'VBCl': [
                    52, 54, 56, 61, 62, 63],
                'nVBCl': 6,
                'VBC': [
                    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
                    75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86,
                    54, 56, 61, 62, 63],
                'nVBC': 27,
                'ARTf': [
                    113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
                    123, 124, 125, 126, 127],
                'nARTf': 15
            }, 'gChannels': {
                'HBCm': [
                    1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                'VBCr': [
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16,
                    17, 18, 19, 20, 21, 22, 23],
                'VBCl': [
                    13, 15, 17, 22, 23, 24],
                'VCB': [
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17,
                    18, 19, 20, 21, 22, 23, 24, 26, 32, 33, 34],
                'ARTf': [
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}}}

    return (channels)
