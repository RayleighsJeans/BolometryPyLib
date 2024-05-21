""" **************************************************************************
    so header """

import numpy as np
from numpy.core.fromnumeric import shape
import requests as req
import json

import mClass
import database_inv as database
import mfr_plot as mfrp
import phantom_metrics as ph_m
import mfr2D_matrix_gridtransform as mfr_transf
import phantom_comparison as phc

Z = np.zeros
ones = np.ones

base_loc = '//share.ipp-hgw.mpg.de/' + \
    'documents/pih/documents/git/bolometer_mfr/'
location = '//share.ipp-hgw.mpg.de/' + \
    'documents/pih/documents/git/bolometer_mfr/output'

Z = np.zeros

""" end of header
************************************************************************** """


def load_dphi_grad(
        program='20181010.032',
        strgrid='tN4_30x20x75_1.35',
        fs_reff=np.zeros((20)),
        kani=[1.5, 0.8],
        nVals=[21, 6],
        nigs=1,
        no_ani=0,
        RGS=False, reduced=True,
        add_camera=False,
        phantom=False,
        debug=False):
    format1 = '.2f' if kani[0] < 1.e1 else '.1f'
    format2 = '.2f' if kani[1] < 1.e1 else '.1f'

    name = location + '/'
    if phantom:
        name += 'phantom/'
    name += program + '/' + strgrid
    if not phantom:
        name += '/' + program
    else:
        name += '/' + 'phan_'

    if RGS:
        name += 'RGS_'
    if (no_ani == 1):
        name += 'no_ani'
    elif (no_ani == 0):
        name += 'kani_' + format(kani[0], format1) + '_' + format(
            kani[1], format2)
    elif (no_ani in [2, 3, 4]):
        name += 'aniM' + str(no_ani) + '_' + format(
            kani[0], format1) + '_' + format(kani[1], format2)
        if (nVals is not None) and (no_ani == 4):
            name += '_nT' + str(nVals[0]) + '_nW' + str(nVals[1])
        elif (nVals is not None) and (no_ani == 3):
            name += '_nT' + str(nVals)
    # if RGS:
    #     name += '_RGS'

    if reduced:
        name += '_reduced'
    if add_camera:
        name += '_artf'
    name += '_nigs' + str(nigs) + '_' + strgrid

    loaded = np.loadtxt(name + '_dphi_gradient.dat')
    dphi_profile = np.zeros((4, np.shape(loaded)[0]))

    for S in range(np.shape(loaded)[0]):
        dphi_profile[0, S] = fs_reff[S]    # radius
        dphi_profile[1, S] = loaded[S, 0]  # dX
        dphi_profile[2, S] = loaded[S, 1]  # orig dY
        dphi_profile[3, S] = loaded[S, 2]  # weighted dY
    if debug:
        print('dphi_profile:', np.shape(dphi_profile),
              dphi_profile.__class__)
    return (dphi_profile)


def load_kani_output(
        program='20181010.032',
        strgrid='31x75x4',
        kani=[1.5, 0.8],
        nVals=[21, 6],
        times=3.42,
        nigs=5,
        nr=14,
        no_ani=0,
        RGS=False,
        reduced=True,
        add_camera=False,
        phantom=False,
        debug=False):
    format1 = '.2f' if kani[0] < 1.e1 else '.1f'
    format2 = '.2f' if kani[1] < 1.e1 else '.1f'

    name = location + '/'
    if phantom:
        name += 'phantom/'
    name += program + '/' + strgrid
    if not phantom:
        name += '/' + program
    else:
        name += '/' + 'phan_'

    if RGS:
        name += 'RGS_'
    if (no_ani == 1):
        name += 'no_ani'
    elif (no_ani == 0):
        name += 'kani_' + format(kani[0], format1) + '_' + format(
            kani[1], format2)
    elif (no_ani in [2, 3, 4]):
        name += 'aniM' + str(no_ani) + '_' + format(
            kani[0], format1) + '_' + format(kani[1], format2)
        if (nVals is not None) and (no_ani == 4):
            name += '_nT' + str(nVals[0]) + '_nW' + str(nVals[1])
        elif (nVals is not None) and (no_ani == 3):
            name += '_nT' + str(nVals)
    # if RGS:
    #     name += '_RGS'

    if reduced:
        name += '_reduced'
    if add_camera:
        name += '_artf'
    name += '_nigs' + str(nigs) + '_' + strgrid

    try:
        kani_profile = np.loadtxt(name + '_' + format(
            times, '.2f') + '_kani_profile.dat').transpose()
    except Exception:
        kani_profile = np.loadtxt(
            name + '__kani_profile.dat').transpose()

    kani_profile = kani_profile.reshape(-1, nr)

    if kani_profile[0, 0] != .0:
        kani_profile = np.insert(
            kani_profile, 0, [.0, kani_profile[1, 0]],
            axis=1)

    if debug:
        print('kani_profile:', np.shape(kani_profile),
              kani_profile.__class__)
    return (kani_profile)


def load_error_input(
        program='20181010.032',
        strgrid='31x75x4',
        kani=[1.5, 0.8],
        nVals=[21, 6],
        times=3.42,
        nigs=5,
        nr=14,
        nt=100,
        no_ani=0,
        RGS=False,
        reduced=True,
        add_camera=False,
        phantom=False,
        debug=False):
    format1 = '.2f' if kani[0] < 1.e1 else '.1f'
    format2 = '.2f' if kani[1] < 1.e1 else '.1f'

    name = location + '/'
    if phantom:
        name += 'phantom/'
    name += program + '/' + strgrid
    if not phantom:
        name += '/' + program
    else:
        name += '/' + 'phan_'

    if RGS:
        name += 'RGS_'
    if (no_ani == 1):
        name += 'no_ani'
    elif (no_ani == 0):
        name += 'kani_' + format(kani[0], format1) + '_' + format(
            kani[1], format2)
    elif (no_ani in [2, 3, 4]):
        name += 'aniM' + str(no_ani) + '_' + format(
            kani[0], format1) + '_' + format(kani[1], format2)
        if (nVals is not None) and (no_ani == 4):
            name += '_nT' + str(nVals[0]) + '_nW' + str(nVals[1])
        elif (nVals is not None) and (no_ani == 3):
            name += '_nT' + str(nVals)
    # if RGS:
    #     name += '_RGS'

    if reduced:
        name += '_reduced'
    if add_camera:
        name += '_artf'
    name += '_nigs' + str(nigs) + '_' + strgrid
    name += '_' + format(times, '.2f')

    dfur0 = np.loadtxt(name + '_df_ur0.dat')
    if debug:
        print('\t\tdfur0:', np.shape(dfur0), dfur0.__class__)

    return (dfur0)


def reform_error_input(
        program='20181010.032',
        strgrid='31x75x4',
        kani=[1.5, 0.8],
        nVals=[21, 6],
        times=3.42,
        nigs=5,
        nr=14,
        nt=100,
        no_ani=0,
        RGS=False,
        add_camera=False,
        reduced=True,
        phantom=False,
        debug=False):

    channels = mfr_transf.mfr_channel_list()
    dfur0 = load_error_input(
        program=program, kani=kani, nigs=nigs, phantom=phantom,
        nr=nr, nt=nt, debug=debug, reduced=reduced, add_camera=add_camera,
        RGS=RGS, times=times, no_ani=no_ani, nVals=nVals, strgrid=strgrid)

    if reduced:
        nCh = channels['reduced']['eChannels']
    else:
        nCh = channels['full']['eChannels']
    nhbcm, nvbcl, nvbcr, nartf = nCh['nHBCm'], nCh['nVBCl'], \
        nCh['nVBCr'], nCh['nARTf']

    if add_camera:
        cams = ['HBCm', 'VBCl', 'VBCr', 'ARTf']
    else:
        cams = ['HBCm', 'VBCl', 'VBCr']

    if reduced:
        if ((np.shape(dfur0)[0] < 58
             ) or ('ARTf' in cams and np.shape(dfur0)[0] < 82)):
            # 1 32 // EIM_000
            # nhbcm = nhbcm - 2
            # 0, 1, 2, 28, 29, 30, 31 //  KJM_027
            nhbcm = nhbcm - 7
    else:
        if ((np.shape(dfur0)[0] < 67
             ) or ('ARTf' in cams and np.shape(dfur0)[0] < 82)):
            # 1 32 // EIM_000
            # nhbcm = nhbcm - 2
            # 0, 1, 2, 28, 29, 30, 31 //  KJM_027
            nhbcm = nhbcm - 7

    xp_error = np.zeros((128))
    df_hbcm = dfur0[:nhbcm]
    df_vbcr = dfur0[nhbcm:nhbcm + nvbcr]
    df_vbcl = dfur0[nhbcm + nvbcr:nhbcm + nvbcr + nvbcl]

    if add_camera:
        df_artf = dfur0[nhbcm + nvbcl + nvbcr:]
    else:
        df_artf = Z((nartf))

    if debug:
        print('\t\tdf_hbcm', np.shape(df_hbcm), nhbcm, '\n',
              '\t\tdf_vbcl', np.shape(df_vbcl), nvbcl, '\n',
              '\t\tdf_vbcr', np.shape(df_vbcr), nvbcr)
        if add_camera:
            print('\t\tdf_artf', np.shape(df_artf), nartf)

    f = [df_hbcm, df_vbcl, df_vbcr, df_artf]

    for n, cam in enumerate(cams):
        if reduced:
            nCh = channels['reduced']['eChannels'][cam]
        else:
            nCh = channels['full']['eChannels'][cam]

        if (np.shape(nCh)[0] != np.shape(f[n])):
            if (cam == 'ARTf'):
                # 113 114 115 // EIM_000
                nCh.remove(113)
                nCh.remove(114)
                nCh.remove(115)
                pass
            if (cam == 'HBCm'):
                # 1 32 // EIM_000
                # nCh.remove(0)
                # nCh.remove(31)
                # 0, 1, 2, 28, 29, 30, 31 //  KJM_027
                nCh.remove(0)
                nCh.remove(1)
                nCh.remove(2)
                nCh.remove(28)
                nCh.remove(29)
                nCh.remove(30)
                nCh.remove(31)
                pass

        for c, ch in enumerate(nCh):
            xp_error[ch] = f[n][c]

    if (np.max(xp_error) > 1.e5):
        # assuming xp_error in mW
        xp_error, df_hbcm, df_vbcl, df_vbcr, df_artf = \
            xp_error / 1000., df_hbcm / 1000., \
            df_vbcl / 1000., df_vbcr / 1000., df_artf / 1000.

    return (xp_error, df_hbcm, df_vbcl, df_vbcr, df_artf)


def load_chordal_input(
        program='20181010.032',
        strgrid='31x75x4',
        kani=[1.5, 0.8],
        nVals=[21, 6],
        times=3.42,
        nigs=5,
        nr=14,
        nt=100,
        RGS=False,
        no_ani=False,
        reduced=True,
        add_camera=True,
        phantom=False,
        debug=False):
    format1 = '.2f' if kani[0] < 1.e1 else '.1f'
    format2 = '.2f' if kani[1] < 1.e1 else '.1f'

    name = location + '/'
    if phantom:
        name += 'phantom/'
    name += program + '/' + strgrid
    if not phantom:
        name += '/' + program
    else:
        name += '/' + 'phan_'

    if RGS:
        name += 'RGS_'
    if (no_ani == 1):
        name += 'no_ani'
    elif (no_ani == 0):
        name += 'kani_' + format(kani[0], format1) + '_' + format(
            kani[1], format1)
    elif (no_ani in [2, 3, 4]):
        name += 'aniM' + str(no_ani) + '_' + format(
            kani[0], format1) + '_' + format(kani[1], format2)
        if (nVals is not None) and (no_ani == 4):
            name += '_nT' + str(nVals[0]) + '_nW' + str(nVals[1])
        elif (nVals is not None) and (no_ani == 3):
            name += '_nT' + str(nVals)
    # if RGS:
    #     name += '_RGS'

    if reduced:
        name += '_reduced'
    if add_camera:
        name += '_artf'
    name += '_nigs' + str(nigs) + '_' + strgrid
    name += '_' + format(times, '.2f')

    sigur0 = np.loadtxt(name + '_sig_ur0.dat')
    if debug:
        print('\t\tsigur0:', np.shape(sigur0), sigur0.__class__)

    return (sigur0)


def reform_chordal_input(
        program='20181010.032',
        strgrid='31x75x4',
        kani=[1.5, 0.8],
        nVals=[21, 0.6],
        times=3.42,
        nigs=5,
        nr=14,
        nt=100,
        no_ani=0,
        RGS=False,
        add_camera=False,
        reduced=True,
        phantom=False,
        debug=False):

    channels = mfr_transf.mfr_channel_list()
    sigur0 = load_chordal_input(
        program=program, kani=kani, nigs=nigs, phantom=phantom,
        nr=nr, nt=nt, debug=debug, reduced=reduced, add_camera=add_camera,
        RGS=RGS, times=times, no_ani=no_ani, nVals=nVals, strgrid=strgrid)

    if reduced:
        nCh = channels['reduced']['eChannels']
    else:
        nCh = channels['full']['eChannels']

    nhbcm, nvbcl, nvbcr, nartf = nCh['nHBCm'], \
        nCh['nVBCl'], nCh['nVBCr'], nCh['nARTf']

    if add_camera:
        cams = ['HBCm', 'VBCl', 'VBCr', 'ARTf']
    else:
        cams = ['HBCm', 'VBCl', 'VBCr']

    if reduced:
        if ((np.shape(sigur0)[0] < 58
             ) or ('ARTf' in cams and np.shape(sigur0)[0] < 82)):
            # 1 32 // EIM_000
            # nhbcm = nhbcm - 2
            # 0, 1, 2, 28, 29, 30, 31 //  KJM_027
            nhbcm = nhbcm - 7
    else:
        if ((np.shape(sigur0)[0] < 67
             ) or ('ARTf' in cams and np.shape(sigur0)[0] < 82)):
            # 1 32 // EIM_000
            # nhbcm = nhbcm - 2
            # 0, 1, 2, 28, 29, 30, 31 //  KJM_027
            nhbcm = nhbcm - 7

    xp_chordal = np.zeros((128))
    sig_hbcm = sigur0[:nhbcm]
    sig_vbcr = sigur0[nhbcm:nhbcm + nvbcr]
    sig_vbcl = sigur0[nhbcm + nvbcr:]
    if add_camera:
        sig_artf = sigur0[nhbcm + nvbcr + nvbcl:]
    else:
        sig_artf = np.zeros((nartf))

    if debug:
        print('\t\tsig_hbcm', np.shape(sig_hbcm), nhbcm, '\n',
              '\t\tsig_vbcr', np.shape(sig_vbcr), nvbcr, '\n',
              '\t\tsig_vbcl', np.shape(sig_vbcl), nvbcl)
        if add_camera:
            print('\t\tsig_artf', np.shape(sig_artf), nartf)

    if not add_camera:
        cams = ['HBCm', 'VBCl', 'VBCr']
    else:
        cams = ['HBCm', 'VBCl', 'VBCr', 'ARTf']

    f = [sig_hbcm, sig_vbcl, sig_vbcr, sig_artf]
    for n, cam in enumerate(cams):
        if reduced:
            nCh = channels['reduced']['eChannels'][cam]
        else:
            nCh = channels['full']['eChannels'][cam]

        if (np.shape(nCh)[0] != np.shape(f[n])):
            if (cam == 'ARTf'):
                nCh.remove(113)
                nCh.remove(114)
                nCh.remove(115)
            if (cam == 'HBCm'):
                # 1 32 // EIM_000
                # nCh.remove(0)
                # nCh.remove(31)
                # 0, 1, 2, 28, 29, 30, 31 //  KJM_027
                nCh.remove(0)
                nCh.remove(1)
                nCh.remove(2)
                nCh.remove(28)
                nCh.remove(29)
                nCh.remove(30)
                nCh.remove(31)

        for c, ch in enumerate(nCh):
            xp_chordal[ch] = f[n][c]

    if np.max(xp_chordal > 0.1):
        # assuming is in mW
        xp_chordal, sig_hbcm, sig_vbcl, sig_vbcr, sig_artf = \
            xp_chordal / 1000., sig_hbcm / 1000., \
            sig_vbcl / 1000., sig_vbcr / 1000.

    return (xp_chordal, sig_hbcm, sig_vbcl, sig_vbcr, sig_artf)


def load_tm2D(
        program='20181010.032',
        strgrid='31x75x4',
        kani=[1.5, 0.8],
        nVals=[21, 6],
        nigs=5,
        nr=14,
        nt=100,
        no_ani=0,
        RGS=False,
        reduced=True,
        add_camera=False,
        phantom=False,
        debug=False):
    format1 = '.2f' if kani[0] < 1.e1 else '.1f'
    format2 = '.2f' if kani[1] < 1.e1 else '.1f'

    name = location + '/'
    if phantom:
        name += 'phantom/'
    name += program + '/' + strgrid
    if not phantom:
        name += '/' + program
    else:
        name += '/' + 'phan_'

    if RGS:
        name += 'RGS_'
    if (no_ani == 1):
        name += 'no_ani'
    elif (no_ani == 0):
        name += 'kani_' + format(kani[0], format1) + '_' + format(
            kani[1], format2)
    elif (no_ani in [2, 3, 4]):
        name += 'aniM' + str(no_ani) + '_' + format(
            kani[0], format1) + '_' + format(kani[1], format2)
        if (nVals is not None) and (no_ani == 4):
            name += '_nT' + str(nVals[0]) + '_nW' + str(nVals[1])
        elif (nVals is not None) and (no_ani == 3):
            name += '_nT' + str(nVals)
    # if RGS:
    #     name += '_RGS'

    if reduced:
        name += '_reduced'
    if add_camera:
        name += '_artf'
    name += '_nigs' + str(nigs) + '_' + strgrid

    tm2D = np.loadtxt(name + '_tm2D.dat')
    if debug:
        print('\t\ttm2D:', tm2D.__class__, np.shape(tm2D))

    return (tm2D)


def reform_tm2D(
        program='20181010.032',
        strgrid='31x75x4',
        kani=[1.5, 0.8],
        nVals=[21, 6],
        nigs=5,
        nr=14,
        nt=100,
        no_ani=0,
        RGS=False,
        add_camera=False,
        reduced=True,
        phantom=False,
        debug=False):

    channels = mfr_transf.mfr_channel_list()
    tm2D = load_tm2D(
        strgrid=strgrid, program=program, kani=kani,
        nigs=nigs, phantom=phantom, nr=nr, nt=nt,
        debug=debug, reduced=reduced, nVals=nVals,
        RGS=RGS, no_ani=no_ani, add_camera=add_camera)
    ngrid = np.shape(tm2D)[1]

    if reduced:
        nCh = channels['reduced']
    else:
        nCh = channels['full']

    nhbcm, nvbcl, nvbcr, nartf = nCh['eChannels']['nHBCm'], nCh[
        'eChannels']['nVBCl'], nCh['eChannels']['nVBCr'], nCh[
            'eChannels']['nARTf']

    if add_camera:
        cams = ['HBCm', 'VBCl', 'VBCr', 'ARTf']
    else:
        cams = ['HBCm', 'VBCl', 'VBCr']

    if reduced:
        if ((np.shape(tm2D)[0] < 58
             ) or ('ARTf' in cams and np.shape(tm2D)[0] < 82)):
            # 1 32 // EIM_000
            # nhbcm = nhbcm - 2
            # 0, 1, 2, 28, 29, 30, 31 //  KJM_027
            nhbcm = nhbcm - 7
    else:
        if ((np.shape(tm2D)[0] < 67
             ) or ('ARTf' in cams and np.shape(tm2D)[0] < 82)):
            # 1 32 // EIM_000
            #  nhbcm = nhbcm - 2
            # 0, 1, 2, 28, 29, 30, 31 //  KJM_027
            nhbcm = nhbcm - 7

    tm_hbcm, tm_vbcl, tm_vbcr, tm_artf = \
        Z((nhbcm, ngrid)), Z((nvbcl, ngrid)), \
        Z((nvbcr, ngrid)), Z((nartf, ngrid))
    tm_hbcm = tm2D[:nhbcm, :]
    tm_vbcr = tm2D[nhbcm:nhbcm + nvbcr, :]
    tm_vbcl = tm2D[nhbcm + nvbcr:, :]
    if add_camera:
        tm_artf = tm2D[nhbcm + nvbcr + nvbcl:, :]

    if debug:
        print('\t\ttm_hbcm', np.shape(tm_hbcm), nhbcm, '\n',
              '\t\ttm_vbcl', np.shape(tm_vbcl), nvbcl, '\n',
              '\t\ttm_vbcr', np.shape(tm_vbcr), nvbcr)
        if add_camera:
            print('\t\ttm_artf', np.shape(tm_artf), nartf)

    def reform_tm2Ds(
            tm=np.zeros((32, 23 * 75))):
        nCh = np.shape(tm)[0]
        return (tm.reshape(nCh, nr, nt))

    if ((np.min(tm_hbcm) < 1e-10) or (np.min(tm_vbcl) < 1e-10) or
            (np.min(tm_vbcr) < 1e-10) or (np.min(tm_artf) < 1e-10)):
        # assuming geometry matrices are in cm^3, transform to m^3
        tm_hbcm, tm_vbcl, tm_vbcr, tm_artf = \
            tm_hbcm / (100.**3), tm_vbcl / (100.**3), \
            tm_vbcr / (100.**3), tm_artf / (100.**3)

    return (reform_tm2Ds(tm_hbcm),
            reform_tm2Ds(tm_vbcl),
            reform_tm2Ds(tm_vbcr),
            reform_tm2Ds(tm_artf),
            tm_hbcm, tm_vbcl, tm_vbcr, tm_artf)


def load_LoS_reff(
        nt=100,
        nr=30,
        base_location='../results/INVERSION/MFR/',
        new_type='ARTv',
        magconf='EIM_beta000',
        strgrid='sN8_50x30x100_1.4',
        debug=False,
        reduced=False,
        add_camera=False):
    name = base_location + strgrid + '/' + magconf + '_cam'

    cams = ['HBCm', 'VBCl', 'VBCr']
    if add_camera:
        cams.append('ARTf')
    channels = mfr_transf.mfr_channel_list()

    reff_LoS = np.zeros((128))
    for cam in cams:
        if reduced:
            eCh, gCh = \
                channels['reduced']['eChannels'][cam], \
                channels['reduced']['gChannels'][cam]
        else:
            eCh, gCh = \
                channels['full']['eChannels'][cam], \
                channels['full']['gChannels'][cam]

        if cam == 'ARTf':
            cam = new_type
        data = np.loadtxt(name + cam + '_reffLoS.dat')
        if debug:
            print(eCh, gCh, np.shape(data))

        for i, ch in enumerate(gCh):
            reff_LoS[eCh[i]] = data[ch - 1, 1]
        if debug:
            print('\t\treff_LoS:', np.shape(data), data[:, 1])

    if (np.max(reff_LoS) > 1.3):
        reff_LoS = reff_LoS / 100.

    return (reff_LoS)


def load_fs_reff(
        program='20181010.032',
        strgrid='31x75x4',
        kani=[1.5, 0.8],
        nVals=[21, 6],
        times=3.42,
        nigs=5,
        nr=14,
        nt=100,
        RGS=False,
        no_ani=False,
        reduced=True,
        add_camera=True,
        phantom=False,
        debug=False):
    format1 = '.2f' if kani[0] < 1.e1 else '.1f'
    format2 = '.2f' if kani[1] < 1.e1 else '.1f'

    name = location + '/'
    if phantom:
        name += 'phantom/'
    name += program + '/' + strgrid
    if not phantom:
        name += '/' + program
    else:
        name += '/' + 'phan_'

    if RGS:
        name += 'RGS_'
    if (no_ani == 1):
        name += 'no_ani'
    elif (no_ani == 0):
        name += 'kani_' + format(kani[0], format1) + '_' + format(
            kani[1], format2)
    elif (no_ani in [2, 3, 4]):
        name += 'aniM' + str(no_ani) + '_' + format(
            kani[0], format1) + '_' + format(kani[1], format2)
        if (nVals is not None) and (no_ani == 4):
            name += '_nT' + str(nVals[0]) + '_nW' + str(nVals[1])
        elif (nVals is not None) and (no_ani == 3):
            name += '_nT' + str(nVals)
    # if RGS:
    #     name += '_RGS'

    if reduced:
        name += '_reduced'
    if add_camera:
        name += '_artf'
    name += '_nigs' + str(nigs) + '_' + strgrid

    try:
        reff = np.loadtxt(name + '_' + format(
            times, '.2f') + '_reff.dat')
    except Exception:
        reff = np.loadtxt(name + '_reff.dat')

    if (np.max(reff) > 1.3):
        reff = reff / 100.

    if debug:
        print('\t\tfs reff:', np.shape(reff), reff)
    return (reff)


def load_gridfile(
        program='20181010.032',
        strgrid='31x75x4',
        kani=[1.5, 0.8],
        nVals=[21, 6],
        nigs=5,
        nr=14,
        nt=100,
        no_ani=0,
        RGS=False,
        reduced=True,
        add_camera=False,
        phantom=False,
        debug=False):
    format1 = '.2f' if kani[0] < 1.e1 else '.1f'
    format2 = '.2f' if kani[1] < 1.e1 else '.1f'

    name = location + '/'
    if phantom:
        name += 'phantom/'
    name += program + '/' + strgrid
    if not phantom:
        name += '/' + program
    else:
        name += '/' + 'phan_'

    if RGS:
        name += 'RGS_'
    if (no_ani == 1):
        name += 'no_ani'
    elif (no_ani == 0):
        name += 'kani_' + format(kani[0], format1) + '_' + format(
            kani[1], format2)
    elif (no_ani in [2, 3, 4, ]):
        name += 'aniM' + str(no_ani) + '_' + format(
            kani[0], format1) + '_' + format(kani[1], format2)
        if (nVals is not None) and (no_ani == 4):
            name += '_nT' + str(nVals[0]) + '_nW' + str(nVals[1])
        elif (nVals is not None) and (no_ani == 3):
            name += '_nT' + str(nVals)
    # if RGS:
    #     name += '_RGS'

    if reduced:
        name += '_reduced'
    if add_camera:
        name += '_artf'
    name += '_nigs' + str(nigs) + '_' + strgrid

    grid_r = np.loadtxt(name + '_rgrid_data.dat')
    grid_z = np.loadtxt(name + '_zgrid_data.dat')

    if (np.max(grid_r) > 7.) or (np.max(grid_z) > 1.):
        grid_r, grid_z = grid_r / 100., grid_z / 100.

    if debug:
        print('\t\tgrid_r:', grid_r.__class__, np.shape(grid_r))
        print('\t\tgrid_z:', grid_z.__class__, np.shape(grid_z))

    return (grid_r, grid_z)


def reform_gridfile(
        program='20181010.032',
        strgrid='31x75x4',
        kani=[1.5, 0.8],
        nVals=[21, 6],
        nigs=5,
        grid_nt=100,
        grid_nr=14,
        no_ani=0,
        RGS=False,
        reduced=True,
        add_camera=False,
        phantom=False,
        debug=False):

    grid_r, grid_z = load_gridfile(
        program=program, kani=kani, nigs=nigs,
        phantom=phantom, reduced=reduced, strgrid=strgrid,
        RGS=RGS, add_camera=add_camera, nVals=nVals,
        nr=grid_nr, nt=grid_nt, debug=debug, no_ani=no_ani)

    r, z = \
        np.zeros((grid_nr, grid_nt, np.shape(grid_r)[1])), \
        np.zeros((grid_nr, grid_nt, np.shape(grid_z)[1]))

    for k in range(0, np.shape(grid_r)[1]):
        r[:, :, k] = grid_r[:, k].reshape(grid_nr, grid_nt)
        z[:, :, k] = grid_z[:, k].reshape(grid_nr, grid_nt)

    if debug:
        print('\t\tr:', np.shape(r))
        print('\t\tz:', np.shape(z))

    return (r, z, grid_r, grid_z)


def load_mfr_result(
        program='20181010.032',
        strgrid='31x75x4',
        kani=[1.5, 0.8],
        nVals=[21, 6],
        nigs=5,
        times=3.42,
        nr=14,
        nt=100,
        base='_mfr1D',
        no_ani=0,
        RGS=False,
        reduced=True,
        add_camera=False,
        phantom=False,
        debug=False):
    format1 = '.2f' if kani[0] < 1.e1 else '.1f'
    format2 = '.2f' if kani[1] < 1.e1 else '.1f'

    name = location + '/'
    if phantom:
        name += 'phantom/'
    name += program + '/' + strgrid
    if not phantom:
        name += '/' + program
    else:
        name += '/' + 'phan_'

    if RGS:
        name += 'RGS_'
    if (no_ani == 1):
        name += 'no_ani'
    elif (no_ani == 0):
        name += 'kani_' + format(kani[0], format1) + '_' + format(
            kani[1], format2)
    elif (no_ani in [2, 3, 4]):
        name += 'aniM' + str(no_ani) + '_' + format(
            kani[0], format1) + '_' + format(kani[1], format2)
        if (nVals is not None) and (no_ani == 4):
            name += '_nT' + str(nVals[0]) + '_nW' + str(nVals[1])
        elif (nVals is not None) and (no_ani == 3):
            name += '_nT' + str(nVals)
    # if RGS:
    #     name += '_RGS'

    if reduced:
        name += '_reduced'
    if add_camera:
        name += '_artf'
    name += '_nigs' + str(nigs) + '_' + strgrid
    name += '_' + format(times, '.2f')

    out = np.loadtxt(name + base + '.dat')
    if debug:
        print('\t\tout1D:', out.__class__, np.shape(out))

    if isinstance(out, list):
        nigs_real = len(out[0, :])
    elif isinstance(out, np.ndarray):
        if len(np.shape(out)) >= 2:
            nigs_real = np.shape(out)[1]
        else:
            nigs_real = 1
    else:
        nigs_real = 1

    return (out, nigs_real, name + str(times) + base + '.dat')


def reform_mfr_result(
        program='20181010.032',
        strgrid='31x75x4',
        kani=[1.5, 0.8],
        nVals=[21, 6],
        nigs=5,
        times=3.42,
        grid_nt=100,
        grid_nr=14,
        base='_mfr1D',
        no_ani=0,
        RGS=False,
        reduced=True,
        add_camera=False,
        phantom=False,
        debug=False):

    inn, nigs_real = load_mfr_result(
        program=program, kani=kani, nigs=nigs,
        nr=grid_nr, nt=grid_nt, times=times, base=base, strgrid=strgrid,
        RGS=RGS, add_camera=add_camera, debug=debug, nVals=nVals,
        phantom=phantom, reduced=reduced, no_ani=no_ani)[:2]

    if nigs_real > 1:
        out = np.zeros((grid_nr, grid_nt, nigs_real))
        mfr2D = np.zeros((grid_nr, grid_nt, nigs_real))

        for k in range(nigs_real):
            minimum = np.min(inn[:, k])
            if minimum <= 0.0:
                out[:, :, k] = inn[:, k].clip(min=.0).reshape(
                    grid_nr, grid_nt)  # + np.abs(minimum)
            else:
                out[:, :, k] = inn[:, k].reshape(
                    grid_nr, grid_nt)
            mfr2D[:, :, k] = inn[:, k].reshape(grid_nr, grid_nt)

    else:
        minimum = np.min(inn)
        if minimum <= 0.0:
            out = inn.clip(min=.0).reshape(
                grid_nr, grid_nt)  # + np.abs(minimum)
        else:
            out = inn.reshape(grid_nr, grid_nt)
        mfr2D = inn.reshape(grid_nr, grid_nt)
    mfr1D = inn.clip(min=.0)

    if (np.max(out) < 1.e4):
        # assuming it's W/cm^3, transform to W/m^3
        out, mfr1D, mfr2D = \
            out * (100.**3), \
            mfr1D * (100.**3), \
            mfr2D * (100.**3)

    if debug:
        print('\t\tinn:', np.shape(inn))
        print('\t\tout:', np.shape(out))
        print('\t\tmfr2D:', np.shape(mfr2D))
    return (out, nigs_real, mfr1D, mfr2D)


def mfr_chi2(
        sigur=np.zeros((128)),
        dfur=np.zeros((128)),
        linback=np.zeros((128)),
        add_camera=False,
        reduced=True,
        debug=False):
    channels = mfr_transf.mfr_channel_list()
    if reduced:
        nCh = channels['reduced']['eChannels']
    else:
        nCh = channels['full']['eChannels']
    nhbcm, nvbcl, nvbcr, nartf = nCh['nHBCm'], nCh['nVBCl'], \
        nCh['nVBCr'], nCh['nARTf']

    chi_channels = np.zeros((np.shape(sigur)))
    for n, sig in enumerate(sigur):
        if dfur[n] != .0:
            chi_channels[n] = (sig - linback[n])**2 / dfur[n]**2

    N = nhbcm + nvbcr + nvbcl
    if add_camera:
        N += nartf

    chi2 = np.sum(chi_channels) / N
    if debug:
        print('chi_channels:', np.shape(chi_channels),
              chi_channels, '\nchi2:', chi2)

    return (chi2, chi_channels)


def get_MinMaj_radius(
        magconf='EIM_beta000',
        debug=False):
    vmID = magconf.replace('beta', '')
    databs = database.import_database()
    link = databs['values']['magnetic_configurations'][
        vmID]['URI']  # link to geiger archive

    base_URI = 'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/'
    r_m = req.get(base_URI + link + '/minorradius.json').json()['minorRadius']
    R_M = req.get(base_URI + link + '/majorradius.json').json()['majorRadius']
    if debug:
        print('\t\tm_r:', r_m, ' M_R:', R_M)
    return (r_m, R_M)


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
            print(cam, np.shape(forward), np.shape(nCh))

        if (np.shape(nCh)[0] != np.shape(forward[n])):
            if (cam == 'ARTf'):
                nCh.remove(113)
                nCh.remove(114)
                nCh.remove(115)
            if (cam == 'HBCm'):
                # 1 32 // EIM_000
                # nCh.remove(0)
                # nCh.remove(31)
                # 0, 1, 2, 28, 29, 30, 31 //  KJM_027
                nCh.remove(0)
                nCh.remove(1)
                nCh.remove(2)
                nCh.remove(28)
                nCh.remove(29)
                nCh.remove(30)
                nCh.remove(31)

        for c, ch in enumerate(nCh):
            chord[ch] = forward[c]
    return (chord)


def avg_radial_profile(
        data=np.zeros((14 * 100)),
        fs_radius=np.zeros((14)),
        grid_nr=14,
        grid_nt=100,
        debug=False):
    average = np.zeros((2, grid_nr))
    for r in range(grid_nr):
        average[0, r] = fs_radius[r]
        average[1, r] = np.mean(data[r * grid_nt:(r + 1) * grid_nt])
    return (average)


def profile_radial_and_chord(
        nr=31,
        nt=75,
        tomogram1D=np.zeros((31 * 75)),
        fs_reff=np.zeros((31)),
        emiss=[np.zeros((32, 26 * 75))],
        add_camera=False,
        reduced=True,
        debug=False):

    radial_profile = avg_radial_profile(
        data=tomogram1D, fs_radius=fs_reff,
        grid_nr=nr, grid_nt=nt, debug=False)
    chordal_profile = forward_integral(
        data=tomogram1D, emissivity=emiss, add_camera=add_camera,
        channels=mfr_transf.mfr_channel_list(),
        reduced=reduced, debug=debug)

    if debug:
        print(np.shape(radial_profile), radial_profile.__class__)
        print(np.shape(chordal_profile), chordal_profile.__class__)

    return (chordal_profile, radial_profile)


def get_mfr_results(
        program='20181010.032',
        no_ani=4,
        kani=[.1, 1.],
        nVals=[20, 1],
        nigs=5,
        times=3.42,
        grid_nt=100,
        grid_nr=30,
        strgrid='sN8_50x30x100_1.4',
        base='_mfr1D',
        magconf='EIM_beta000',
        new_type=None,
        RGS=False,
        add_camera=False,
        reduced=True,
        phantom=False,
        plot=False,
        saving=False,
        debug=False):

    r, z = reform_gridfile(
        program=program, kani=kani, nigs=nigs, grid_nt=grid_nt,
        grid_nr=grid_nr, debug=debug, add_camera=add_camera,
        RGS=RGS, phantom=phantom, nVals=nVals,
        reduced=reduced, no_ani=no_ani, strgrid=strgrid)[:2]
    fs_reff = load_fs_reff(
        program=program, kani=kani, nigs=nigs, nt=grid_nt,
        nr=grid_nr, debug=debug, strgrid=strgrid, RGS=RGS,
        add_camera=add_camera, phantom=phantom, nVals=nVals,
        reduced=reduced, no_ani=no_ani, times=times)
    LoS_reff = load_LoS_reff(
        nt=grid_nt, nr=grid_nr, new_type=new_type,
        debug=debug, add_camera=add_camera, reduced=reduced,
        magconf=magconf, strgrid=strgrid)
    m_r, M_R = get_MinMaj_radius(magconf=magconf, debug=debug)

    out_2D, nigs_real, mfr1D, mfr2D = reform_mfr_result(
        program=program, kani=kani, nigs=nigs, times=times,
        grid_nr=grid_nr, grid_nt=grid_nt, base=base, strgrid=strgrid,
        RGS=RGS, add_camera=add_camera, debug=debug, nVals=nVals,
        phantom=phantom, reduced=reduced, no_ani=no_ani)
    tm_hbcm, tm_vbcl, tm_vbcr, tm_artf = reform_tm2D(
        program=program, kani=kani, nigs=nigs, nr=grid_nr,
        RGS=RGS, nt=grid_nt, debug=debug, nVals=nVals,
        phantom=phantom, reduced=reduced, no_ani=no_ani,
        add_camera=add_camera, strgrid=strgrid)[4:]
    xp_error, df_hbcm, df_vbcl, df_vbcr, df_artf = reform_error_input(
        program=program, kani=kani, nigs=nigs, nVals=nVals,
        RGS=RGS, nr=grid_nr, nt=grid_nt, add_camera=add_camera,
        debug=debug, phantom=phantom, reduced=reduced,
        times=times, no_ani=no_ani, strgrid=strgrid)

    xp_chordal, sig_hbcm, sig_vbcl, sig_vbcr, sig_artf = reform_chordal_input(
        program=program, kani=kani, nigs=nigs, nVals=nVals,
        nr=grid_nr, nt=grid_nt, add_camera=add_camera,
        debug=debug, phantom=phantom, reduced=reduced, times=times,
        RGS=RGS, no_ani=no_ani, strgrid=strgrid)
    tomogram_chordal, tomogram_radial = profile_radial_and_chord(
        tomogram1D=mfr1D, nr=grid_nr, nt=grid_nt, reduced=reduced,
        fs_reff=fs_reff, add_camera=add_camera, debug=debug,
        emiss=[tm_hbcm, tm_vbcl, tm_vbcr, tm_artf])

    xp_chordal = np.array([LoS_reff, xp_chordal])
    tomogram_chordal = np.array([LoS_reff, tomogram_chordal])

    chi2, chi_channel = mfr_chi2(
        sigur=xp_chordal[1], dfur=xp_error, linback=tomogram_chordal[1],
        reduced=reduced, add_camera=add_camera, debug=debug)
    kani_profile = load_kani_output(
        program=program, kani=kani, nigs=nigs, phantom=phantom, RGS=RGS,
        nr=grid_nr, debug=debug, nVals=nVals, add_camera=add_camera,
        reduced=reduced, strgrid=strgrid, times=times, no_ani=no_ani)
    dphi_profile = load_dphi_grad(
        program=program, kani=kani, nigs=nigs, phantom=phantom,
        debug=debug, nVals=nVals, add_camera=add_camera, fs_reff=fs_reff,
        RGS=RGS, reduced=reduced, strgrid=strgrid, no_ani=no_ani)

    peaks_r, peaks_hw, peaks_id, half_widths = \
        phc.fwhm_radial_profiles(profile=tomogram_radial)

    tomo_total, tomo_core, error_2D = ph_m.power_2D_profile(
        profile=out_2D, raw=mfr2D, r_grid=r, z_grid=z,
        radial_profile=tomogram_radial, minor_radius=m_r,
        vmec_ID=magconf.replace('beta', ''), fs_reff=fs_reff,
        nr=grid_nr, nt=grid_nt, major_radius=M_R,
        debug=debug)

    prad_tomogram = ph_m.fast_prad_chordal(
        chordal=tomogram_chordal[1], magconf=magconf,
        strgrid=strgrid, reduced=reduced, debug=debug)
    prad_xp = ph_m.fast_prad_chordal(
        chordal=xp_chordal[1], magconf=magconf,
        strgrid=strgrid, reduced=reduced, debug=debug)

    label = 'results_xp_tomogram_tomo_' + program
    if RGS:
        label += '_RGS'
    if (no_ani == 1):
        label += '_no_ani_'

    else:
        label += 'aniM' + str(no_ani) + '_' + format(kani[0], '.2f') + \
            '_' + format(kani[1], '.2f')
        if (no_ani in [3, 4]) and (nVals is not None):
            if (no_ani == 3):
                label += '_nT' + str(nVals)
            if (no_ani == 4):
                label += '_nT' + str(nVals[0]) + '_nW' + str(nVals[1])

    if reduced:
        label += '_reduced'
    if add_camera:
        label += '_ARTf'
    label += '_nigs' + str(nigs) + '_' + strgrid
    label += '_' + format(times, '.2f') + base

    if plot:
        mfrp.tomogram_plot_wrapper(
            data=out_2D, x=r, y=z, nr=grid_nr, nt=grid_nt,
            chordal_tomo=tomogram_chordal, chordal_xp=xp_chordal,
            radial_tomo=tomogram_radial, total_power=tomo_total,
            prad_tomo=prad_tomogram, prad_xp=prad_xp,
            core_power=tomo_core, chi=chi_channel, chi2=chi2,
            kani_profile=kani_profile, nigs_real=nigs_real,
            minor_radius=m_r, peaks_r=peaks_r, peaks_id=peaks_id,
            half_widths=half_widths, label=label, strgrid=strgrid,
            add_camera=add_camera, xp_error=xp_error,
            radial_error=tomogram_radial[0, 1] - tomogram_radial[0, 0],
            absolute_error=np.abs(np.min(mfr2D)) / 2.,
            debug=debug, magconf=magconf.replace('beta', ''),)

    if saving:
        data = {'label': label, 'values': {}}
        d = data['values']

        # basic info on the results and phantom image
        d['tomogram'] = out_2D
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
        d['VMEC_ID'] = magconf.replace('beta', '')

        # difference between tomogram and phantom image
        d['difference'] = {}
        d['difference']['chi2'] = chi2

        # radial and chordial profile from both
        # the tomogram and phantom image
        d['profiles'] = {}
        d['profiles']['radial_tomogram'] = tomogram_radial
        d['profiles']['chordal_tomogram'] = tomogram_chordal

        # radial errobar from cell
        d['profiles']['xp_error'] = xp_error
        d['profiles']['abs_error'] = np.abs(np.min(mfr2D)) / 2.
        d['profiles']['2D_error'] = error_2D
        d['profiles']['radial_error'] = \
            tomogram_radial[0, 1] - tomogram_radial[0, 0]

        d['peaks'] = {}
        d['peaks']['half_widths'] = {}
        d['peaks']['half_widths']['tomogram'] = half_widths
        d['peaks']['radial_pos'] = {}
        d['peaks']['radial_pos']['tomogram'] = peaks_r
        d['peaks']['index_pos'] = {}
        d['peaks']['index_pos']['tomogram'] = peaks_id

        # total radiated power from profiles
        d['P_rad'] = {}
        d['P_rad']['tomogram'] = prad_tomogram
        d['P_rad']['experiment'] = prad_xp
        d['total_power'] = {}
        d['total_power']['tomogram'] = tomo_total
        d['core_power'] = {}
        d['core_power']['tomogram'] = tomo_core

        file = '../results/INVERSION/MFR/' + strgrid + '/' + label + '.json'

        print('\t\tWriting to', file)
        outdict = mClass.dict_transf(data, to_list=True)
        with open(file, 'w') as outfile:
            json.dump(outdict, outfile, indent=4, sort_keys=False)
        outfile.close()
        data = mClass.dict_transf(outdict, to_list=False)

    return (out_2D,             # 0
            r,                  # 1
            z,                  # 2
            fs_reff,            # 3
            LoS_reff,           # 4
            m_r,                # 5
            M_R,                # 6
            tomogram_chordal,   # 7
            xp_chordal,         # 8
            tomogram_radial,    # 9
            peaks_r,            # 10
            peaks_hw,           # 11
            peaks_id,           # 12
            half_widths,        # 13
            tomo_total,         # 14
            tomo_core,          # 15
            prad_tomogram,      # 16
            prad_xp,            # 17
            xp_error,           # 18
            kani_profile,       # 19
            dphi_profile,       # 20
            chi2,               # 21
            tm_hbcm,            # 22
            tm_vbcl,            # 23
            tm_vbcr,            # 24
            tm_artf,            # 25
            label,              # 26
            base,               # 27
            nigs_real,          # 28
            mfr2D)              # 29
