""" **************************************************************************
    so header """

import numpy as np

import mfr_plot as mfrp
import mfr2D_matrix_gridtransform as mfr_transf

Z = np.zeros
ones = np.ones

""" end of header
************************************************************************** """


def compare_emissivity_matrices(
        debug=True):

    daz_tm, daz_grid, daz_tm_m, daz_tm_l, daz_tm_r, \
        pih_tm, pih_grid, pih_tm_m, pih_tm_l, pih_tm_r = load_check_matrices()
    ch = mfr_transf.mfr_channel_list()['reduced']
    ech = ch['eChannels']

    M_pih, M_daz = np.shape((pih_tm))[0], np.shape((daz_tm))[0]
    nHBCm, nVBCl, nVBCr = ech['nHBCm'], ech['nVBCl'], ech['nVBCr']

    HBCm_pih, VBCr_pih, VBCl_pih = np.zeros((nHBCm, M_pih)), \
        np.zeros((nVBCr, M_pih)), np.zeros((nVBCl, M_pih))
    HBCm_daz, VBCr_daz, VBCl_daz = np.zeros((nHBCm, M_daz)), \
        np.zeros((nVBCr, M_daz)), np.zeros((nVBCl, M_daz))

    if debug:
        print('daz_tm:', np.shape(daz_tm), 'nHBCm', nHBCm,
              'nVBCl:', nVBCl, 'nVBCr:', nVBCr)

    for n, c in enumerate(ch['gChannels']['HBCm']):
        HBCm_daz[n, :] = daz_tm[:, n]
        HBCm_pih[n, :] = pih_tm[:, n]

    for n, c in enumerate(ch['gChannels']['VBCr']):
        VBCr_daz[n, :] = daz_tm[:, n + nHBCm]
        VBCr_pih[n, :] = pih_tm[:, n + nHBCm]

    for n, c in enumerate(ch['gChannels']['VBCl']):
        VBCl_daz[n, :] = daz_tm[:, n + nHBCm + nVBCr]
        VBCl_pih[n, :] = pih_tm[:, n + nHBCm + nVBCr]

    mfrp.emissivity_comparison_plot(
        channels=ch, pih_grid=pih_grid, HBCm_pih=HBCm_pih,
        VBCr_pih=VBCr_pih, VBCl_pih=VBCl_pih,
        daz_grid=daz_grid, HBCm_daz=HBCm_daz,
        VBCr_daz=VBCr_daz, VBCl_daz=VBCl_daz)

    return


def load_check_matrices(
        daz_base='../../bolometer_mfr/standard/14x100/',
        pih_base='../results/INVERSION/MFR/23x25/',
        debug=True):

    daz_grid = np.loadtxt(
        '../../bolometer_mfr/tmw7x/grids2D/' +
        'standard_grids2D_14x100_phi108.dat',
        skiprows=2)
    daz_tm_m = np.loadtxt(
        daz_base + 'standard_camHBCm_tm2D.dat')
    daz_tm_r = np.loadtxt(
        daz_base + 'standard_camVBCr_tm2D.dat')
    daz_tm_l = np.loadtxt(
        daz_base + 'standard_camVBCl_tm2D.dat')
    daz_tm = np.loadtxt(
        '../../bolometer_mfr/output/20181010.032/14x100/' +
        '20181010.032kani_1.50_0.80_nigs1_14x100_tm2D_loaded_results.dat')
    if debug:
        print(np.shape(daz_grid), np.shape(daz_tm_m), np.shape(daz_tm_r),
              np.shape(daz_tm_l), np.shape(daz_tm))

    pih_grid = np.loadtxt(
        pih_base + 'EIM_beta000_grids2D_23x25_phi108.dat',
        skiprows=2)
    pih_tm_m = np.loadtxt(
        pih_base + 'EIM_beta000_camHBCm_tm2D.dat')
    pih_tm_r = np.loadtxt(
        pih_base + 'EIM_beta000_camVBCr_tm2D.dat')
    pih_tm_l = np.loadtxt(
        pih_base + 'EIM_beta000_camVBCl_tm2D.dat')
    pih_tm = np.loadtxt(
        '../../bolometer_mfr/output/20181010.032/23x25/' +
        '20181010.032kani_1.50_0.80_nigs1_23x25_tm2D_loaded_results.dat')
    if debug:
        print(np.shape(pih_grid), np.shape(pih_tm_m), np.shape(pih_tm_r),
              np.shape(pih_tm_l), np.shape(pih_tm))

    return (daz_tm, daz_grid, daz_tm_m, daz_tm_l, daz_tm_r,
            pih_tm, pih_grid, pih_tm_m, pih_tm_l, pih_tm_r)
