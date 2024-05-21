""" **************************************************************************
    so header """

import os
import json
import pickle
import requests

from scipy.signal import chirp, find_peaks, peak_widths
import numpy as np

import mClass
import mfr_plot as mfrp
import phantom_metrics as pm

Z = np.zeros
ones = np.ones

""" end of header
************************************************************************** """


def load_phantomvtomograms(
        base_location='../results/INVERSION/MFR/',
        strgrid=['31x125'],
        file_list=[['None', 'None']],
        debug=False):
    data = []
    for k, grid in enumerate(strgrid):
        for l, file in enumerate(file_list[k]):
            F = base_location + grid + '/' + file
            if os.path.isfile(F):
                with open(F, 'r') as infile:
                    indict = json.load(infile)
                infile.close()
                indict = mClass.dict_transf(
                    indict, to_list=False)
                data.append(indict)
            else:
                print('\t\t\\\ failed to find', F)
    return (data, file_list)


def compare(
        label='Nones_scan',
        definition_space=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        abscissa_label='none [none]',
        base_location='../results/INVERSION/MFR/',
        strgrid=['31x125'],
        file_list=[['None', 'None']],
        saving=False,
        debug=False):
    # strgrid[0] + \
    dump = base_location + 'old_comp' + \
        '/phantom_v_tomogram_' + label + '.pkl'

    # if os.path.isfile(dump):
    #     with open(dump, 'rb') as infile:
    #         indict = pickle.load(infile)
    #     infile.close()
    #     data = mClass.dict_transf(indict, list_bool=False)

    #     mfrp.phantom_v_tomogram(
    #         data=data, debug=debug)
    #     return (data)
    # else:
    #     pass

    objects, files = load_phantomvtomograms(
        base_location=base_location + 'backup/',
        file_list=file_list,
        strgrid=strgrid, debug=debug)
    # peak locations and hafl/full widths of reff position peaks
    tPeaks, pPeaks = [], []  # radial profile peaks
    tHWidths, pHWidths = [], []  # maxima half widths
    # mean square deviation statistics
    cMSD, sMSD, tMSD, pcs = [], [], [], []
    # chi2 as error measurement of actually available data
    chi2 = []
    # have to extend abscissa/definition space for multiples
    pAbs, tAbs = [], []
    # power in core and SOL
    pTotal, pCore, tTotal, tCore = [], [], [], []
    # prads of VBC/HBC
    pHBC, pVBC, tHBC, tVBC = [], [], [], []

    N = np.shape(file_list)[1]
    for k, grid in enumerate(strgrid):
        for l, file in enumerate(file_list[k]):
            m_r = objects[k * N + l]['values']['minor_radius']

            if debug:
                print(k, l, k * N + l,
                      np.shape(objects[(k * N) + l]['values'][
                          'profiles']['radial_tomogram']))

            if 'peaks' in objects[k * N + l].keys():
                tP, pP = objects[k * N + l]['values']['peaks'][
                    'radial_pos']['tomogram'] / m_r, objects[k * N + l][
                        'values']['peaks']['radial_pos']['phantom'] / m_r
                pHW = fwhm_radial_profiles(profile=objects[k * N + l][
                    'values']['profiles']['radial_phantom'])[1]
                tHW = fwhm_radial_profiles(profile=objects[k * N + l][
                    'values']['profiles']['radial_tomogram'])[1]
            else:
                tP, tHW = fwhm_radial_profiles(
                    profile=objects[k * N + l]['values']['profiles'][
                        'radial_tomogram'])[:2]
                pP, pHW = fwhm_radial_profiles(
                    profile=objects[k * N + l]['values']['profiles'][
                        'radial_phantom'])[:2]

            for i in range(np.shape(tP)[0]):
                tAbs.append(definition_space[k])
            for i in range(np.shape(pP)[0]):
                pAbs.append(definition_space[k])

            # peaks, FWHM etc.
            tPeaks.append(tP)
            pPeaks.append(pP)
            tHWidths.append(tHW)
            pHWidths.append(pHW)

            if 'core_power' in objects[k * N + l]['values'].keys():
                pTotal.append(objects[k * N + l]['values'][
                    'total_power']['phantom'])
                tTotal.append(objects[k * N + l]['values'][
                    'total_power']['tomogram'])
                pCore.append(objects[k * N + l]['values'][
                    'core_power']['phantom'])
                tCore.append(objects[k * N + l]['values'][
                    'core_power']['tomogram'])
            else:
                tT, tC = pm.power_2D_profile(
                    profile=objects[k * N + l]['values']['tomogram'],
                    r_grid=objects[k * N + l]['values']['r_grid'],
                    z_grid=objects[k * N + l]['values']['z_grid'],
                    minor_radius=objects[k * N + l]['values']['minor_radius'],
                    major_radius=objects[k * N + l]['values']['major_radius'],
                    fs_reff=objects[k * N + l]['values']['profiles'][
                        'radial_tomogram'][0])[:2]
                pT, pC = pm.power_2D_profile(
                    profile=objects[k * N + l]['values']['phantom'],
                    r_grid=objects[k * N + l]['values']['r_grid'],
                    z_grid=objects[k * N + l]['values']['z_grid'],
                    minor_radius=objects[k * N + l]['values']['minor_radius'],
                    major_radius=objects[k * N + l]['values']['major_radius'],
                    fs_reff=objects[k * N + l]['values']['profiles'][
                        'radial_phantom'][0])[:2]

                pCore.append(pC * 1.e6)
                tCore.append(tC * 1.e6)
                pTotal.append(pT * 1.e6)
                tTotal.append(tT * 1.e6)

            if 'P_rad' in objects[k * N + l]['values'].keys():  
                # prad for both cameras
                pHBC.append(objects[k * N + l]['values'][
                    'P_rad']['phantom']['HBCm'])
                pVBC.append(objects[k * N + l]['values'][
                    'P_rad']['phantom']['VBC'])
                tHBC.append(objects[k * N + l]['values'][
                    'P_rad']['tomogram']['HBCm'])
                tVBC.append(objects[k * N + l]['values'][
                    'P_rad']['tomogram']['VBC'])
            else:
                prad_tomogram = pm.fast_prad_chordal(
                    chordal=objects[k * N + l]['values']['profiles'][
                        'chordial_tomogram'][1], reduced=False)
                prad_phantom = pm.fast_prad_chordal(
                    chordal=objects[k * N + l]['values']['profiles'][
                        'chordial_phantom'][1], reduced=False)

                pHBC.append(prad_phantom['HBCm'])
                tHBC.append(prad_tomogram['HBCm'])
                pVBC.append(prad_phantom['VBC'])
                tVBC.append(prad_tomogram['VBC'])

            tomo, phantom = objects[k * N + l]['values']['tomogram'], \
                objects[k * N + l]['values']['phantom']
            r, z = objects[k * N + l]['values']['r_grid'], \
                objects[k * N + l]['values']['z_grid']
            nr, nt = np.shape(r)[:2]

            core_msd, SOL_msd, total_msd = pm.diff_tomo_phantom(
                nr=nr, nt=nt, r_grid=r, z_grid=z, nCore=nr - 5,
                phantom=phantom, tomogram=tomo)[1:]
            # MSD for tomogram areas seperated
            cMSD.append(core_msd)
            sMSD.append(SOL_msd)
            tMSD.append(total_msd)

            if 'pearson_coefficient' in objects[k * N + l][
                    'values']['difference'].keys():
                # pearson coefficient from cross correlation
                pcs.append(objects[k * N + l]['values'][
                    'difference']['pearson_coefficient'])
            else:
                pcs.append(pm.pearson_covariance(
                    tomogram1D=objects[k * N + l]['values'][
                        'tomogram'].reshape(nr * nt), phantom1D=objects[
                            k * N + l]['values']['phantom'].reshape(nr * nt)))

            # chi2
            chi2.append(objects[k * N + l][
                'values']['difference']['chi2'])

    data = {'label': label,
            'strgrid': strgrid,
            'values': {}}
    v = data['values']

    v['abscissa'] = {}
    v['abscissa']['label'] = abscissa_label
    v['abscissa']['values'] = definition_space

    # flattening since multiple cases per image are possible
    pPeaks = [item for sublist in pPeaks for item in sublist]
    pHWidths = [item for sublist in pHWidths for item in sublist]

    tPeaks = [item for sublist in tPeaks for item in sublist]
    tHWidths = [item for sublist in tHWidths for item in sublist]

    v['FWHM'] = {}
    v['FWHM']['tomogram'] = {}
    v['FWHM']['tomogram']['peaks'] = tPeaks
    v['FWHM']['tomogram']['half_widths'] = tHWidths
    v['FWHM']['tomogram']['abscissa'] = tAbs

    v['FWHM']['phantom'] = {}
    v['FWHM']['phantom']['peaks'] = pPeaks
    v['FWHM']['phantom']['half_widths'] = pHWidths
    v['FWHM']['phantom']['abscissa'] = pAbs

    v['chi2'] = chi2

    v['MSD'] = {}
    v['MSD']['core'] = cMSD
    v['MSD']['SOL'] = sMSD
    v['MSD']['total'] = tMSD
    v['MSD']['pearson_coefficient'] = pcs

    v['power'] = {}
    v['power']['core'] = {}
    v['power']['core']['tomogram'] = tCore
    v['power']['core']['phantom'] = pCore
    v['power']['total'] = {}
    v['power']['total']['tomogram'] = tTotal
    v['power']['total']['phantom'] = pTotal

    v['P_rad'] = {}
    v['P_rad']['tomogram'] = {}
    v['P_rad']['tomogram']['HBCm'] = tHBC
    v['P_rad']['tomogram']['VBC'] = tVBC
    v['P_rad']['phantom'] = {}
    v['P_rad']['phantom']['HBCm'] = pHBC
    v['P_rad']['phantom']['VBC'] = tHBC

    mfrp.phantom_v_tomogram(
        data=data, debug=debug)

    if saving:
        print('\\\ Writing to', dump)
        outdict = mClass.dict_transf(data, to_list=True)
        with open(dump, 'wb') as outfile:
            pickle.dump(outdict, outfile, pickle.HIGHEST_PROTOCOL)
        outfile.close()
        data = mClass.dict_transf(outdict, to_list=False)
    return (data, objects)


def fwhm_radial_profiles(
        profile=np.zeros((2, 31)),
        debug=False):
    pP, _ = find_peaks(profile[1])
    profHW = peak_widths(profile[1], pP, rel_height=0.5)  # widths

    f = 10000
    index_stretch = np.linspace(
        .0, np.shape(profile)[1] - 1, np.shape(profile)[1] * f)
    r_stretch = np.interp(
        np.linspace(.0, 1., np.shape(profile)[1] * f),
        np.linspace(.0, 1., np.shape(profile)[1]), profile[0, :])

    half_widths = np.zeros((np.shape(profHW)))
    pHW = []
    for i in range(np.shape(pP)[0]):
        j0, _ = mClass.find_nearest(index_stretch, profHW[2][i])
        j1, _ = mClass.find_nearest(index_stretch, profHW[3][i])
        pHW.append((r_stretch[j1] - r_stretch[j0]) * 100.)

        half_widths[0][i] = pHW[i]  # r width
        half_widths[1][i] = profHW[1][i]  # height at which line evaluated
        half_widths[2][i] = r_stretch[j0]  # r left
        half_widths[3][i] = r_stretch[j1]  # r right

    if debug:
        print('profile:', pP, pHW)
    pP_r = profile[0, pP]
    return (pP_r, pHW, pP, half_widths)
