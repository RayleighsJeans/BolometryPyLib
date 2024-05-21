""" *************************************************************************
    so HEADER
    Created on Mon Feb 18 17:01:00 2019
    Author: Hannes Damm
        git source:
            https://git.ipp-hgw.mpg.de/hdamm/ts_archive_2_profiles

    used and edited at own risk, not sanctioned by orignal creator
    Editor: Philipp Hacker (Mi Sep 25 11:03:00 2019) """

import sys
import archivedb
import numpy as np
import json
import glob
import os
from tqdm import tqdm
import pickle

import mClass

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

python_version = sys.version_info.major
if python_version == 2:
    import urllib2
elif python_version == 3:
    import urllib.request as urllib2
else:
    print('python version %s is not supported' % python_version)

vmec_base = 'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/'

""" eo header
************************************************************************** """


def TS_json(
        shotno='20181018.041',
        scaling='gauss',
        filename='../results/THOMSON/20181018/041/' +
                 'thomson_data_20181018.041_gauss_V10.txt',
        saving=False,
        debug=True,
        indent_level='\t'):
    """ Loads/interprets the TS data and plots/fits them for plot
    Args:
        shotno (str, optional): XP ID
        scaling (str, optional): Method of scaling
        file (str, optional): File name where to load from
        evalStart (float, optional): Evaluation time start (seconds f. T1)
        evalEnd (float, optional): Evaluation time end (seconds f. T1)
        saving_json (bool, optional): Should save results in json
        neplot (bool, optional): Should plot n_e
        teplot (bool, optional): Should plot t_e
        debug (bool, optional): Print debugging info
        indent_level (str, optional): Printing indentation
    Returns:
        None.
    Notes:
        None.
    """
    file = '../results/THOMSON/' + shotno[0:8] + \
        '/' + shotno[-3:] + '/TS_profile_' + \
        shotno.replace('.', '_')

    if saving:
        if not glob.glob(file + '.pickle') == []:
            print(indent_level + '\t\\\ TS file found, loading ' + file[-36:])
            try:
                with open(file + '.pickle', 'rb') as f:
                    TS_data = pickle.load(f)
                f.close()
                return (TS_data)

            except Exception:
                print(indent_level + '\t\\\ failed loading TS file')
                return None

    names = [
        't', 'shot_time', 'shotno', 'laserno', 'TS_version', 'vmec_id',
        'factor_real_space', 'factor_mapped', 'scattering_volume_number',
        'x_real', 'y_real', 'z_real', 'r_real', 'r_eff', 'r_cyl', 'phi_cyl',
        's_vmec', 'theta_vmec', 'phi_vmec', 'Te_map', 'Te_low', 'Te_high',
        'ne_map', 'ne_low', 'ne_high', 'pe_map', 'pe_low', 'pe_high',
        'ne_map_scaled', 'ne_low_scaled', 'ne_high_scaled',
        'pe_map_scaled', 'pe_low_scaled', 'pe_high_scaled',
        'We_map', 'We_low', 'We_high']

    TS_data = np.genfromtxt(
        filename, comments='#', delimiter=';', names=names,
        dtype=[
            float, float, '<U16', int, '<U16', '<U16', float, float,
            int, float, float, float, float, float, float, float, float,
            float, float, float, float, float, float, float, float, float,
            float, float, float, float, float, float, float, float, float,
            float, float])

    # scale n_e, p_e and W_e + errors
    TS_data['ne_map'] = TS_data['factor_mapped'] * TS_data['ne_map']
    TS_data['ne_low'] = TS_data['factor_mapped'] * TS_data['ne_low']
    TS_data['ne_high'] = TS_data['factor_mapped'] * TS_data['ne_high']
    TS_data['pe_map'] = TS_data['factor_mapped'] * TS_data['pe_map']
    TS_data['pe_low'] = TS_data['factor_mapped'] * TS_data['pe_low']
    TS_data['pe_high'] = TS_data['factor_mapped'] * TS_data['pe_high']
    TS_data['We_map'] = TS_data['factor_mapped'] * TS_data['We_map']
    TS_data['We_low'] = TS_data['factor_mapped'] * TS_data['We_low']
    TS_data['We_high'] = TS_data['factor_mapped'] * TS_data['We_high']

    T0 = TS_data['t']
    T = np.unique(TS_data['t'])  # [:10]
    M = len(T)
    if len(T) == 0:
        print('\t\t\\\ no time for eval')
        return

    M = len(T)
    data = {
        'time': T,
        'minor radius': [None] * M,
        'r_fit': [None] * M,
        's_vmec': [None] * M,
        'values': {
            'n_e': {
                'r without outliers (ne)': [None] * M,
                'ne gauss': [None] * M,
                'ne fit gauss': [None] * M,
                'ne fit gauss conv': [None] * M,
                'ne map': [None] * M,
                'ne map without outliers': [None] * M,
                'ne low97 without outliers': [None] * M,
                'ne high97 without outliers': [None] * M,
                'ne low97 gauss fit': [None] * M,
                'ne high97 gauss fit': [None] * M,
                'ne map factor real space': [None] * M,
                'ne low97 factor real space': [None] * M,
                'ne high97 factor real space': [None] * M,
            },
            'T_e': {
                'r without outliers (Te)': [None] * M,
                'Te gauss': [None] * M,
                'Te fit gauss': [None] * M,
                'Te fit gauss conv': [None] * M,
                'Te map without outliers': [None] * M,
                'Te low97 without outliers': [None] * M,
                'Te high97 without outliers': [None] * M,
                'Te low97 gauss fit': [None] * M,
                'Te high97 gauss fit': [None] * M,
            }
        }
    }
    # base data
    n_e, T_e = data['values']['n_e'], data['values']['T_e']

    t1 = int(archivedb.get_program_t1(shotno, useCache=True))
    print(indent_level + '>> setting up profiles:', end=' ')
    # for all times in region of interest
    for j in tqdm(range(M), desc='time'):
        i = T[j]
        J = np.where(T0 == i)[0]

        if debug:
            print('\n', indent_level, '\tT=',
                  round((i - t1) / 1e9, 4), end='s, ')

        try:
            minor_radius = json.loads(urllib2.urlopen(
                vmec_base + TS_data['vmec_id'][J][0] +
                '/minorradius.json').read().decode('utf-8'))['minorRadius']
            r_fit = np.arange(-minor_radius, minor_radius, 0.01)

            data['minor radius'][j] = minor_radius
            data['r_fit'][j] = r_fit
            data['s_vmec'][j] = TS_data['s_vmec'][J]
            n_e['ne map'][j] = TS_data['ne_map'][J]

            kernel = 100.0 * RBF(
                length_scale=1.0, length_scale_bounds=(1e-1, 1e3)) + \
                WhiteKernel(noise_level=1e-5,
                            noise_level_bounds=(1e-10, 1e+1))
            r_with_outliers = TS_data['r_eff'][J][
                np.abs(TS_data['s_vmec'][J]) <= 1]
            # remove outliers out of the 1sigma confidence intervall
            Te_map_with_outliers = TS_data['Te_map'][J][
                np.abs(TS_data['s_vmec'][J]) <= 1]

        except Exception:
            if debug:
                print('failed basic assertions', end=', ')
            continue  # failed most basic assertion

        try:  # gaussian Te fit with convenience
            gp_Te = GaussianProcessRegressor(
                kernel=kernel, alpha=0.0).fit(
                TS_data['r_eff'][J][np.abs(TS_data['s_vmec'][J]) <= 1][
                        :, np.newaxis], TS_data['Te_map'][J][
                    np.abs(TS_data['s_vmec'][J]) <= 1])
            Te_fit_gauss, Te_fit_gauss_cov = gp_Te.predict(
                r_fit[:, np.newaxis], return_cov=True)

            T_e['Te fit gauss'][j] = Te_fit_gauss.tolist()
            T_e['Te fit gauss conv'][j] = \
                np.sqrt(np.diag(Te_fit_gauss_cov)).tolist()

        except Exception:
            if debug:
                print('failed T_e gauss 2 x sigma', end=', ')

        try:  # interpolate mirrored TS data in vmec, Guassian process fit
            Te_conf = 1
            r_no_outliers = r_with_outliers[(
                Te_map_with_outliers >= np.interp(
                    r_with_outliers, r_fit, Te_fit_gauss -
                    Te_conf * np.sqrt(np.diag(Te_fit_gauss_cov)))) &
                (Te_map_with_outliers <= np.interp(
                    r_with_outliers, r_fit, Te_fit_gauss +
                    Te_conf * np.sqrt(np.diag(Te_fit_gauss_cov))))]

            try:  # for r without outliers: Te map
                Te_map_no_outliers = Te_map_with_outliers[(
                    Te_map_with_outliers >= np.interp(
                        r_with_outliers, r_fit, Te_fit_gauss -
                        Te_conf * np.sqrt(np.diag(Te_fit_gauss_cov)))) &
                    (Te_map_with_outliers <= np.interp(
                        r_with_outliers, r_fit, Te_fit_gauss +
                        Te_conf * np.sqrt(np.diag(Te_fit_gauss_cov))))]
                gp_Te_plot = GaussianProcessRegressor(
                    kernel=kernel, alpha=0.0).fit(
                        r_no_outliers[:, np.newaxis], Te_map_no_outliers)
                Te_fit_gauss_plot, Te_fit_gauss_plot_cov = \
                    gp_Te_plot.predict(
                        r_fit[:, np.newaxis], return_cov=True)

                T_e['r without outliers (Te)'][j] = \
                    r_no_outliers.tolist()
                T_e['Te gauss'][j] = \
                    Te_fit_gauss_plot.tolist()
                T_e['Te map without outliers'][j] = \
                    Te_map_no_outliers.tolist()

            except Exception:
                if debug:
                    print('failed Te map', end=', ')

            try:  # for r without outliers: Te 97.5
                Te_low_with_outliers = TS_data[
                    'Te_low'][J][np.abs(TS_data['s_vmec'][J]) <= 1]
                Te_high_with_outliers = TS_data[
                    'Te_high'][J][np.abs(TS_data['s_vmec'][J]) <= 1]

                Te_low_no_outliers = Te_low_with_outliers[(
                    Te_map_with_outliers >= np.interp(
                        r_with_outliers, r_fit, Te_fit_gauss -
                        Te_conf * np.sqrt(np.diag(Te_fit_gauss_cov)))) &
                    (Te_map_with_outliers <= np.interp(
                        r_with_outliers, r_fit, Te_fit_gauss +
                        Te_conf * np.sqrt(np.diag(Te_fit_gauss_cov))))]
                Te_high_no_outliers = Te_high_with_outliers[(
                    Te_map_with_outliers >= np.interp(
                        r_with_outliers, r_fit, Te_fit_gauss -
                        Te_conf * np.sqrt(np.diag(Te_fit_gauss_cov)))) &
                    (Te_map_with_outliers <= np.interp(
                        r_with_outliers, r_fit, Te_fit_gauss +
                        Te_conf * np.sqrt(np.diag(Te_fit_gauss_cov))))]

                T_e['Te low97 without outliers'][j] = \
                    Te_low_no_outliers.tolist()
                T_e['Te high97 without outliers'][j] = \
                    Te_high_no_outliers.tolist()

            except Exception:
                if debug:
                    print('failed Te 97.5%', end=', ')

        except Exception:
            if debug:
                print('failed r without outliers', end=', ')

        try:
            # fit error, fit the remaining error data
            gp_Te_low_plot = GaussianProcessRegressor(
                kernel=kernel, alpha=0.0).fit(
                r_no_outliers[:, np.newaxis], Te_low_no_outliers)
            Te_low_gauss_plot, Te_low_gauss_plot_cov = \
                gp_Te_low_plot.predict(
                    r_fit[:, np.newaxis], return_cov=True)
            gp_Te_high_plot = GaussianProcessRegressor(
                kernel=kernel, alpha=0.0).fit(
                r_no_outliers[:, np.newaxis], Te_high_no_outliers)
            Te_high_gauss_plot, Te_high_gauss_plot_cov = \
                gp_Te_high_plot.predict(
                    r_fit[:, np.newaxis], return_cov=True)

            T_e['Te low97 gauss fit'][j] = Te_low_gauss_plot.tolist()
            T_e['Te high97 gauss fit'][j] = Te_high_gauss_plot.tolist()

        except Exception:
            if debug:
                print('failed T_e 97.5%', end=', ')

        try:  # interpolate mirrored TS data in vmec coordinates
            gp_ne = GaussianProcessRegressor(kernel=kernel, alpha=0.0).fit(
                TS_data['r_eff'][J][
                    np.abs(TS_data['s_vmec'][J]) <= 1][
                    :, np.newaxis],
                TS_data['ne_map'][J][
                    np.abs(TS_data['s_vmec'][J]) <= 1])
            ne_fit_gauss, ne_fit_gauss_cov = gp_ne.predict(
                r_fit[:, np.newaxis], return_cov=True)

            ne_conf = 1
            r_with_outliers = TS_data['r_eff'][J][
                np.abs(TS_data['s_vmec'][J]) <= 1]
            ne_high_with_outliers = TS_data['ne_high'][J][
                np.abs(TS_data['s_vmec'][J]) <= 1]
            ne_map_with_outliers = TS_data['ne_map'][J][
                np.abs(TS_data['s_vmec'][J]) <= 1]
            ne_low_with_outliers = TS_data['ne_low'][J][
                np.abs(TS_data['s_vmec'][J]) <= 1]

            if int(shotno[0:4]) == 2018:
                try:
                    factor_real_space_with_outliers = TS_data[
                        'factor_real_space'][J]
                    factor_mapped_with_outliers = TS_data[
                        'factor_mapped'][J]

                except Exception:
                    if debug:
                        print('failed factor real space', end=', ')

            try:
                r_no_outliers = r_with_outliers[(
                    ne_map_with_outliers >= np.interp(
                        r_with_outliers, r_fit, ne_fit_gauss -
                        ne_conf * np.sqrt(np.diag(ne_fit_gauss_cov)))) &
                    (ne_map_with_outliers <= np.interp(
                        r_with_outliers, r_fit, ne_fit_gauss +
                        ne_conf * np.sqrt(np.diag(ne_fit_gauss_cov))))]
                ne_high_no_outliers = ne_high_with_outliers[(
                    ne_map_with_outliers >= np.interp(
                        r_with_outliers, r_fit, ne_fit_gauss -
                        ne_conf * np.sqrt(np.diag(ne_fit_gauss_cov)))) &
                    (ne_map_with_outliers <= np.interp(
                        r_with_outliers, r_fit, ne_fit_gauss +
                        ne_conf * np.sqrt(np.diag(ne_fit_gauss_cov))))]
                ne_low_no_outliers = ne_low_with_outliers[(
                    ne_map_with_outliers >= np.interp(
                        r_with_outliers, r_fit, ne_fit_gauss -
                        ne_conf * np.sqrt(np.diag(ne_fit_gauss_cov)))) &
                    (ne_map_with_outliers <= np.interp(
                        r_with_outliers, r_fit, ne_fit_gauss +
                        ne_conf * np.sqrt(np.diag(ne_fit_gauss_cov))))]

                n_e['r without outliers (ne)'][j] = \
                    r_no_outliers.tolist()
                n_e['ne low97 without outliers'][j] = \
                    ne_low_no_outliers.tolist()
                n_e['ne high97 without outliers'][j] = \
                    ne_high_no_outliers.tolist()

            except Exception:
                if debug:
                    print('failed ne map 2xsigma', end=', ')

            try:  # fit the remaining data
                ne_map_no_outliers = ne_map_with_outliers[(
                    ne_map_with_outliers >= np.interp(
                        r_with_outliers, r_fit, ne_fit_gauss -
                        ne_conf * np.sqrt(np.diag(ne_fit_gauss_cov)))) &
                    (ne_map_with_outliers <= np.interp(
                        r_with_outliers, r_fit, ne_fit_gauss +
                        ne_conf * np.sqrt(np.diag(ne_fit_gauss_cov))))]

                gp_ne_plot = GaussianProcessRegressor(
                    kernel=kernel, alpha=0.0).fit(
                        r_no_outliers[:, np.newaxis], ne_map_no_outliers)
                ne_fit_gauss_plot, ne_fit_gauss_cov = gp_ne_plot.predict(
                    r_fit[:, np.newaxis], return_cov=True)

                n_e['ne gauss'][j] = ne_fit_gauss_plot
                n_e['ne fit gauss'][j] = ne_fit_gauss
                n_e['ne fit gauss conv'][j] = np.sqrt(np.diag(ne_fit_gauss_cov))
                n_e['ne map without outliers'][j] = ne_map_no_outliers

            except Exception:
                if debug:
                    print('failed ne gauss 2xsigma', end=', ')

            if int(shotno[0:4]) == 2018:  # real space scaling
                try:  # factor mapped real space
                    factor_real_space_no_outliers = \
                        factor_real_space_with_outliers[(
                            ne_map_with_outliers >= np.interp(
                                r_with_outliers, r_fit,
                                ne_fit_gauss - ne_conf *
                                np.sqrt(np.diag(ne_fit_gauss_cov)))) &
                            (ne_map_with_outliers <= np.interp(
                                r_with_outliers, r_fit,
                                ne_fit_gauss + ne_conf *
                                np.sqrt(np.diag(ne_fit_gauss_cov))))]
                    factor_mapped_no_outliers = factor_mapped_with_outliers[(
                        ne_map_with_outliers >= np.interp(
                            r_with_outliers, r_fit, ne_fit_gauss - ne_conf *
                            np.sqrt(np.diag(ne_fit_gauss_cov)))) &
                        (ne_map_with_outliers <= np.interp(
                            r_with_outliers, r_fit, ne_fit_gauss + ne_conf *
                            np.sqrt(np.diag(ne_fit_gauss_cov))))]

                    n_e['ne_map factor real space'][j] = \
                        (factor_real_space_no_outliers /
                         factor_mapped_no_outliers *
                         ne_map_no_outliers)

                except Exception:
                    if debug:
                        print('failed factor real space', end=', ')

            try:
                # fit error, fit the remaining error data
                gp_ne_low_plot = GaussianProcessRegressor(
                    kernel=kernel, alpha=0.0).fit(
                    r_no_outliers[:, np.newaxis], ne_low_no_outliers)
                ne_low_gauss_plot, ne_low_gauss_plot_cov = \
                    gp_ne_low_plot.predict(
                        r_fit[:, np.newaxis], return_cov=True)
                gp_ne_high_plot = GaussianProcessRegressor(
                    kernel=kernel, alpha=0.0).fit(
                    r_no_outliers[:, np.newaxis], ne_high_no_outliers)
                ne_high_gauss_plot, ne_high_gauss_plot_cov = \
                    gp_ne_high_plot.predict(
                        r_fit[:, np.newaxis], return_cov=True)

                n_e['ne low97 gauss fit'][j] = ne_low_gauss_plot
                n_e['ne high97 gauss fit'][j] = ne_high_gauss_plot

            except Exception:
                if debug:
                    print('failed ne 97.5', end=', ')

        except Exception:
            if debug:
                print('failed ne data assertion', end=' ')
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(' \\\ ', exc_type, fname,
                      exc_tb.tb_lineno, end='\n')

    print('... done!', end='\n')
    if saving:
        data = mClass.dict_transf(data, to_list=False)
        with open(file + '.pickle', 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
    return (data)
