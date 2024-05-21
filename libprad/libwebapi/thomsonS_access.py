""" *************************************************************************
    so HEADER """

import thomson_scattering as ThomsonS
import TS_to_json as TSjson
import plot_funcs
import mClass
import pickle

import warnings
import sys
import os
import numpy as np
import json
import glob
import archivedb

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")

Z = np.zeros
stdwrite = sys.stdout.write
stdflush = sys.stdout.flush

""" eo header
************************************************************************** """


def thomson_scaled_grab(
        shotno='20181018.041',
        debug=False,
        scaling='gauss',
        evalStart=None,
        evalEnd=None,
        saving=True,
        plot=False,
        indent_level='\t'):
    """ scaled thomson scattering map wrapper
    Args:
        shotno (str, optional): XP ID
        debug (bool, optional): Should print debugging info
        scaling (str, optional): Method of scaling
        indent_level (str, optional): Printing indetation
    Returns:
        None.
    Notes:
        None.
    """
    if debug:
        print(indent_level + '>> Starting thomson scattering base routine')
    location = \
        '../results/THOMSON/' + shotno[0:8] + '/' + shotno[-3:] + '/'
    if not os.path.exists(location):
        os.makedirs(location)

    foo = glob.glob(
        location + 'thomson_data_' + shotno + '_' + scaling + '_*')

    if foo == []:  # general data
        print(indent_level + '\t\\\ TS data not found, calculating')
        data, names, dtypes, units, info = ThomsonS.ts_data_full_analysis(
            shotno=shotno, debug=debug, scaling=scaling,
            indent_level=indent_level + '\t')
        # get filename after the fact
        filename = glob.glob(
            '../results/THOMSON/' + shotno[0:8] + '/' + shotno[-3:] +
            '/thomson_data_' + shotno + '_' + scaling + '_*')[0]

    else:
        filename = foo[0]
        if debug:
            print(indent_level +
                  '\t\\\ TS data file found, loading from ' + filename[-39:])

    TS_data = TSjson.TS_json(
        shotno=shotno, scaling=scaling,
        filename=filename, debug=debug,
        saving=saving, indent_level=indent_level)
    avg_TS_data = average_TSprofile(
        shotno=shotno, TS=TS_data,
        evalStart=evalStart, evalEnd=evalEnd,
        saving=saving, indent_level=indent_level)

    if plot:
        plot_funcs.thomson_plot(
            shotno=shotno, TS_data=TS_data,
            evalStart=evalStart, evalEnd=evalEnd,
            neplot=True, teplot=True, debug=debug,
            rewrite=True, indent_level=indent_level)
        plot_funcs.plot_avg_TS_profiles(
            shotno=shotno, avg_TS_data=avg_TS_data,
            debug=debug, indent_level='\t')

    return (TS_data, avg_TS_data)


def return_TS_profile_for_t(
        shotno='20181010.032',
        scaling='gauss',
        t=0.1,
        debug=False):
    """returns closest TS profile for given time
    Args:
        shotno (str, optional): Program ID
        debug (bool, optional): Debugging bool
        scaling (str, optional): Scaling method
        t (float, optional): Time to look for
    Returns:
        data (list): of dicts with ne and Te profiles with radii
    Notes:
        None.
    """
    file = '../results/THOMSON/' + shotno[0:8] + \
        '/' + shotno[-3:] + '/TS_profile_' + \
        shotno.replace('.', '_')

    if not glob.glob(file + '.pickle') == []:
        print('\t\\\ TS file found, loading ' + file[-36:])
        try:
            with open(file + '.pickle', 'rb') as f:
                TS_data = pickle.load(f)
            f.close()
        except Exception:
            print(indent_level + '\t\\\ failed loading TS file')
            return (None, None)

    else:
        if foo == []:  # general data
            print('\t\\\ TS data not found, calculating')
            data, names, dtypes, units, info = ThomsonS.ts_data_full_analysis(
                shotno=shotno, debug=debug, scaling=scaling,
                indent_level=indent_level + '\t')
            # get filename after the fact
            filename = glob.glob(
                '../results/THOMSON/' + shotno[0:8] + '/' + shotno[-3:] +
                '/thomson_data_' + shotno + '_' + scaling + '_*')[0]
            # put into format
            TS_data = TSjson.TS_json(
                shotno=shotno, scaling=scaling,
                filename=filename, debug=debug,
                saving=True, indent_level=indent_level + '\t')

    # got TS data? got milk?
    if TS_data is not None:
        if isinstance(t, float):
            time = [t]
        else:
            time = t

        data, T = [], []
        for t in time:
            i, val = mClass.find_nearest(
                TS_data['time'], t * 1e9 + archivedb.get_program_t1(shotno))

            data.append(
                {'minor radius': TS_data['minor radius'][i],
                 'r_fit': TS_data['r_fit'][i],
                 's_vmec': TS_data['s_vmec'][i],
                 'r without outliers (ne)':
                    TS_data['values']['n_e']['r without outliers (ne)'][i],
                 'ne map without outliers':
                    TS_data['values']['n_e']['ne map without outliers'][i],
                 'r without outliers (Te)':
                    TS_data['values']['T_e']['r without outliers (Te)'][i],
                 'Te map without outliers':
                    TS_data['values']['T_e']['Te map without outliers'][i],
                 'ne fit gauss':
                    TS_data['values']['n_e']['ne fit gauss'][i],
                 'Te fit gauss':
                    TS_data['values']['T_e']['Te fit gauss'][i]})
            T.append((val - archivedb.get_program_t1(
                shotno, useCache=False)) / 1e9)
        return (data, T)

    else:
        print('\t\\\ no TS data acqusition for', shotno)
        return (None, t)


def average_TSprofile(
        shotno='20181018.041',
        TS={'none': None},
        evalStart=None,
        evalEnd=None,
        debug=False,
        saving=False,
        indent_level='\t'):
    """ take the json dump and process average profiles + min/max errors
    Args:
        shotno (str, optional): XP ID
        TS_data (dict, optional): thomson scattering data from json
        evalStart (float, optional): time point to start evaluation (s f. T1)
        evalEnd (float, optional): time point to stop evaluation
        debug (bool, optional): Debug printing bool
        indent_level (str, optional): Printing indentation
    Returns:
        None.
    Notes:
        None.
    """
    # set up time limits
    t1 = int(archivedb.get_program_t1(shotno, useCache=False))
    t0 = int(archivedb.get_program_from_to(shotno, useCache=False)[0])
    t5 = int(archivedb.get_program_from_to(shotno, useCache=False)[1])
    evalStart = int(t1 + evalStart * 1e9) \
        if (evalStart is not None) else t1
    evalEnd = int(t1 + evalEnd * 1e9) \
        if (evalEnd is not None) else t5

    file = '../results/THOMSON/' + shotno[0:8] + \
        '/' + shotno[-3:] + '/avg_TS_profile_' + \
        shotno.replace('.', '_') + '_S' + \
        str(round((evalStart - t1) / 1e9, 3)) + '_E' + \
        str(round((evalEnd - t1) / 1e9, 3))

    if saving:
        if not glob.glob(file + '.pickle') == []:
            print(indent_level + '\t\\\ average TS found, loading ... ')
            try:
                with open(file + '.pickle', 'rb') as f:
                    avg_TS_data = pickle.load(f)
                f.close()
                return (avg_TS_data)

            except Exception:
                print(indent_level + '\t\\\ failed loading TS json')
                return None

    T0 = np.unique(TS['time'])
    t_ind = np.where((T0 <= evalEnd) & (T0 >= evalStart))[0]
    T = T0[(T0 <= evalEnd) & (T0 >= evalStart)]

    # for loop easier access
    M, perc_old = len(T), -1
    v = TS['values']

    # define ROI and the approx. width of the
    # r intervall where Te/ne is defined without outliers
    S, E = \
        np.min(np.where((T0 <= evalEnd) & (T0 >= evalStart))), \
        np.max(np.where((T0 <= evalEnd) & (T0 >= evalStart)))
    l_te, l_ne = \
        np.max([np.shape(x) for x in v['T_e']['r without outliers (Te)']]), \
        np.max([np.shape(x) for x in v['n_e']['r without outliers (ne)']])
    l_te = l_te if isinstance(l_te, np.int32) else l_te[0]
    l_ne = l_ne if isinstance(l_ne, np.int32) else l_ne[0]

    # bining of radii and then averaging later
    r_no_te, r_no_ne = \
        [(np.max(x), np.min(x))
         for x in v['T_e']['r without outliers (Te)'][S:E]], \
        [(np.max(x), np.min(x))
         for x in v['n_e']['r without outliers (ne)'][S:E]]

    bin_r_te, bin_r_ne = \
        np.linspace(np.min(r_no_te), np.max(r_no_te), l_te), \
        np.linspace(np.min(r_no_ne), np.max(r_no_ne), l_ne)

    # set up json to save and hand over later
    avg_TS_data = {
        'time': T,
        'minor radius': np.mean(TS['minor radius']),
        'r_fit': TS['r_fit'][0],
        'evalStart': round((evalStart - t1) / 1e9, 3),
        'evalEnd': round((evalEnd - t1) / 1e9, 3),
        'values': {
            'n_e': {
                'bin r ne': bin_r_ne,
                'ne gauss': np.zeros((len(TS['r_fit'][0]))),
                'ne gauss conv': np.zeros((len(TS['r_fit'][0]))),
                'ne map': np.zeros((np.shape(bin_r_ne)[0])),
                'ne map high97': np.zeros((np.shape(bin_r_ne)[0])),
                'ne map low97': np.zeros((np.shape(bin_r_ne)[0])),
            },
            'T_e': {
                'bin r te': bin_r_te,
                'Te gauss': np.zeros((len(TS['r_fit'][0]))),
                'Te gauss conv': np.zeros((len(TS['r_fit'][0]))),
                'Te map': np.zeros((np.shape(bin_r_te)[0])),
                'Te map high97': np.zeros((np.shape(bin_r_te)[0])),
                'Te map low97': np.zeros((np.shape(bin_r_te)[0]))
            }
        }
    }

    import matplotlib.pyplot as p

    t_e, n_e = v['T_e'], v['n_e']
    print(indent_level + '>> averaging profiles:', end=' ')
    for k, j in enumerate(t_ind):  # in region of interest
        i = T0[k]

        if debug:
            print('\n', indent_level, '\tT=',
                  round((i - t1) / 1e9, 4), end='s, ')
        else:
            if (round(j / M, 2) % 0.01 == 0.0) and \
               (int(j / M * 100.) is not perc_old):
                print(str(int(j / M * 100.)) + '%', end=' ')
                perc_old = int(j / M * 100.)

        # Te map and confidence
        avg_TS_data['values']['T_e']['Te map'] += avg_map_interp(
            target=avg_TS_data['values']['T_e']['bin r te'],
            proj=t_e['r without outliers (Te)'][j],
            profile=t_e['Te map without outliers'][j])
        avg_TS_data['values']['T_e']['Te map low97'] += avg_map_interp(
            target=avg_TS_data['values']['T_e']['bin r te'],
            proj=t_e['r without outliers (Te)'][j],
            profile=t_e['Te low97 without outliers'][j])
        avg_TS_data['values']['T_e']['Te map high97'] += avg_map_interp(
            target=avg_TS_data['values']['T_e']['bin r te'],
            proj=t_e['r without outliers (Te)'][j],
            profile=t_e['Te high97 without outliers'][j])
        # Te gauss fit and confidence
        avg_TS_data['values']['T_e']['Te gauss'] += avg_gauss_fit(
            gauss_fit=t_e['Te fit gauss'][j])
        avg_TS_data['values']['T_e']['Te gauss conv'] += avg_gauss_conv(
            gauss_conv=t_e['Te fit gauss conv'][j])

        # ne map and confidence
        avg_TS_data['values']['n_e']['ne map'] += avg_map_interp(
            target=avg_TS_data['values']['n_e']['bin r ne'],
            proj=n_e['r without outliers (ne)'][j],
            profile=n_e['ne map without outliers'][j])
        avg_TS_data['values']['n_e']['ne map low97'] += avg_map_interp(
            target=avg_TS_data['values']['n_e']['bin r ne'],
            proj=n_e['r without outliers (ne)'][j],
            profile=n_e['ne low97 without outliers'][j])
        avg_TS_data['values']['n_e']['ne map high97'] += avg_map_interp(
            target=avg_TS_data['values']['n_e']['bin r ne'],
            proj=n_e['r without outliers (ne)'][j],
            profile=n_e['ne high97 without outliers'][j])
        # ne gauss fit and confidence
        avg_TS_data['values']['n_e']['ne gauss'] += avg_gauss_fit(
            gauss_fit=n_e['ne fit gauss'][j])
        avg_TS_data['values']['n_e']['ne gauss conv'] += avg_gauss_conv(
            gauss_conv=n_e['ne fit gauss conv'][j])

    print('... done!', end='\n')
    if saving:
        with open(file + '.pickle', 'wb') as f:
            pickle.dump(
                avg_TS_data, f,
                protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
    return (avg_TS_data)


def avg_gauss_conv(
        gauss_conv=np.zeros((40, 40)),
        debug=False):
    """ return gauss convenience area
    Args:
        gauss_conv (TYPE, optional):
        debug (bool, optional):
    Returns:
        gauss_conv (ndarray): convenience area
    """
    return np.sqrt(gauss_conv)


def avg_gauss_fit(
        gauss_fit=np.zeros((40)),
        debug=False):
    """ return gauss fit
    Args:
        gauss_fit (TYPE, optional):
        debug (bool, optional):
    Returns:
        gauss_fit (ndarray): gauss fit
    """
    return gauss_fit


def avg_map_interp(
        target=np.zeros((40)),
        proj=np.zeros((35)),
        profile=np.zeros((35)),
        debug=False):
    """ return average maps
    Args:
        target (TYPE, optional):
        proj (TYPE, optional):
        profile (TYPE, optional):
        debug (bool, optional):
    Returns:
        interp_map (ndarray): interpolated map on basic grid
    """
    target = np.nan_to_num([x if x is not None else np.nan for x in target])
    proj = np.nan_to_num([x if x is not None else np.nan for x in proj])
    profile = np.nan_to_num([x if x is not None else np.nan for x in profile])
    return np.interp(
        target + np.abs(np.min(target)), proj + np.abs(np.min(proj)),
        profile)


def weights_up_down(
        target=np.zeros((35)),
        up=np.zeros((35)),
        down=np.zeros((35)),
        debug=False):
    """ weighting factors for map avergaging in convenience area
    Args:
        target (TYPE, optional):
        up (TYPE, optional):
        down (TYPE, optional):
        debug (bool, optional):
    Returns:
        weights (ndarray): weighting factors
    """
    weights = np.ones((np.shape(target)[0]))
    for i, v in enumerate(target):
        if (up[i] != 0.0) and (down[i] != 0.0):
            uF = 1 - (up[i] - v) / v if (np.abs(up[i]) < 2 * v) \
                else 0.0
            dF = 1 - (v - down[i]) / v if (down[i] > 0.0) \
                else 0.0
            weights[i] = np.sqrt(uF**2 + dF**2)
        print(weights, '\n', np.mean(weights))
    return weights


def weights_left_right(
        target=np.zeros((40)),
        proj=np.zeros((35)),
        debug=False):
    """ weighting factors for map averaging for average grid
    Args:
        target (TYPE, optional):
        proj (TYPE, optional):
        debug (bool, optional):
    Returns:
        weights (ndarray): weighting factors
    """
    # for each point in old shape, calculated weights for new map
    weights = np.zeros((np.shape(proj)[0], np.shape(target)[0]))
    for i, v in enumerate(proj):

        # if not first and not last point
        if (i != 0) and (i != np.shape(proj)[0] - 1):
            k = np.where(target >= proj[i - 1])[0][0] - 1  # next bin left
            m = np.where(target >= proj[i + 1])[0][0]  # next bin right
        elif i == 0:  # if first point
            k = np.where(target >= proj[i])[0][0]  # next bin left
            m = np.where(target >= proj[i + 1])[0][0]  # next bin right
        elif i == np.shape(proj)[0] - 1:  # if last point
            k = np.where(target >= proj[i - 1])[0][0] - 1  # next bin left
            m = np.where(target >= proj[i])[0][0]  # this bin

        # avoid wrapping around the array
        k = k if k != -1 else 0
        m = m if m != -1 else 0

        # contribution to left and right compartments
        a, q = \
            1 - (v - target[k]) / (target[m] - target[k]), \
            1 - (target[m] - v) / (target[m] - target[k])

        if (k != m - 1):  # if not only one compartment
            for n in range(k, m + 1):
                if debug:
                    print('i:', i, 'k:', k, 'm:', m, 'n:', n, end=' >> ')

                if (n != m):  # not the right most bin
                    # bin is left of v
                    if (target[n] < v) and (target[n + 1] < v):
                        if debug:
                            print('tn < tn+1 < v', end=' ')
                        weights[i, n] += \
                            a * (1 - (target[n + 1] - target[n]) /
                                 (v - target[k]))
                        if debug:
                            print(weights[i, n])

                    # v is in bin
                    elif (target[n] < v < target[n + 1]):
                        if debug:
                            print('tn < v < tn+1', end=' ')
                        if (n == k):
                            weights[i, n] += a
                        else:
                            weights[i, n] += \
                                a * (1 - (v - target[n]) /
                                     (v - target[k]))
                        if debug:
                            print(weights[i, n])

                    # v is left of bin
                    elif (v < target[n] < target[n + 1]):
                        if debug:
                            print('v < tn < tn+1', end=' ')
                        # v is in bin left to the current
                        if (target[n - 1] < v < target[n]):
                            weights[i, n] += \
                                q * (1 - (target[n] - v) /
                                     (target[m] - v))
                        else:  # v is somewhere else left of bin
                            weights[i, n] += \
                                q * (1 - (target[n + 1] - target[n]) /
                                     (target[m] - v))
                        if debug:
                            print(weights[i, n])

                    # right bin is exactly v
                    elif (target[n + 1] == v):
                        if debug:
                            print('tn+1 == v', end=' ')
                        weights[i, n] += q
                        if debug:
                            print(weights[i, n])

                    # v is exactly bin
                    elif (target[n] == v):
                        if debug:
                            print('tn == v', end=' ')
                        weights[i, n] += a
                        if debug:
                            print(weights[i, n])

                    else:  # catch distrusions
                        if debug:
                            print('exception n != m')

                elif (n == m):  # last point

                    # v is in last bin
                    if (target[n - 1] < v) and (target[n] > v):
                        if debug:
                            print('tn-1 < v < tn', end=' ')
                        weights[i, n] += q
                        if debug:
                            print(weights[i, n])

                    # v is left of last bin
                    elif (target[n - 1] > v) and (target[n] > v):
                        if debug:
                            print('v < tn-1 < tn', end=' ')
                        weights[i, n] += \
                            q * (1 - (target[n] - target[n - 1]) /
                                 (target[m] - v))
                        if debug:
                            print(weights[i, n])

                    # v is exactly last point
                    elif (target[n] == v):
                        if debug:
                            print('tn == v', end=' ')
                        weights[i, n] += q
                        if debug:
                            print(weights[i, n])

                    # v is exactly second to last bin
                    elif (target[n - 1] == v):
                        if debug:
                            print('tn-1 == v', end=' ')
                        weights[i, n] += q
                        if debug:
                            print(weights[i, n])

                else:  # catch exceptions, WTF
                    if debug:
                        print('exception n ?? m')

        elif (k == m - 1):  # it is only in one bin, easy
            if debug:
                print('i:', i, 'k:', k, 'm:', m,
                      '     >> k == m - 1', end=' ')
            weights[i, k], weights[i, m] = a, q
            if debug:
                print(weights[i, k:m + 1])

        if debug:  # check sum and rescale so integral is 1
            print('i:', i, 'k:', k, 'm:', m,
                  '\t\t', round(np.sum(weights[i, :]), 3))
        weights[i, :] = weights[i, :] / np.sum(weights[i, :])

    return weights
