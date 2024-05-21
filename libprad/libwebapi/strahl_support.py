""" ********************************************************************** """

import sys
import numpy as np
import scipy as sp
import requests
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import LSQUnivariateSpline

import mClass
import read_calculation_corona as rcc
import thomsonS_access as TS
import plot_funcs as plot_funcs

Z = np.zeros
M = 10000
stdwrite = sys.stdout.write

""" ********************************************************************** """


def fit_strahl_profiles(
        interpolation_method='exppol0',
        rho_grid=None,
        data_grid=None):
    y0 = data_grid[0]

    def exppol0_func(
            rho=1.0,
            p0=1.0,
            p1=1.0,
            p2=1.0,
            p3=1.0):
        return (y0 * np.exp(
            p0 * rho**2 + p1 * rho**4 + p2 * rho**6 * p3 * rho**8))

    def exppol1_func(
            rho=1.0,
            p0=1.0,
            p1=1.0,
            p2=1.0,
            p3=1.0):
        return (y0 * np.exp(
            p0 * rho**2 + p1 * rho**3 + p2 * rho**4 * p3 * rho**5))

    def ratfun_func(
            rho=1.0,
            p0=1.0,
            p1=1.0,
            p2=1.0,):
        return (y0 * ((1 - p0) * (1 - rho**p1)**p2 + p0))

    if interpolation_method in ['exppol0', 'exppol1']:
        default = (1.0, 1.0, 1.0, 1.0)
        if interpolation_method == 'exppol0':
            func = exppol0_func
        elif interpolation_method == 'exppol1':
            func = exppol1_func
    elif interpolation_method == 'ratfun':
        func, default = ratfun_func, (1.0, 1.0, 1.0)

    results = curve_fit(
        func,
        rho_grid, data_grid, p0=default,
        sigma=None, absolute_sigma=None,
        check_finite=True, method='lm')

    if interpolation_method in ['exppol0', 'exppol1']:
        p0, p1, p2, p3 = results[0]
        results = results[0]
        if interpolation_method == 'exppol0':
            y = exppol0_func(rho=rho_grid, p0=p0, p1=p1, p2=p2, p3=p3)
        elif interpolation_method == 'exppol1':
            y = exppol1_func(rho=rho_grid, p0=p0, p1=p1, p2=p2, p3=p3)
    elif interpolation_method == 'ratfun':
        p0, p1, p2, = results[0]
        y = ratfun_func(rho=rho_grid, p0=p0, p1=p1, p2=p2)
        results = np.append(results[0], results[0][-1])

    return (y0, results, y)


def interp_smooth(
        y=Z((50)),
        x=Z((50)),
        method='spline',
        k=3,
        knots=[0.7],
        N=24,
        M=5,
        E=1.0,
        lcfs_last=0.8,
        debug=False):
    """ spline smoothing, inter/extrapolation of TS profiles for STRAHL
    Keyword Arguments:
        y {[type]} -- input data (default: {Z((50))})
        x {[type]} -- input grid (default: {Z((50))})
        method {str} -- smoothing/interpolation method (default: {'spline'})
        k {int} -- order of spline (default: {3})
        N {int} -- interpolation grid points (default: {24})
        lcfs_last {float} -- LCFS value percentage (default: {0.8})
        debug {bool} -- debugging (default: {False})
    Returns:
        V {ndarray} -- new dimension smoothed/interpolated
        W {ndarray} -- new data smoothed/interpolated
        X {ndarray} -- base dimension
    """
    X = np.linspace(0.0, 1.0, N)
    if debug:
        print('x=', np.shape(x), x, '\n', 'X=', np.shape(X), X)

    if y[0] <= np.mean(y[1:3]):
        if debug:
            print('y0=', y[0], '_y[1:3]=', np.mean(y[1:3]))
        y[0] = np.mean(y[1:3])

    if lcfs_last is not None:
        if (np.max(x) >= E):
            idx, val = mClass.find_nearest(array=np.array(x), value=E)
            x, y = x[0:idx], y[0:idx]

        x = np.append(x, 1.0)
        y = np.append(y, y[-1] * lcfs_last)

    if (method == 'spline'):
        # obj = UnivariateSpline(
        #     x=x, y=y, k=k, s=None, w=None, ext='extrapolate')
        # obj = InterpolatedUnivariateSpline(
        #     x=x, y=y, k=k, s=None, w=None, ext='extrapolate')
        obj = LSQUnivariateSpline(
            x=x, y=y, k=k, t=knots, w=None, ext='extrapolate')

        W = obj(X)
        if debug:
            print('y=', np.shape(y), y, '\n', 'W=', np.shape(W), W)

    elif (method == 'average'):
        y = interp1d(
            x=np.insert(x, 0, 0.0),
            y=np.insert(y, 0, y[0]),
            fill_value=(y[0], y[-1]))(X)

        W = np.convolve(y, np.ones((M)) / M, mode='same')
        if debug:
            print('y=', np.shape(y), y, '\n', 'W=', np.shape(W), W)

    elif (method == 'none'):
        X = np.insert(x, 0, 0.0)
        W = np.insert(y, 0, y[0])
    else:
        X = x
        W = y

    return (W, X)


def print_strahl_info(
        shotno='20181010.032',
        time=[7.1385],
        vmecID='1000_1000_1000_1000_+0000_+0000/01/00jh_l/',
        ne_mode='meas',
        ne_smoothing='spline',
        ne_interpolation_method='interp',
        Te_mode='meas',
        Te_smoothing='spline',
        Te_interpolation_method='interp',
        lcfs_ne=0.8,
        lcfs_Te=0.33,
        ne_order=3,
        Te_order=2,
        knots_Te=[0.7],
        knots_ne=[0.7],
        N=24,
        M=5,
        decay_length=5.0,
        printing=True,
        debug=False):
    """ Prints STRAHL information regarding the XPID etc
    Keyword Arguments:
        shotno {str} -- XPID (def {'20181010.032'})
        time {list} -- time vector (def {[7.0, 7.1385, 7.3]})
        vmecID {str} -- (def {'1000_1000_1000_1000_+0000_+0000/01/00jh_l/'})
        mode {str} -- data mode/source (def {'fit'})
        smoothin {str} -- smoothing method (def {'spline'})
        lcfs_last {float} -- LCFS value percentage (default: {0.8})
        decay_length {float} -- decay length in cm (default: {5.0})
        debug {bool} -- debugging (default: {True})
    Return:
        None.
    """
    try:
        magax = requests.get(
            'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/' +
            vmecID + 'magneticaxis.json?phi=108').json()
    except Exception:
        print('\t\\\ failed magax')
        magax = {'magneticAxis': {  # EIM beta = 0.0
            'x1': 5.204812678833707, 'x3': 9.978857072810671e-17}}

    TS_profile, T = TS.return_TS_profile_for_t(
        shotno=shotno, scaling='gauss', t=time)
    if TS_profile is None:
        print('\t\\\ failed TS profile')
        return

    if isinstance(time, float):
        time = [time]

    try:
        m_R = requests.get(
            'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/' +
            vmecID + 'minorradius.json').json()['minorRadius']
    except Exception:
        print('\t\\\ failed minorradius')
        m_R = TS_profile[0]['minor radius']

    if (ne_mode == 'meas'):
        ne_rad_OG = [x / m_R for i, x in enumerate(
                     TS_profile[0]['r without outliers (ne)'])
                     if ((x > 0.0) and (x - TS_profile[0][
                         'r without outliers (ne)'][i - 1] > 0.01))]
    elif (ne_mode == 'fit'):
        ne_rad_OG = [
            x / m_R for i, x in enumerate(
                TS_profile[0]['r_fit']) if x > 0.0]

    if (Te_mode == 'meas'):
        Te_rad_OG = [x / m_R for i, x in enumerate(
                     TS_profile[0]['r without outliers (Te)'])
                     if ((x > 0.0) and (x - TS_profile[0][
                         'r without outliers (Te)'][i - 1] > 0.01))]
    elif (Te_mode == 'fit'):
        Te_rad_OG = [
            x / m_R for i, x in enumerate(
                TS_profile[0]['r_fit']) if x > 0.0]

    if printing:
        print(
            '\n' +
            '#   T I M E S T E P S\n' +
            'cv  # number of changes (start-time+ ... +stop-time)\n' +
            '    ' + str(int(len(T))) +
            '\n\n' +
            'cv  # time     dt at start    increase of dt after cycle' +
            '     steps per cycle')

        for j, t in enumerate(T):
            print(
                '    ' + str(format(t, '.4f')) + '     1.0e-04' +
                ' ' * 8 + '1.0' + ' ' * 28 + '5')

        print(
            '\n\n' +
            'cv  # rho volume(LCFS)[cm]   R_axis[cm]   U_loop[V]' +
            '    time[s]\n' +
            '    ' + str(format(m_R * 100, '.4f')) +
            ' ' * 18 + str(format(magax['magneticAxis']['x1'] * 100, '.5f')) +
            '    0.' + ' ' * 11 + str(round(T[0], 4)) +
            '\n\n' +
            '# NE\n' +
            'cv  # time-vector [s]\n' +
            '    ' + str(int(len(T))) + '\n' + '    ' +
            str([format(x, '.3f') for x in T]).replace(
                '[', '').replace(']', '').replace(',', ' ').replace("'", '') +
            '\n\n' +
            'cv  # ne function\n' +
            "    '" + ne_interpolation_method + "'" +
            '\n\n' +
            'cv  # interpolation coordinates\n' +
            "    'volume rho'" +
            '\n')

    ne_rad = [ne_rad_OG, None]
    ne_maps = [[], []]
    for j, t in enumerate(T):
        if (ne_mode == 'meas'):
            ne = [
                TS_profile[j]['ne map without outliers'][i] for i, x in
                enumerate(TS_profile[0]['r without outliers (ne)'])
                if ((x > 0.0) and (x - TS_profile[0][
                    'r without outliers (ne)'][i - 1] > 0.01))]
        elif (ne_mode == 'fit'):
            ne = [
                TS_profile[j]['ne fit gauss'][i] for i, x in
                enumerate(TS_profile[j]['r_fit'])
                if x > 0.0]
        ne_maps[0].append(ne)

        if (ne_smoothing is not None):
            y, x = interp_smooth(
                x=ne_rad_OG, y=ne, N=N, debug=False, M=M,
                lcfs_last=lcfs_ne, method=ne_smoothing, k=ne_order,
                knots=knots_ne)[0:2]
        else:
            x, y = np.array(ne_rad_OG), np.array(ne)

        if printing:
            if ne_interpolation_method in ['interp', 'interpa']:
                if (j == 0):
                    print(
                        'cv  # of ne interpolation points\n' +
                        '    ' + str(len(x)) + 
                        '\n\ncv  # x-grid for ne-interpolation\n' +
                        '    ' + str([format(v, '.9f') for v in x]).replace(
                            '[', '').replace(']', '').replace(
                                ',', ' ').replace("'", '') +
                        '\n\ncv  # ne data')
                print(
                    '    ' + str(format(y[0] * 1e13, '.5e')) + '  ' +
                    str([format(v / y[0], '.9f') for v in y]).replace(
                        '[', '').replace(']', '').replace(
                            ',', ' ').replace("'", ''))

            elif ne_interpolation_method in ['exppol0', 'ratfun', 'exppol1']:
                y0, results, y = fit_strahl_profiles(
                    interpolation_method=ne_interpolation_method,
                    rho_grid=x, data_grid=y)

                if (j == 0):
                    print(
                        'cv  # ne ' + ne_interpolation_method + ' values')
                print(
                    '    ' + str(format(y0 * 1e13, '.5e')) + '  ' +
                    str([format(p, '.9f') for p in results]).replace(
                        '[', '').replace(']', '').replace(
                            ',', ' ').replace("'", ''))

        ne_maps[1].append(y)
        ne_rad[1] = x

    if printing:
        print(
            '\ncv  # ne decay length\n' +
            '    ' + str(decay_length) + ('  ' + str(decay_length)) * (
                len(T) - 1) + '\n\n' +
            '# TE\n' +
            'cv  # time-vector [s]\n' +
            '    ' + str(int(len(T))) + '\n' +
            '    ' + str([format(x, '.3f') for x in T]).replace(
                '[', '').replace(']', '').replace(',', ' ').replace("'", '') +
            '\n\n' +
            'cv  # Te function\n' +
            "    '" + Te_interpolation_method + "'" +
            '\n\n' +
            'cv  # interpolation coordinates\n' +
            "    'volume rho'" +
            '\n')

    Te_rad = [Te_rad_OG, None]
    Te_maps = [[], []]
    for j, t in enumerate(T):
        if (Te_mode == 'meas'):
            Te = [
                TS_profile[j]['Te map without outliers'][i] for i, x in
                enumerate(TS_profile[0]['r without outliers (Te)'])
                if ((x > 0.0) and (x - TS_profile[0][
                    'r without outliers (Te)'][i - 1] > 0.01))]
        elif (Te_mode == 'fit'):
            Te = [
                TS_profile[j]['Te fit gauss'][i] for i, x in
                enumerate(TS_profile[j]['r_fit'])
                if x > 0.0]
        Te_maps[0].append(Te)

        if (Te_smoothing is not None):
            y, x = interp_smooth(
                x=Te_rad_OG, y=Te, N=N, M=M, debug=False,
                method=Te_smoothing, lcfs_last=lcfs_Te,
                k=Te_order, knots=knots_Te)[0:2]
        else:
            x, y = np.array(Te_rad_OG), np.array(Te)

        if printing:
            if Te_interpolation_method in ['interp', 'interpa']:
                if (j == 0) and printing:
                    print(
                        'cv  # of Te interpolation points\n' +
                        '    ' + str(len(x)) +
                        '\n\n' +
                        'cv  # x-grid for Te-interpolation\n' +
                        '    ' + str([format(v, '.5f') for v in x]).replace(
                            '[', '').replace(']', '').replace(',', ' ').replace(
                                "'", '') +
                        '\n\ncv  # Te data')
                print(
                    '    ' + str(format(y[0] * 1e3, '.2f')) + '  ' +
                    str([format(v / y[0], '.5f') for v in y]).replace(
                        '[', '').replace(']', '').replace(
                            ',', ' ').replace("'", ''))

            elif Te_interpolation_method in ['exppol0', 'ratfun', 'exppol1']:
                y0, results, y = fit_strahl_profiles(
                    interpolation_method=Te_interpolation_method,
                    rho_grid=x, data_grid=y)

                if (j == 0):
                    print(
                        'cv  # Te ' + Te_interpolation_method + ' values')
                print(
                    '    ' + str(format(y0 * 1e3, '.2f')) + '  ' +
                    str([format(p, '.5f') for p in results]).replace(
                        '[', '').replace(']', '').replace(
                            ',', ' ').replace("'", ''))

        Te_maps[1].append(y)
        Te_rad[1] = x

    if printing:
        print(
            '\ncv  # Te decay length\n' +
            '    ' + str(decay_length) + ('  ' + str(decay_length)) * (
                len(T) - 1))

    if debug:
        print('ne_rad=', np.shape(ne_rad), '\nne_maps=', np.shape(ne_maps))
        print('Te_rad=', np.shape(Te_rad), '\nTe_maps=', np.shape(Te_maps))

    plot_funcs.strahl_info_plot(
        time=T, ne_maps=ne_maps, ne_rad=ne_rad,
        te_maps=Te_maps, te_rad=Te_rad,
        ne_smooth=ne_smoothing, Te_smooth=Te_smoothing)

    return
