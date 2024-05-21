""" ********************************************************************** """

import sys
import os
import numpy as np
import ast
import json
import training_plot as train_plot
from scipy.optimize import curve_fit
import mClass
import requests
import webapi_access as api

Z = np.zeros
one = np.ones
M = 10000

stdwrite = sys.stdout.write
stdflush = sys.stdout.flush

""" ********************************************************************** """


def sensitivity_evaluation(
        method='weighted_deviation',
        program='20181010.032',
        FS_label='EIM_beta000',
        prad={'none': None},
        channels={'none': None},
        prio={'none': {'none': None}},
        cameras=['HBC', 'VBC'],
        amounts=[3, 3],
        plot=False,
        saving=True):
    """ Parent function for local sensitivity evaluation
    Args:
        program (0, str, optional): XPID string
        method (1, str, optional): Evaluation method
        cross_experiment (2, bool, optional): Compare experiments
        prad (3, dict, optional): Archive download 1
        channels (4, dict, optional): Archive download 2
        priority_object(5, list, optional):
    Returns:
        None.
    """
    # time lines for eval
    time_raw = channels['dimensions']
    ix_s = mClass.find_nearest(np.array(time_raw), prio['t0'])[0]
    ix_t = mClass.find_nearest(np.array(time_raw), prio['t4'])[0]

    stdwrite('\n\t\t>> Sensitivity evaluation ...\n')
    stdflush()
    for cam, length in zip(cameras, amounts):
        sensitivity_results = sensitivity_analysis(
            prio=prio, program=program, camera=cam,
            combination=length, method=method,
            id1=ix_s, id2=ix_t, plot=plot, saving=saving)

        path = '../results/COMBINATIONS/' + program + '/' + \
            method + '/' + str(length) + '_' + cam + '/'
        combination_results = fit_combination(
            results=sensitivity_results, plot=plot,
            method=method, prad=prad, saving=saving,
            channels=channels, path=path)
        sys.stdout.write('\n')
        sys.stdout.flush()
    return (sensitivity_results, combination_results)


def sensitivity_analysis(
        method='weighted_deviation',
        program='20181010.032',
        camera='HBC',
        prio={'none': {'none': None}},
        id1=1000,
        id2=9000,
        combination=3,
        saving=True,
        plot=False):
    """ channels sensitivity analysis
    Args:
        program (str, optional): XP ID
        method (str, optional): eval method used
        camera (str, optional): camera used
        combination (str, optional): combination length
        plot (bool, optional): should plot
        startID (int, optional): index to start eval
        stopID (int, optional): index to stop eval
    Returns:
        results (dict): results of sensitivity eval
    """
    # dont repeat
    file = '../results/COMBINATIONS/' + program + '/' + method + \
        '/' + str(combination) + '_' + camera + \
        '/spectrum_analysis_' + method + '.json'

    if saving:
        if (os.path.isfile(file)):
            with open(file, 'r') as f:
                results = json.load(f)
            f.close()
            try:
                if isinstance(results['stopID'], int):
                    stdwrite('\t\t\t>> (PREV) ' + camera +
                             '; C:' + str(combination))
            except Exception:
                pass

    else:
        stdwrite('\t\t\t>> ' + camera + '; C:' + str(combination))
        # setup of data dump
        spectrum = [None, None, Z((128)), Z((128))]

        # specified path for this sensitivity analysis, loading
        path = '../results/COMBINATIONS/' + program + '/' + method + \
            '/' + str(combination) + '_' + camera + '/' + method + '.dat'
        spectrum[0] =  \
            [ast.literal_eval(x) for x in np.genfromtxt(  # combinations
                path, delimiter='\t', dtype=str, usecols=[0])]
        spectrum[1] = np.loadtxt(  # each combinations quality
            path, delimiter='\t', dtype=np.float, usecols=[1])

        # statistic for all combs
        for i, cell in enumerate(spectrum[0]):
            # whether the combination is a list or int (should be array)
            if not (isinstance(cell, int)):
                if ((isinstance(cell, list)) & (len(cell) == 1)):
                    spectrum[2][cell[0]] += spectrum[1][i]
                    spectrum[3][cell[0]] += 1

                else:
                    # add the quality value for each channel in the comb
                    # and add tick per channel appearance to the total
                    for ch in cell:
                        spectrum[2][ch] += spectrum[1][i]
                        spectrum[3][ch] += 1

            else:  # if single channel just add single value
                spectrum[2][cell] += spectrum[1][i]
                spectrum[3][cell] += 1

        # averaging the total quality value of each
        # channel by its appearance count where there is some
        av_sens = (spectrum[2] / spectrum[3])[
            ~np.isnan(spectrum[2] / spectrum[3])]
        av_sense_count = sum(av_sens) / len(av_sens)
        # summed up quality average by
        # amount of combinations * amount of channels per combination
        av_sense_amnt = spectrum[2] / (len(spectrum[0]) * int(combination))

        results = {
            'XPID': program,  # XPID
            'combinations': spectrum[0],  # combinations
            'correlation': spectrum[1].tolist(),  # quality per combination
            'sum_correlation_channels':
                spectrum[2].tolist(),  # summed up quality
            'marked_hits_channels': spectrum[3].tolist(),  # appearance count
            'av_sense_channels': (spectrum[2] / spectrum[3]).tolist(),
            'av_sense_combination': av_sense_amnt.tolist(),  # see above
            'mean_correlation_combination':
                sum(spectrum[1]) / len(spectrum[1]),  # avg combination qual.
            'mean_sense_channels': av_sense_count,  # see above
            'camera': camera,  # camera
            'combination': combination,  # what the combination amount is
            'startID': int(id1),  # experiment start
            'stopID': int(id2)}  # stop of eval

        # saving
        if saving:
            with open('../results/COMBINATIONS/' + program + '/' + method +
                      '/' + str(combination) + '_' + camera +
                      '/spectrum_analysis_' + method + '.json', 'w') as f:
                json.dump(results, f, sort_keys=False, indent=4)
            f.close()

    if plot:
        try:
            train_plot.sensitivity_plot(
                method=method, results=results, prio=prio,
                camera=camera, program=program,
                combination=str(combination))
        except Exception:
            stdwrite('\n\t\t\t\\\ failed sensitivity plot ' +
                     method + ' C:' + str(combination))
    return (results)


def fit_combination(
        path='../results/COMBINATIONS/20180920.042/',
        method='weighted_deviation',
        results={'none': None, 'camera': 'HBC', 'combination': '3'},
        prad={'dimensions': Z(M), 'values': Z(M)},
        channels={'dimensions': Z(M), 'values': Z((128, M))},
        fluxsurface={'none': None},
        saving=True,
        plot=False):
    """ finding best combinations and channels from spectrum
    Args:
        results (dict, optional): combination spectrum results
        method (str, optional): used method
        prad (TYPE, optional): radiated power download json
        channels (TYPE, optional): channel voltage download json
        fluxsurface (dict, optional): fluxsurface data from VMEC webservice
        linesofsight (dict, optional): lines of sight data from geom files
        camgeo (dict, optional): camera geometry as above
        path (str, optional): path
    Returns:
        None.
    """
    base = '../results/COMBINATIONS/' + results['XPID'] + '/' + method + '/'
    file = base + str(results['combination']) + '_' + results['camera'] + \
        '/spectrum_analysis_' + method + '.json'

    if saving:
        if (os.path.isfile(file)):
            with open(file, 'r') as f:
                results = json.load(f)
            f.close()
            try:
                if isinstance(foo['fit_results']['best_cell'], list):
                    stdwrite(
                        '; (PREV) best: ' + str(results['best_channels']) +
                        ' in cell=' + str(results['best_cell']))
                    return (results)
            except Exception:
                pass

    C = int(results['combination'])
    id_1, id_2 = results['startID'], results['stopID']
    max_qual = max(results['correlation'])
    id_maxQ = (np.abs(
        [v - max_qual for v in results['correlation']])).argmin()

    # best combination as is
    cellBest = results['combinations'][id_maxQ]
    results['best_cell'] = cellBest
    # best $combination channels as from weighted sensibility
    chansBest = sorted(
        range(len(results['av_sense_channels'])),
        key=lambda x:
        [0 if np.isnan(v) else v for v in results['av_sense_channels']]
        [x])[-C:]
    results['best_channels'] = chansBest
    stdwrite('; bChans: ' + str(chansBest))
    # best $combination channels as from average sensibility
    combBest = sorted(
        range(len(results['av_sense_combination'])),
        key=lambda x:
        [0 if np.isnan(v) else v for v in results['av_sense_combination']]
        [x])[-C:]
    results['best_combination'] = combBest

    # worst combination possible
    min_qual = min(results['correlation'])
    id_minQ = (np.abs(
        [v - min_qual for v in results['correlation']])).argmin()
    cellWorst = results['combinations'][id_minQ]
    results['worst_cell'] = cellWorst

    stdwrite(
        '; maxQual=' + str(round(max_qual, 4)) + '; cell=' +
        str(cellBest))

    # see if fit is better (duh)
    channels = np.array(channels['values'])
    prad = np.array(prad['values'])

    results['fit_results'] = {}
    tags = ['best_cell', 'best_channels', 'best_combination', 'worst_cell']
    for cc, tag in zip([cellBest, chansBest, combBest, cellWorst], tags):

        if (not isinstance(cc, list)) or isinstance(cc, int) or \
           (len(cc) == 1):
            selection = np.array(channels[cc][id_1:id_2])
            L = 1

        elif isinstance(cc, list) or isinstance(cc, np.array):
            for i, ch in enumerate(cc):
                selection = \
                    np.c_[np.array(channels[ch][id_1:id_2])] if i == 0 else \
                    np.c_[selection, np.array(channels[ch][id_1:id_2])]
            L = np.shape(selection)[1]

        if L > 2:
            params, result = fit_lin(
                lines=selection.T, target=prad[id_1:id_2].T, dim=L, tag=tag)
            results['fit_results'][tag] = params.tolist()

            if plot:
                train_plot.p_lin_fit(
                    target=prad[id_1:id_2], fit=result, cell=cc,
                    lines=channels[:, id_1:id_2], parameters=params,
                    camera=results['camera'], tag=tag,
                    path=path, method=method, combination=str(C))

    if saving:
        with open(path + 'spectrum_analysis_' + method + '.json', 'w') as f:
            json.dump(results, f, sort_keys=False, indent=4)
        f.close()
    return (results)


def fit_lin(
        tag='none',
        lines=np.ones((3, M)),
        target=np.ones((M)),
        dim=3):
    """ fit_lin
    Args:
        lines (0, ndarray): Input data array
        target (1, ndarray): Target data to fit to
        dim (2, float): Dimension of lines
    Returns:
        fit (0, ndarray): Parameters returned from fit
        result (1, ndarray): Fit results with lines multiplied with params
    """
    if dim == 3:
        fit_fun = fit2
    elif dim == 4:
        fit_fun = fit3
    elif dim == 5:
        fit_fun = fit4
    elif dim == 6:
        fit_fun = fit5
    elif dim == 7:
        fit_fun = fit6
    elif dim == 8:
        fit_fun = fit7
    elif dim == 9:
        fit_fun = fit8

    if lines.dtype is not np.dtype('float64'):
        lines = lines.astype(float)
    if target.dtype is not np.dtype('float64'):
        target = target.astype(float)

    p0 = (np.ones((dim)) * 1.e9)
    b = ([0.0] * dim, [np.inf] * dim)

    try:
        fit, foo = curve_fit(
            fit_fun, (lines), target,
            p0=p0, bounds=b, maxfev=100000)
    except Exception:
        stdwrite('\n\t\t\t\t\t\\\ failed at ' + tag + ' dim=' + str(dim))
        return (np.array([0.0] * dim), np.zeros((lines.shape[1])))

    result = np.sum(fit * lines.T, axis=1)
    return fit, result


def fit2(
        lines=one((3, M)),
        param0=1.,
        param1=1.,
        param2=1.):
    """ fit2
    Args:
        lines (0, ndarray): Input data array
        param0 (1, float): Parameter
        param1 (2, float): Parameter
        param2 (3, float): Parameter
    Returns:
        param0 * lines ... (0, ndarray): Result
    """
    return param0 * lines[0] + param1 * lines[1] + param2 * lines[2]


def fit3(
        lines=one((4, M)),
        param0=1.,
        param1=1.,
        param2=1.,
        param3=1.):
    """ fit3
    Args:
        lines (0, ndarray): Input data array
        param0 (1, float): Parameter
        param1 (2, float): Parameter
        param2 (3, float): Parameter
        param3 (4, float): Parameter
    Returns:
        param0 * lines ... (0, ndarray): Result
    """
    return param0 * lines[0] + param1 * lines[1] + \
        param2 * lines[2] + param3 * lines[3]


def fit4(
        lines=one((5, M)),
        param0=1.,
        param1=1.,
        param2=1.,
        param3=1.,
        param4=1.):
    """ fit4
    Args:
        lines (0, ndarray): Input data array
        param0 (1, float): Parameter
        param1 (2, float): Parameter
        param2 (3, float): Parameter
        param3 (4, float): Parameter
    Returns:
        param0 * lines ... (0, ndarray): Result
    """
    return param0 * lines[0] + param1 * lines[1] + param2 * lines[2] + \
        param3 * lines[3] + param4 * lines[4]


def fit5(
        lines=one((6, M)),
        param0=1.,
        param1=1.,
        param2=1.,
        param3=1.,
        param4=1.,
        param5=1.):
    """ fit5
    Args:
        lines (0, ndarray): Input data array
        param0 (1, float): Parameter
        param1 (2, float): Parameter
        param2 (3, float): Parameter
        param3 (4, float): Parameter
        param4 (5, float): Parameter
    Returns:
        param0 * lines ... (0, ndarray): Result
    """
    return param0 * lines[0] + param1 * lines[1] + param2 * lines[2] + \
        param3 * lines[3] + param4 * lines[4] * param5 * lines[5]


def fit6(
        lines=one((7, M)),
        param0=1.,
        param1=1.,
        param2=1.,
        param3=1.,
        param4=1.,
        param5=1.,
        param6=1.):
    """ fit6
    Args:
        lines (0, ndarray): Input data array
        param0 (1, float): Parameter
        param1 (2, float): Parameter
        param2 (3, float): Parameter
        param3 (4, float): Parameter
        param4 (5, float): Parameter
        param5 (6, float): Parameter
        param6 (7, float): Parameter
    Returns:
        param0 * lines ... (0, ndarray): Result
    """
    return param0 * lines[0] + param1 * lines[1] + param2 * lines[2] + \
        param3 * lines[3] + param4 * lines[4] + param5 * lines[5] + \
        param6 * lines[6]


def fit7(
        lines=one((8, M)),
        param0=1.,
        param1=1.,
        param2=1.,
        param3=1.,
        param4=1.,
        param5=1.,
        param6=1.,
        param7=1.):
    """ fit7
    Args:
        lines (0, ndarray): Input data array
        param0 (1, float): Parameter
        param1 (2, float): Parameter
        param2 (3, float): Parameter
        param3 (4, float): Parameter
        param4 (5, float): Parameter
        param5 (6, float): Parameter
        param6 (7, float): Parameter
        param7 (8, float): Parameter
    Returns:
        param0 * lines ... (0, ndarray): Result
    """
    return param0 * lines[0] + param1 * lines[1] + param2 * lines[2] + \
        param3 * lines[3] + param4 * lines[4] + param5 * lines[5] + \
        param6 * lines[6] + param7 * lines[7]


def fit8(
        lines=one((9, M)),
        param0=1.,
        param1=1.,
        param2=1.,
        param3=1.,
        param4=1.,
        param5=1.,
        param6=1.,
        param7=1.,
        param8=1.):
    """ fit8
    Args:
        lines (0, ndarray): Input data array
        param0 (1, float): Parameter
        param1 (2, float): Parameter
        param2 (3, float): Parameter
        param3 (4, float): Parameter
        param4 (5, float): Parameter
        param5 (6, float): Parameter
        param6 (7, float): Parameter
        param7 (8, float): Parameter
        param8 (9, float): Parameter
    Returns:
        param0 * lines ... (0, ndarray): Result
    """
    return param0 * lines[0] + param1 * lines[1] + param2 * lines[2] + \
        param3 * lines[3] + param4 * lines[4] + param5 * lines[5] + \
        param6 * lines[6] + param7 * lines[7] + param8 * lines[8]


def cross_experiment(
        programs=['20181010.032', '20181009.046'],
        method='weighted_deviation',
        C=['ALL', 'HBC', 'VBC', 'HBC', 'VBC', 'HBC', 'VBC', 'HBC', 'VBC',
           'HBC', 'HBC_VBC', 'VBC', 'HBC', 'VBC', 'HBC', 'HBC_VBC', 'HBC'],
        A=[1, 1, 1, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 9]):
    """ cross experiment comparison of eval
    Args:
        programs (list, optional): list of programs
        method (str, optional): used eval method
        C (list, optional): all camera combinations
        A (list, optional): all number combinations
    Returns:
        None.
    """
    stdflush()
    stdwrite('\n\t>> comparing P0:' + programs[0])
    for j, s in enumerate(programs[1:]):
        stdwrite(' and P' + str(j + 1) + ':' + s)
    stdwrite('\n')
    stdflush()

    base_URI = 'http://archive-webapi.ipp-hgw.mpg.de/'
    base = '../results/COMBINATIONS/'
    results = []
    data = [[]] * (len(programs) * 2)
    for k, P in enumerate(programs):

        i = 0
        for c, a in zip(C, A):
            loc = base + P + '/' + method + '/' + str(a) + '_' + c + '/' + \
                'spectrum_analysis_' + method + '.json'

            stdwrite('\r\t\t>> reading P' + str(k) + ': ' + str(i * '.'))

            with open(loc, 'r') as f:
                results.append(json.load(f))
            f.close()
            i += 1
        stdwrite('\n')
        stdflush()

        req = requests.get(base_URI + 'programs.json' + '?from=' + P)
        program_info = req.json()

        arch = 'Test/raw/W7XAnalysis/QSB_Bolometry/'
        data_req = base_URI + arch + 'PradHBCm_DATASTREAM/V2/0/PradHBCm/'
        data[(k * 2) + 0] = api.download_single(
            data_req=data_req, program_info=program_info, debug=False)
        data_req = base_URI + arch + 'BoloAdjusted_DATASTREAM/V2/'
        data[(k * 2) + 1] = api.download_single(
            data_req=data_req, program_info=program_info, debug=False)

    stdwrite('\n>> flushing to plot...')
    stdflush()

    train_plot.av_sense_X(programs=programs, results=results, method=method)

    loc = '../results/INVERSION/'
    # lines of sight
    f = loc + 'CAMGEO/ez_losight.json'
    with open(f, 'r') as file:
        injson = json.load(file)
        ez_losight = mClass.dict_transf(
            injson, list_bool=False, prints=False)
    file.close()

    # camera geometry
    f = loc + 'CAMGEO/camera_geometry.json'
    with open(f, 'r') as file:
        injson = json.load(file)
        camera_geometry = mClass.dict_transf(
            injson, list_bool=False, prints=False)
    file.close()

    train_plot.best_chans_X(
        programs=programs, results=results, method=method, data=data,
        ez_losight=ez_losight, camera_geometry=camera_geometry)

    return
