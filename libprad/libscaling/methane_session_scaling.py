""" **************************************************************************
    start of file """

import os
import json
import sys

import numpy as np
import pandas as pd
import math
from scipy.signal import find_peaks, peak_widths
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.integrate import odeint, solve_ivp

import logbook_api as logbook_api
import webapi_access as api
import dat_lists
import plot_funcs as pf
import mClass

import warnings
warnings.filterwarnings("ignore", "FutureWarning")

Z = np.zeros
one = np.ones
line = np.linspace

""" eo header
************************************************************************** """


def load_methane_session(
        file='S49_dischargeplan.xlsx',
        sheet='S49 Discharge completion'):
    return (
        pd.read_excel(
            io='../results/PEAKS/data/' + file,
            sheet_name=sheet))


def load_database(
        file='../results/PEAKS/data/database.json',
        debug=False):
    if os.path.isfile(file):
        with open(file, 'r') as f:
            based = json.load(f)
        f.close()
        return (based)

    else:
        return ({
            '20180920': {
                'ids': [],
                'info': [{}],
                'comments': [[None]],
                'peaks': [{}],
                'components': [{}]
            }
        })


def store_database(
        data={},
        file='../results/PEAKS/data/database.json',
        debug=False):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=False)
    f.close()
    return


def methane_session_stat(
        date='20180920',
        id=4,
        peak_width=0.075,  # 75ms
        debug=False):
    # base is 20180920.004
    session = load_methane_session()
    for i, id in enumerate(session['Discharge #']):
        xp_stat(date=date, id=id,
                peak_width=peak_width, debug=debug)
    return


def xp_stat(
        date='20180920',
        id=4,
        solo=False,
        peak_width=0.075,  # 75ms
        debug=False):

    # old
    # bolo_list = (7, 8, 9, 10, 12, 13, 29, 30, 41, 42, 44, 45)
    # data_req, data_labels = dat_lists.link_list()
    # names = [data_labels[k] for k in bolo_list]

    # links and labels
    names = [
        'PradHBCm', 'PradVBC',
        'T_e ECE core', 'T_e ECE out', 'n_e lint',
        'W_dia', 'ECRH', 'Main valve BG011', 'Main valve BG031',
        'QSQ Feedback AEH51', 'QSQ Feedback AEH30']
    print('>> Downloading from:', names)

    if not solo:
        f = '../results/PEAKS/data/database_' + str(int(
            peak_width * 1.e3)) + 'ms.json'
        based = load_database(file=f)

    else:
        based = {
            date: {
                'ids': [id],
                'info': [{}],
                'comments': [[None]],
                'peaks': [{}],
                'components': [{}]}
        }

    if debug:
        print('\\\ ', date + '.' + str(id).zfill(3))

    if date in based.keys():
        if id in based[date]['ids']:
            index = based[date]['ids'].index(id)
            if debug:
                print(date, id, index)

        else:
            based[date]['ids'].append(id)
            based[date]['info'].append({})
            based[date]['comments'].append([None])
            based[date]['peaks'].append({})
            based[date]['components'].append({})
            index = based[date]['ids'].index(id)

    else:
        index = 0
        based[date] = {
            'ids': [id],
            'info': [{}],
            'comments': [None],
            'peaks': [{}],
            'components': [{}]}

    # get entire fuelling data
    info = logbook_api.logbook_json_load(
        lu_list=['Fuelling'], group='category', date=date,
        shot=id, debug=False, indent_level='', printing=False)
    based[date]['info'][index] = filter_data(data=info)

    based[date]['comments'][index] = logbook_api.filter_comments(
        date=date, shot=id, users=['viwi', 'cak'], debug=False)

    based[date]['components'][index]['QSQ'] = logbook_api.db_request_component(
        date=date, id=id, component='QSQ')[1]  # is none if no log

    # XPID info and times ins ns
    program_info, req = api.xpid_info(
        program=date + '.' + str(id).zfill(3))
    start = str(program_info['programs'][0]['from'])
    stop = str(program_info['programs'][0]['upto'])

    download = {}
    for i, tag in enumerate(names):
        download[tag] = api.download_single(
            api.download_link(name=tag), program_info=program_info,
            start_POSIX=start, stop_POSIX=stop)

    # get ne and Te from thomson scattering individually
    n, t = api.get_thomson_traces(
        XPID=date + '.' + str(id).zfill(3))
    for i, d in enumerate(n):  # n_e QTB volX
        download['n_e QTB vol' + d[0]] = {
            'dimensions': d[1][0], 'values': d[1][1]}
    for i, d in enumerate(t):  # T_e QTB volX
        download['T_e QTB vol' + d[0]] = {
            'dimensions': d[1][0], 'values': d[1][1]}

    based[date]['peaks'][index] = peak_wrapper(
        date=date, id=id, width=peak_width,
        program_info=program_info, data=download,
        info=filter, smoothN=25, debug=False)

    if not solo:
        store_database(file=f, data=based)
    return (based, download)


def peak_wrapper(
        date='20180920',
        id=4,
        width=0.075,  # 75ms
        program_info={'none': None},
        data={'none': None},
        info={'none': None},
        smoothN=40,
        debug=False):

    def adjust_traces(
            D={'none': None}):
        # on/off time of ECRH
        ecrh_on = program_info["programs"][0]["trigger"]["1"][0]
        ecrh_off = program_info["programs"][0]["trigger"]["4"][0] + 5.e8

        for i, key in enumerate(D.keys()):
            time = np.array(D[key]['dimensions'])
            if (np.shape(time)[0] == 0):
                if debug:
                    print('\\\ empty timetrace', key)
                continue

            # inex where ECRH on/off
            id = (np.argmin(np.abs(time - ecrh_on)),
                  np.argmin(np.abs(time - ecrh_off)))

            # adjust accordingly so in s
            D[key]['dimensions'] = (time[id[0]:id[1]] - ecrh_on) / 1e9
            D[key]['values'] = np.convolve(
                D[key]['values'][id[0]:id[1]],
                np.ones((smoothN,)) / smoothN, mode='same')
        return (D)

    adjD = adjust_traces(D=data)  # adjust time and data

    peaks = {}
    for j, key in enumerate(data.keys()):
        if (len(adjD[key]['dimensions']) == 0 or
                len(adjD[key]['values']) == 0):
            continue
        P = peaks_fwhm(  # find peaks
            x=adjD[key]['dimensions'],
            y=adjD[key]['values'],
            width=width, debug=debug)

        if (np.shape(P)[1] > 0):
            peaks[key] = {
                'index': list(map(int, P[0])),
                'time': P[1].tolist(),
                'start_index': list(map(int, P[2])),
                'stop_index': list(map(int, P[3])),
                'start_time': P[4].tolist(),
                'stop_time': P[5].tolist(),
                'width_index': P[6].tolist(),
                'width_time': P[7].tolist(),
                'start_val': P[8].tolist(),
                'peak_val': P[9].tolist(),
                'stop_val': P[10].tolist()}

        else:
            peaks[key] = {
                'index': [], 'time': [], 'start_index': [],
                'stop_index': [], 'start_time': [], 'stop_time': [],
                'width_index': [], 'width_time': []}

    if True and width >= 0.1:
        pf.peak_plot_wrapper(
            date=date, id=id, time=width,
            data=adjD, peaks=peaks)
    return (peaks)


def peaks_fwhm(
        x=np.zeros((10000)),
        y=np.zeros((10000)),
        width=.075,  # min width in s
        debug=False):

    # dt, in s
    dt = x[1] - x[0]
    index_width = round(width / dt)

    peaks, properties = find_peaks(
        y, width=index_width)
    # rel_height=0.5, prominence=(None, 0.6), distance=1,
    # wlen=1., threshold=None, height=0.5, plateau_size=None
    widths = peak_widths(y, peaks)

    f = 100
    index = np.linspace(
        .0, np.shape(y)[0] - 1, np.shape(y)[0] * f)

    # new_x = np.linspace(x[0], x[-1], np.shape(x)[0] * f)
    new_x = np.interp(
        index, np.linspace(.0, np.shape(y)[0] - 1, np.shape(y)[0]), x)

    if (np.shape(peaks) == 0):
        return (None)
    # index, time, start index, stop index, point width, time width
    object = np.zeros((11, np.shape(peaks)[0]))
    for i, peak in enumerate(peaks):
        h = np.argmin(np.abs(index - peak))
        j = np.argmin(np.abs(index - widths[2][i]))  # start
        k = np.argmin(np.abs(index - widths[3][i]))  # stop

        if (widths[1][i] == .0 or new_x[k] - new_x[j] < width):
            if debug:
                print('empty')
            continue

        # peak pos
        object[0, i] = peak
        # peak time
        object[1, i] = new_x[h]

        # start and stop in index
        object[2:4, i] = round(widths[2][i]), round(widths[3][i])
        # start and stop in time
        object[4:6, i] = new_x[j], new_x[k]

        # point dist
        object[6, i] = widths[3][i] - widths[2][i]
        # time dist
        object[7, i] = (new_x[k] - new_x[j])  # * 100.

        # start stop and peak value
        object[8, i] = y[int(round(widths[2][i]))]
        object[9, i] = y[int(round(peak))]
        object[10, i] = y[int(round(widths[3][i]))]

    object = object[~np.all(object == 0, axis=1)]
    if debug:
        print('profile:', peaks, widths)
        print(np.shape(peaks), np.shape(widths), np.shape(object))
    return (object)


def filter_data(
        data={},
        debug=False):

    out = {}
    for j, key in enumerate(data.keys()):
        out[key] = split_data(
            data=data[key],
            names=['Gas1', 'Gas2', 'HeBeam'])

    return (out)


def split_data(
        data=[],
        names=['Gas1', 'Gas2', 'HeBeam'],
        debug=False):

    def info_from_split(
            tag='test',
            split=[]):
        for i, S in enumerate(split):
            if tag in S:
                return (S, tag)
        return (None, None)

    key_out = {}
    for j, entry in enumerate(data):
        if entry['name'] in names:
            if entry['name'] not in key_out.keys():
                key_out[entry['name']] = {
                    'type': [], 'duration': [], 'mode': []}

            for i, I in enumerate(
                    ['H2', 'He', 'CH4', 'ms', 'FeedForward',
                     'FlowCtrl', 'DensityCtrl']):
                info, tag = info_from_split(
                    tag=I, split=entry['value'].replace(', ', ',').split())

                if info is not None:
                    if tag in ['H2', 'He', 'CH4']:
                        key_out[entry['name']]['type'].append(info)
                    if tag == 'ms':
                        key_out[entry['name']]['duration'].append(info)
                    if tag in ['FeedForward', 'FlowCtrl', 'DensityCtrl']:
                        key_out[entry['name']]['mode'].append(info)

    return (key_out)


def match_peaks(
        db_file='../results/PEAKS/data/database_100ms.json',
        data=None,
        solo=True,
        debug=False):
    ms = int(db_file.replace(
        '../results/PEAKS/data/database_', '').replace('ms.json', ''))

    def load(file='../results/PEAKS/data/peaks_100ms.json'):
        if os.path.isfile(file):
            with open(file, 'r') as f:
                based = json.load(f)
            f.close()
            return (based)
        else:
            return ({'20180920.004': {}})

    def store(data={}, file='../results/PEAKS/data/peaks.json'):
        with open(file, 'w') as f:
            json.dump(data, f, indent=4, sort_keys=False)
        f.close()
        return (None)

    # load base file
    if data is None:
        data, solo = load_database(file=db_file), False
    D = mClass.dict_transf(
        dictionary=data, to_list=False, verbose=False)

    f = '../results/PEAKS/data/peaks_' + str(ms) + 'ms.json'
    if not solo:
        res = load(file=f)
    else:
        res = {}

    for date in D.keys():
        # go through each discharge
        for i, id in enumerate(D[date]['ids']):
            print('\\\ i:', i, date + '.' + str(id).zfill(3))
            res[date + '.' + str(id).zfill(3)] = \
                match_metric(data=D[date], id=i)

        if not solo:
            store(res, file=f)
    return (res)


def match_metric(
        data,
        id=0,
        debug=False):

    def compare_dicts(a, b):
        if isinstance(a, dict) and isinstance(
                b, dict) and a.keys() == b.keys():
            return (np.all([np.array_equal(a[key], b[key])
                            for key in a.keys()]))
        return (False)

    def get_peaks(d, i, l, verbose=False):
        try:
            if not d['peaks'][i][l]['index'] == []:
                return (d['peaks'][i][l])
        except Exception:
            if verbose:
                print('\\\ ' + l + ' no peaks')
        return (None)

    def get_comment(d, i, verbose=False):
        try:
            if not d['comments'][i] == []:
                return (d['comments'][i])
        except Exception:
            if verbose:
                print('\\\ no comments')
        return (None)

    def get_info(d, i, l0, l1, l2, verbose=False):
        try:
            if not d['info'][i][l0][l1][l2] == []:
                return (d['info'][i][l0][l1][l2])
        except Exception:
            if verbose:
                print('\\\ ' + l1 + ' no ' + l2)
        return (None)

    def match_peak(  # two props
            u=0, a={}, b={}, params=[], thL=0.1, thR=0.1):

        par_res = {}
        if (a is not None) and (b is not None):
            pl = np.where(
                (a['time'][u] - thL < b['time']) &
                (b['time'] < a['time'][u] + thR))[0]

            if pl.size > 0:
                # times of peaks found
                t = b['time'][pl]
                # peak closest to origin
                pc = (t - a['time'][u]).argmin()
                pl = pl[pc]

                if not isinstance(pl, list):
                    pl = [pl]

                for name, par in zip(
                        ['PradHBCm', 'PradVBC', 'ECRH', 'n_e',
                         'T_e in', 'T_e out'], params):
                    par_res[name] = {'time': [], 'value': []}

                    if par is None:
                        par_res[name] = {
                            'time': [None], 'value': [None]}
                        continue

                    for t in b['time'][pl]:  # times found
                        pPL = np.where(
                            (t - 0.15 < par['time']) &
                            (par['time'] < t + 0.3))[0]

                        if pPL.size > 0:
                            pPC = (par['time'][pPL] - t).argmin()
                            pPL = pPL[pPC]

                            par_res[name]['time'].append(par['time'][pPL])
                            par_res[name]['value'].append(
                                par['start_val'][pPL])

                        else:
                            par_res[name]['time'].append(None)
                            par_res[name]['value'].append(None)

                return (
                    b['time'][pl], b['width_time'][pl],  # time, width
                    b['time'][pl] - a['time'][u],  # distance
                    b['start_val'][pl], b['peak_val'][pl],  # base and peak
                    par_res)  # parameter fit results
        return (None, None, None, None, None, None)

    def assort_peak(  # match the peaks and append to dat
            res, params, src_key, source, key, obj, index, left, right, types):
        # tindr, 'QSQ Feedback AEH51', aeh51, key, obj, j, L, R, [hebeam])

        # find matching peaks in obj
        time, width, distance, base, peak, par_res = match_peak(
            u=index, a=source, b=obj, params=params, thL=left, thR=right)
        if isinstance(time, float):  # if only float, then 'stretch'
            time, width, distance, base, peak = \
                [time], [width], [distance], [base], [peak]

        gastype = ''
        for T in types:
            if T is not None:
                for gas in T:
                    gastype += gas + ' '
            else:
                gastype += 'None '

        if (time is not None) and (time != [None]):  # only real results
            for i, t in enumerate(time):

                for k, val in zip(
                        ['time', 'width', 'distance', 'base', 'peak'],
                        [time[i], width[i], distance[i], base[i], peak[i]]):
                    # matched peaks in obj
                    res[key][k].append(val)

                for k, val in zip(
                        ['type', 'gas', 'time', 'width', 'base', 'peak'],
                        [src_key, gastype, source['time'][index],
                         source['width_time'][index],
                         source['start_val'][index],
                         source['peak_val'][index]]):
                    # fitting source peak for match
                    res[key]['source'][k].append(val)

                for pk in par_res.keys():
                    res[key]['params'][pk]['time'].append(
                        par_res[pk]['time'][i])

                    res[key]['params'][pk]['value'].append(
                        par_res[pk]['value'][i])

        return (res)

    # feedback
    aeh51 = get_peaks(data, id, 'QSQ Feedback AEH51')
    aeh30 = get_peaks(data, id, 'QSQ Feedback AEH30')

    # main gas valve
    bg011 = get_peaks(data, id, 'Main valve BG011')
    bg031 = get_peaks(data, id, 'Main valve BG031')

    # QSB data
    hbcm = get_peaks(data, id, 'PradHBCm')
    vbc = get_peaks(data, id, 'PradVBC')

    # plasma parameters
    ecrh = get_peaks(data, id, 'ECRH')

    # ne
    if get_peaks(data, id, 'n_e QTB vol2') is None:
        ne = get_peaks(data, id, 'n_e lint')
    else:
        ne = get_peaks(data, id, 'n_e QTB vol2')

    # Te
    if get_peaks(data, id, 'T_e QTB vol2') is None:
        Te_in = get_peaks(data, id, 'T_e ECE core')
    else:
        Te_in = get_peaks(data, id, 'T_e QTB vol2')
    Te_out = get_peaks(data, id, 'T_e ECE out')

    # gas input info and comments
    # comments = get_comment(data, id)
    gas1 = get_info(data, id, 'Fuelling', 'Gas1', 'type')
    gas2 = get_info(data, id, 'Fuelling', 'Gas2', 'type')
    hebeam = get_info(data, id, 'Fuelling', 'HeBeam', 'type')

    # [.2, .2, .1, .1, .1, .1, .1, .1], \
    # [.5, .5, .1, .1, .1, .1, .1, .1]
    left, right = \
        [.0, .0, .1, .1, .1, .1, .1, .1], \
        [.5, .5, .1, .1, .1, .1, .1, .1]

    params = [hbcm, vbc, ecrh, ne, Te_in, Te_out]
    tindr = {}
    for k, key in enumerate([
            'PradHBCm', 'PradVBC', 'ECRH', 'n_e QTB vol2',
            'n_e lint', 'T_e QTB vol2', 'T_e ECE core', 'T_e ECE out']):
        tindr[key] = {
            'time': [], 'width': [], 'distance': [], 'base': [], 'peak': [],
            'source': {
                'type': [], 'gas': [], 'time': [],
                'width': [], 'base': [], 'peak': []
            }, 'params': {}}

        for key2 in ['PradHBCm', 'PradVBC', 'ECRH', 'n_e',
                     'T_e in', 'T_e out']:
            tindr[key]['params'][key2] = {'time': [], 'value': []}

    if aeh51 is not None:
        for j, S in enumerate(aeh51['start_index']):
            for key, obj, L, R in zip(
                    ['PradHBCm', 'PradVBC', 'ECRH', 'n_e QTB vol2',
                     'n_e lint', 'T_e QTB vol2', 'T_e ECE core', 'T_e ECE out'],
                    [hbcm, vbc, ecrh, ne, ne, Te_in, Te_in, Te_out],
                    left, right):

                tindr = assort_peak(
                    tindr, params, 'QSQ Feedback AEH51',
                    aeh51, key, obj, j, L, R, [hebeam])

    if aeh30 is not None:
        for k, S in enumerate(aeh30['start_index']):
            for key, obj, L, R in zip(
                    ['PradHBCm', 'PradVBC', 'ECRH', 'n_e QTB vol2',
                     'n_e lint', 'T_e QTB vol2', 'T_e ECE core', 'T_e ECE out'],
                    [hbcm, vbc, ecrh, ne, ne, Te_in, Te_in, Te_out],
                    left, right):

                tindr = assort_peak(
                    tindr, params, 'QSQ Feedback AEH30',
                    aeh30, key, obj, k, L, R, [hebeam])

    if bg011 is not None:
        for k, S in enumerate(bg011['start_index']):
            for key, obj, L, R in zip(
                    ['PradHBCm', 'PradVBC', 'ECRH', 'n_e QTB vol2',
                     'n_e lint', 'T_e QTB vol2', 'T_e ECE core', 'T_e ECE out'],
                    [hbcm, vbc, ecrh, ne, ne, Te_in, Te_in, Te_out],
                    left, right):

                tindr = assort_peak(
                    tindr, params, 'Main valve BG011',
                    bg011, key, obj, k, L, R, [gas1, gas2])

    if bg031 is not None:
        for k, S in enumerate(bg031['start_index']):
            for key, obj, L, R in zip(
                    ['PradHBCm', 'PradVBC', 'ECRH', 'n_e QTB vol2',
                     'n_e lint', 'T_e QTB vol2', 'T_e ECE core', 'T_e ECE out'],
                    [hbcm, vbc, ecrh, ne, ne, Te_in, Te_in, Te_out],
                    left, right):

                tindr = assort_peak(
                    tindr, params, 'Main valve BG031',
                    bg031, key, obj, k, L, R, [gas1, gas2])

    return (tindr)


def peak_database(
        dbfile='../results/PEAKS/data/peaks_100ms.json',
        paramfile='../results/PEAKS/data/peak_params_100ms.json',
        res=None,
        solo=True,
        src_filter_key=None,
        src_filter_label=None,
        debug=False):

    def load(file):
        if os.path.isfile(file):
            with open(file, 'r') as f:
                base = json.load(f)
            f.close()
            return (base)
        else:
            return ({'none': None})

    if res is None:
        res, solo = load(dbfile), False
    res = mClass.dict_transf(
        res, to_list=False)

    params = load(paramfile)
    params = mClass.dict_transf(
        params, to_list=False)
    D = {}

    for x, xpid in enumerate(res.keys()):  # XPIDs
        if int(xpid[:4]) < 2018:
            continue

        data = res[xpid]  # experiments

        for g, keyA in enumerate(data.keys()):  # plasma params
            try:
                if D[keyA] == {}:
                    if debug:
                        print('\\\ [' + keyA + '] exists')
            except Exception:
                D[keyA] = {}

            if src_filter_key is not None:
                filter_list = data[keyA]['source'][src_filter_key]

            for j, keyB in enumerate(data[keyA].keys()):  # time width dist etc

                if isinstance(data[keyA][keyB], np.ndarray):
                    try:
                        if D[keyA][keyB] == []:
                            if debug:
                                print('\\\ [' + keyA +
                                      '][' + keyB + '] exists')
                    except Exception:
                        D[keyA][keyB] = []

                    for l, val in enumerate(data[keyA][keyB]):
                        if src_filter_key is not None:
                            if (src_filter_label not in filter_list[l]):
                                continue
                        else:
                            pass
                        D[keyA][keyB].append(val)

                elif isinstance(data[keyA][keyB], dict):  # if source
                    try:
                        if D[keyA][keyB] == {}:
                            if debug:
                                print('\\\ [' + keyA +
                                      '][' + keyB + '] exists')
                    except Exception:
                        if (keyB == 'source'):
                            D[keyA][keyB] = {}

                        elif (keyB == 'params'):
                            D[keyA][keyB] = {}

                    if keyB == 'params':
                        for keyP in params.keys():  # for all params
                            try:
                                if D[keyA][keyB][keyP] == {}:
                                    if debug:
                                        print('\\\ [' + keyA + '][' +
                                              keyB + '][' + keyP + '] exists')
                            except Exception:
                                D[keyA][keyB][keyP] = {
                                    'time': [], 'value': []}

                            P = params[keyP]  # different params
                            pM = np.where(P['xpid'] == xpid)[0]  # where info

                            for j, val in enumerate(
                                    data[keyA]['time']):  # for peaks

                                if pM.size > 0:  # found XPIDs
                                    # matching the time
                                    pI = pM[np.abs(
                                        P['time'][pM] - val).argmin()]

                                    if src_filter_key is not None:  # filter?
                                        if (src_filter_label not in
                                                filter_list[j]):
                                            continue
                                    else:
                                        pass

                                    D[keyA][keyB][keyP]['value'].append(
                                        P['start_val'][pI])
                                    D[keyA][keyB][keyP]['time'].append(
                                        P['time'][pI])

                                else:
                                    if src_filter_key is not None:  # filter?
                                        if (src_filter_label not in
                                                filter_list[j]):
                                            continue
                                        else:
                                            pass
                                    D[keyA][keyB][keyP]['time'].append(0.0)
                                    D[keyA][keyB][keyP]['value'].append(0.0)

                    elif keyB == 'source':
                        for k, keyC in enumerate(data[keyA][keyB].keys()):
                            try:  # time width type etc
                                if D[keyA][keyB][keyC] == []:
                                    if debug:
                                        print('\\\ [' + keyA +
                                              '][' + keyB + '][' +
                                              keyC + '] exists')
                            except Exception:
                                D[keyA][keyB][keyC] = []

                            for h, val in enumerate(data[keyA][keyB][keyC]):

                                if src_filter_key is not None:
                                    if (src_filter_label not in
                                            filter_list[h]):
                                        continue
                                else:
                                    pass

                                D[keyA][keyB][keyC].append(val)

    return (D)


def diag_peaks(
        file='../results/PEAKS/data/peaks_100ms.json',
        paramfile='../results/PEAKS/data/peak_params_100ms.json',
        peaks=None,
        filter_key=None,
        filter_value=None,
        debug=False):
    if peaks is None:  # sort
        data = peak_database(
            dbfile=file,
            paramfile=paramfile,
            src_filter_key=filter_key,
            src_filter_label=filter_value)

    else:
        data = peak_database(
            paramfile=paramfile,
            res=peaks,
            solo=True,
            src_filter_key=filter_key,
            src_filter_label=filter_value)

    ms = int(file.replace(
        '../results/PEAKS/data/peaks_', '').replace(
        'ms.json', ''))
    pf.peak_database_plots(
        data=data, width=ms,
        filter_value=filter_value,
        filter_key=filter_key)
    return (data)


def scatter_peaks(
        file='../results/PEAKS/data/peaks_100ms.json',
        paramfile='../results/PEAKS/data/peak_params_100ms.json',
        peaks=None,
        filter_key=None,
        filter_value=None,
        debug=False):
    if peaks is None:  # sort
        data = peak_database(
            dbfile=file, paramfile=paramfile,
            src_filter_key=filter_key,
            src_filter_label=filter_value)

    else:
        data = peak_database(
            paramfile=paramfile,
            res=peaks, solo=True,
            src_filter_key=filter_key,
            src_filter_label=filter_value)

    data = mClass.dict_transf(
        data, to_list=False)
    ms = int(file.replace(
        '../results/PEAKS/data/peaks_',
        '').replace('ms.json', ''))

    pf.peak_scatter_colorbar(
        data=data, filter_key=filter_key,
        width=ms, filter_value=filter_value)
    return (data)


def plasma_param_at_peaks(
        dbfile='../results/PEAKS/data/database_100ms.json',
        date='20180920',
        id=4,
        peak_width=0.075,
        res=None,
        solo=False,
        debug=False):

    def load(file='peak_params_100ms.json'):
        file = '../results/PEAKS/data/' + file
        if os.path.isfile(file):
            with open(file, 'r') as f:
                base = json.load(f)
            f.close()
            return (base)
        else:
            return ({'ECRH': {
                'xpid': [], 'time': [], 'start_time': [], 'stop_time': [],
                'start_val': [], 'peak_val': [], 'params': {}}})

    def store(data={}, file='peak_params_100ms.json'):
        file = '../results/PEAKS/data/' + file
        with open(file, 'w') as f:
            json.dump(data, f, indent=4, sort_keys=False)
        f.close()
        return (None)

    def adjust_time(t, t0):
        return (np.array([(x - t0) / 1e9 for x in t]))

    ms = int(peak_width * 1.e3)
    file = 'peak_params_' + str(ms) + 'ms.json'
    dbfile = '../results/PEAKS/data/database_' + str(ms) + 'ms.json'

    based = load_database(file=dbfile)
    based = mClass.dict_transf(based, to_list=False)

    if (res is None) and (solo is False):
        res, solo = load(file=file), False
    if solo and (res in None):
        res = {}  # start fresh

    if solo:
        dates = [date]
    else:
        dates = list(based.keys())

    for d, date in enumerate(dates.copy()):
        if solo:
            xpids = [id]
        else:
            xpids = based[date]['ids']

        for i, id in enumerate(xpids):
            prog = date + '.' + str(id).zfill(3)

            if 'ECRH' not in res.keys():  # fresh start
                res[key] = {
                    'xpid': [],
                    'time': [],
                    'start_time': [],
                    'stop_time': [],
                    'start_val': [],
                    'peak_val': [],
                    'params': {}
                }

            elif prog in res['ECRH']['xpid']:
                print('\\\ ' + prog + ': already in database')
                continue

            index = np.where(based[date]['ids'] == id)[0][0]
            data = based[date]['peaks'][index]

            # links and labels
            bolo_list = (7, 8, 9, 10, 11, 12, 13, 29, 30, 41, 42, 44, 45)
            data_req, data_labels = dat_lists.link_list()
            names = [data_labels[k] for k in bolo_list]

            if debug:
                print('\\\ Downloading from:', names)

            for l, key in enumerate([
                    'ECRH', 'T_e ECE core', 'T_e ECE out', 'n_e lint',
                    'n_e QTB vol2', 'T_e QTB vol2', 'Main valve BG011',
                    'Main valve BG031', 'QSQ Feedback AEH51',
                    'QSQ Feedback AEH30', 'PradHBCm', 'PradVBC']):

                if key not in res.keys():  # fresh start
                    res[key] = {
                        'xpid': [],
                        'time': [],
                        'start_time': [],
                        'stop_time': [],
                        'start_val': [],
                        'peak_val': [],
                        'params': {}
                    }

                for k, key2 in enumerate([
                        "time", "start_time", "stop_time",
                        "start_val", "peak_val"]):
                    try:
                        for i, v in enumerate(data[key][key2]):
                            T, t = np.array(data[key]['time']), \
                                data[key]['time'][i]

                            if ((t is None) or (np.shape(np.where(
                                    T == t)[0])[0] > 1) or (t == 0.0)):
                                continue

                            if key2 == 'time':
                                res[key]['xpid'].append(prog)
                            res[key][key2].append(v)

                    except Exception:
                        pass

                        # key (quantity) or key2 (property) not found
                        # for j, t in enumerate(data[key]['time']):
                        #     res[key][key2].append(None)

            # XPID info and times ins ns
            program_info, req = api.xpid_info(program=prog)
            start = str(program_info['programs'][0]['from'])
            stop = str(program_info['programs'][0]['upto'])
            # triggers
            t0 = program_info['programs'][0]['trigger']['1'][0]  # in ns

            print('\\\ downloading ' + prog + '...')
            download = {}
            for i, tag in enumerate(names):
                download[tag] = api.download_single(
                    api.download_link(name=tag), program_info=program_info,
                    start_POSIX=start, stop_POSIX=stop)
            if (np.shape(download[tag]['dimensions'])[0] == 0):
                continue

            adjT = adjust_time(
                download[tag]['dimensions'], t0)

            for k, key in enumerate(res.copy()):
                res[key]['params'][tag] = {
                    'start_val': [], 'peak_val': []}

                for j, t in enumerate(res[key]['time']):
                    if t is None or (np.shape(np.where(res[key][
                            'time'] == t)[0])[0] > 1) or (t == 0.0):
                        continue

                    ind1 = np.abs(adjT - t).argmin()
                    ind2 = np.abs(adjT - res[key]['start_time'][j]).argmin()

                    res[key]['params'][tag]['peak_val'].append(
                        download[tag]['values'][ind1])
                    res[key]['params'][tag]['start_val'].append(
                        download[tag]['values'][ind2])

        if not solo:
            store(data=res, file=file)
    return (res)


def diag_plasma_peak_params(
        dbfile='../results/PEAKS/data/peak_params_100ms.json',
        data=None,
        solo=True,
        frad_filt=None):
    label = dbfile.replace(
        '../results/PEAKS/data/', '').replace('peak_params_',
        '').replace('.json', '')

    def load():
        if os.path.isfile(dbfile):
            with open(dbfile, 'r') as f:
                base = json.load(f)
            f.close()
            return (mClass.dict_transf(base, to_list=False))
        else:
            return (None)

    if data is None:
        print('... loading ' + label)
        data, solo = load(), False

    if frad_filt is not None:
        for key in data.copy():
            filter = np.array([
                (v / 1.e6) / (data[key]['params']['ECRH'][
                    'start_val'][i] / 1.e3) for i, v in
                enumerate(data[key]['params']['PradHBCm']['start_val'])])
            nn = np.where(filter > frad_filt)[0]

            # filter 2018 OP1.2b discharges
            mm = np.where(np.array([
                int(v[:4]) for v in data[key]['xpid'][nn]]) > 2018)[0]
            nn = nn[mm]  # recast to OG data

            for keyB in data[key].copy():
                if isinstance(data[key][keyB], list):
                    data[key][keyB] = np.array(data[key][keyB])[nn]
                elif isinstance(data[key][keyB], np.ndarray):
                    data[key][keyB] = data[key][keyB][nn]

                if keyB == 'params':
                    for keyP in data[key][keyB].copy():

                        for keyC in ['start_val', 'peak_val']:
                            if isinstance(data[key][keyB][
                                    keyP][keyC], list):
                                data[key][keyB][keyP][keyC] = np.array(data[
                                    key][keyB][keyP][keyC])[nn]
                            elif isinstance(data[key][keyB][
                                    keyP][keyC], np.ndarray):
                                data[key][keyB][keyP][keyC] = data[key][
                                    keyB][keyP][keyC][nn]

    pf.plot_plasma_peak_params(
        data, label + '_frad' + str(frad_filt), False)
    return (data)


def regress_peak_params(
        dbfile='../results/PEAKS/data/peak_params_100ms.json',
        data=None):

    def load():
        if os.path.isfile(dbfile):
            with open(dbfile, 'r') as f:
                base = json.load(f)
            f.close()
            return (base)
        else:
            return (None)

    def regress_lin(x, y, z):
        def lin_func(J, a, b, c):
            k, h = J
            return (a * k**b + h**c)
        [a, b, c], res = curve_fit(
            lin_func, (x, y), z,
            p0=(1., 1., 1.), absolute_sigma=None,
            check_finite=True, method='lm')
        return (a, b, c)

    if data is None:
        print('loading file ...')
        data = load()['PradHBCm']

        # filter 2018 OP1.2b discharges
        nn = np.where(np.array([
            int(v[:4]) for v in data['xpid'][nn]]) > 2018)[0]

        for key in data.copy():
            if isinstance(data[key], list):
                data[key] = np.array(data[key])[nn]
            elif isinstance(data[key], np.ndarray):
                data[key] = data[key][nn]

            if key == 'params':
                for keyP in data[key].copy():
                    for keyC in ['start_val', 'peak_val']:
                        if isinstance(data[key][
                                keyP][keyC], list):
                            data[key][keyP][keyC] = np.array(data[
                                key][keyP][keyC])[nn]
                        elif isinstance(data[key][
                                keyP][keyC], np.ndarray):
                            data[key][keyP][keyC] = data[key][
                                keyP][keyC][nn]

    # make arrays for calc
    data = mClass.dict_transf(data, to_list=False)

    params = data['params']
    ecrh, ne = params['ECRH'], params['n_e lint']
    te_out, te_core = params['T_e ECE out'], params['T_e ECE core']

    bg0 = {'start': (params['Main valve BG011']['start_val'] +
                     params['Main valve BG031']['start_val']),
           'peak': (params['Main valve BG011']['peak_val'] +
                    params['Main valve BG031']['peak_val'])}

    res = {
        'xlabel': [],
        'ylabel': [],
        'x': [], 'y': [], 'z': [],
        'a': [], 'b': [], 'c': []}

    def try_regress(
            x, y, z, ranges, xlabel, ylabel):
        try:
            nn = np.where(
                (x > ranges[0, 0]) & (x < ranges[0, 1]) &
                (y > ranges[1, 0]) & (y < ranges[1, 1]) &
                (z > ranges[2, 0]) & (z < ranges[2, 1]))
            X, Y, Z = x[nn], y[nn], z[nn]
            a, b, c = regress_lin(X, Y, Z)

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(xlabel, np.shape(X), np.shape(Y), np.shape(Z))
            print('\\\ failed regression:', exc_type, fname, exc_tb.tb_lineno)
            return

        res['x'].append(X)
        res['y'].append(Y)
        res['z'].append(Z)

        res['a'].append(a)
        res['b'].append(b)
        res['c'].append(c)

        res['xlabel'].append(xlabel)
        res['ylabel'].append(ylabel)
        return

    # prad = a * ne^b * ECRH^c
    try_regress(
        ecrh['start_val'] / 1.e3,  # x
        ne['start_val'] / 1.e19,  # y
        data['peak_val'] / 1.e6,  # z
        np.array([[0.5, 10.], [0.1, 1.e2], [0.25, 6.]]),
        'a$\cdot$n$_{e}^{b}$[10$^{19}$m$^{-3}$]$\cdot$P$_{ECRH}^{c}$[MW]',
        'P$_{rad}$ [MW]')

    # dprad = a * dne^b * dECRH^c
    try_regress(
        np.abs(ecrh['peak_val'] - ecrh['start_val']) / 1.e3,  # x
        np.abs(ne['peak_val'] - ne['start_val']) / 1.e19,  # y
        np.abs(data['peak_val'] - data['start_val']) / 1.e6,  # z
        np.array([[.0, 2.], [0.2, 2.5], [0.0, 3.]]),
        'a$\cdot\Delta$n$_{e}^{b}$[10$^{19}$m$^{-3}$]' +
        '$\cdot\Delta$P$_{ECRH}^{c}$[MW]', '$\Delta$P$_{rad}$ [MW]')

    # prad = a * ne^b * Te^c
    try_regress(
        te_core['start_val'],  # x
        ne['start_val'] / 1.e19,  # y
        data['peak_val'] / 1.e6,  # z
        np.array([[.5, 5.], [0.1, 1.e2], [1., 6.]]),
        'a$\cdot$n$_{e}^{b}$[10$^{19}$m$^{-3}$]$\cdot$T$_{e}^{c}$[keV]',
        'P$_{rad}$ [MW]')

    # dprad = a * dne^b * dTe^c
    try_regress(
        np.abs(te_core['peak_val'] - te_core['start_val']),  # x
        np.abs(ne['peak_val'] - ne['start_val']) / 1.e19,  # y
        np.abs(data['peak_val'] - data['start_val']) / 1.e6,  # z
        np.array([[.0, 2.], [0.2, 2.5], [.0, 3.]]),
        'a$\cdot\Delta$n$_{e}^{b}$[10$^{19}$m$^{-3}$]' +
        '$\cdot\Delta$T$_{e}^{c}$[keV]', '$\Delta$P$_{rad}$ [mw]')

    # dprad = a * dfBG^b * dTe^c
    try_regress(
        np.abs(te_core['peak_val'] - te_core['start_val']),  # x
        np.abs(bg0['peak'] - bg0['start']),  # y
        np.abs(data['peak_val'] - data['start_val']) / 1.e6,  # z
        np.array([[.0, 2.], [0., 50.], [.0, 4.]]),
        'a$\cdot\Delta\Gamma_{BG}^{b}$[mbar$\cdot$l/s]' +
        '$\cdot\Delta$T$_{e}^{c}$[keV]', '$\Delta$P$_{rad}$ [mw]')

    pf.plot_plasma_peak_regress(
        res, debug=False)
    return (res)


def seeding_wall(
        program='20180920.49',
        debug=False):
    # xp info
    program_info, req = api.xpid_info(program=program)
    start = str(program_info['programs'][0]['from'])
    stop = str(program_info['programs'][0]['upto'])

    # triggers
    t0 = program_info['programs'][0]['trigger']['1'][0]  # in ns
    t4 = program_info['programs'][0]['trigger']['4'][0]  # in ns

    # get prad traces for fitting of equations
    obj = api.download_single(
        api.download_link(name='PradHBCm'),
        program_info=program_info,
        start_POSIX=start, stop_POSIX=stop)
    s = np.array([(v - t0) / 1.e9 for v in obj['dimensions']])
    u = np.array([v / 1.e6 for v in obj['values']])

    # cut to relevant area
    T = s[np.where((s > 4.2) & (s < 7.2))[0]]
    V = u[np.where((s > 4.2) & (s < 7.2))[0]]

    # get seeding info
    _, info = logbook_api.db_request_component(
        date=program[:8], id=int(program[-2:]), component='QSQ')
    Gamma_s = float(info['AEH51 flow rate']['value'])

    # plasma capacity is n_p
    # wall capacity is n_w
    # plasma to wall transport is t_pw
    # wall to plasma transport is t_wp

    # plasma pump loss is t_p
    # wall pump loss is t_w
    # seeding is gamma_s

    # assuming after initial filling, wall approaches limit capacity and
    # then the system relaxes though seeding procedes

    """                        exchange between
        ---------------------     tau_pw        -------------------------
        |      plasma       |   -------->       |        wall           |
        |                   |                   |                       |
        |       N_p         |      tau_wp       |        N_w            |
        |                   |   <--------       |                       |
        ---------------------                   -------------------------

            ^          |
            |          |
            |          |
            |          v
         seeding     tau_p
         Gamma_s    pump loss

        N_P,  N_w in atom(s) (unitless, 1/1)
        tau_p, tau_w, tau_wp, tau_pw, Gamma_s in atoms(s) / second (1/s)

        d/dt N_p = Gamma_s + N_w * tau_wp - N_p * (tau_pw * f + tau_p)
        d/dt N_w = (N_p * tau_pw) * f - N_w * tau_wp

            lim       d/dt N_w == 0 = N_p * tau_pw * f - Nwlim * tau_wp
        N_w -> N_wlim

            ---> f = (N_wlim * tau_wp) / (N_p * tau_pw)

        d/dt N_w = (N_wlim - N_w) * tau_wp
        d/dt N_p = Gamma_s + N_w * tau_wp - N_p * (
            tau_pw * (N_wlim * tau_wp) / (N_p * tau_pw) + tau_p)

    """
    N_wlim = 1.e20
    tau_wp, tau_pw, tau_p = 1.e19, 1.e19, .5e19
    if debug:
        print(Gamma_s, tau_wp, tau_pw, N_wlim, tau_p)

    def y1(
            z, t, a, b, c, d, e):
        x, y = z
        return [
            a + y * b - y * (c * (d * b) / (x * c) + e),
            (d - y) * b]

    def y2(t, z, a, b, c, d, e):
        x, y = z
        return [
            a + y * b - y * (c * (d * b) / (x * c) + e),
            (d - y) * b]

    # solution = solve_ivp(
    # solution = odeint(
    #     func=y1, y0=np.array([1.e17, 0.0]), t=T - T[0],
    #     args=(Gamma_s, tau_wp, tau_pw, N_wlim, tau_p),
    #     printmessg=True, tfirst=False)

    # solution = solve_ivp(
    #     fun=y2, t_span=[0.0, T[-1] - T[0]], y0=[1.e17, 0.0], t_eval=tau,
    #     args=(Gamma_s, tau_wp, tau_pw, N_wlim, tau_p))

    Gamma_s, N_wlim = 10., 30.
    tau_wp, tau_pw, tau_p = 1., 4., .1
    T = np.linspace(0.0, 10., 10)

    solution = solve_ivp(
        fun=y2, t_span=[0.0, T[-1]], y0=[1., 1.0], t_eval=T,
        args=(Gamma_s, tau_wp, tau_pw, N_wlim, tau_p))  

    return (T, solution)
