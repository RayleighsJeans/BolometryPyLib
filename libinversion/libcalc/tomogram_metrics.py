""" **************************************************************************
    so header """

import os
import re
import sys
import numpy as np
import json
import matplotlib.pyplot as p
from scipy.optimize import curve_fit

from sklearn.linear_model import LinearRegression

import dat_lists
import webapi_access as api
import plot_funcs as pf
import mfr_plot as mfrp

Z = np.zeros
ones = np.ones

""" end of header
************************************************************************** """


def get_file_list():

    L = []

    list = os.walk('../results/INVERSION/MFR/')
    for i, entry in enumerate(list):
        if 'backup' in entry[0]:
            continue

        list2 = os.listdir(entry[0])
        for j, entry2 in enumerate(list2):
            if '.json' not in entry2:
                continue
            L.append(entry[0] + '/' + entry2)

    return (L)


def load_files(
        group='comp',
        verbose=False):
    list = get_file_list()
    print('\\\ files found:', len(list))

    res = {
        'peaks': {
            'height': {
                'tomogram': [],
                'phantom': [],
                'error': []
            }, 'widths': {
                'tomogram': [],
                'phantom': [],
            }, 'position': {
                'tomogram': [],
                'phantom': [],
                'error': []}
        }, 'P_rad': {
            'tomogram': {'HBCm': [], 'VBC': []},
            'phantom': {'HBCm': [], 'VBC': []}
        }, 'power': {
            'core': {
                'phantom': [],
                'tomogram': []
            }, 'total': {
                'phantom': [],
                'tomogram': []},
            'error': []
        }, 'difference': {
            'msd': [],
            'pearson': [],
            'xi2': []
        }, 'time': []
    }

    for i, file in enumerate(list):
        if group.replace('comp_', '').replace(
                'xp_', '') not in file:
            continue
        if verbose:
            print('\\\ f, i:', file, i)

        with open(file, 'r') as f:
            data = json.load(f)
        f.close()

        res['time'].append(data['values']['times'])

        tags = ['tomogram'] if 'xp' in group else ['tomogram', 'phantom']
        for key in tags:

            res['peaks']['widths'][key].append(
                data['values']['peaks']['half_widths'][key][0])

            res['peaks']['position'][key].append(
                data['values']['peaks']['radial_pos'][key])

            res['peaks']['height'][key].append(
                data['values']['peaks']['half_widths'][key][1])

            res['P_rad'][key]['HBCm'].append(
                data['values']['P_rad'][key]['HBCm'])

            res['P_rad'][key]['VBC'].append(
                data['values']['P_rad'][key]['VBC'])

            if ('xp' in group) and (key == 'tomogram'):
                res['P_rad']['phantom']['HBCm'].append(
                    data['values']['P_rad']['experiment']['HBCm'])
                res['P_rad']['phantom']['VBC'].append(
                    data['values']['P_rad']['experiment']['VBC'])

            res['power']['core'][key].append(
                data['values']['core_power'][key])

            res['power']['total'][key].append(
                data['values']['total_power'][key])

            if (key == 'tomogram'):
                if 'comp' in group:
                    res['difference']['msd'].append(np.mean(
                        data['values']['difference'][
                            'mean_square_deviation']))

                    res['difference']['pearson'].append(
                        data['values']['difference'][
                            'pearson_coefficient'])

                    res['difference']['xi2'].append(
                        data['values']['difference']['chi2'])

                try:
                    res['power']['error'].append(
                        data['values']['profiles']['2D_error'])
                except Exception:
                    if verbose:
                        print('\\\ failed errors')
                    res['power']['error'].append(0.0)

                try:
                    res['peaks']['position']['error'].append(
                        data['values']['profiles']['radial_error'])
                except Exception:
                    res['peaks']['position']['error'].append(0.0)

                try:
                    res['peaks']['height']['error'].append(
                        data['values']['profiles']['abs_error'])
                except Exception:
                    res['peaks']['height']['error'].append(0.0)

    return (res)


def plot_statistics(
        group='comp',
        data=None):

    def regress_lin(x, y):
        xx = np.array(x)
        yy = np.array(y)

        index = xx.argsort()
        xx_n = xx[index][:-30].reshape(-1, 1)
        yy_n = yy[index][:-30].reshape(-1, 1)

        regressor = LinearRegression()
        regressor.fit(xx_n, yy_n)
        yy_pred = regressor.predict(xx[index].reshape(-1, 1)).reshape(-1)
        line = np.array([xx[index].reshape(-1), yy_pred])
        return (regressor.coef_[0][0], regressor.intercept_[0], line)

    if data is None:
        print('\\\ loading tomograms')
        data = load_files(group=group)

    for cam in ['HBCm', 'VBC']:
        a, b, L1 = regress_lin(
            data['power']['core']['tomogram'],
            data['P_rad']['phantom'][cam])
        c, d, L2 = regress_lin(
            data['power']['total']['tomogram'],
            data['P_rad']['phantom'][cam])
        data['power']['core']['lin_' + cam] = [a, b, L1]
        data['power']['total']['lin_' + cam] = [c, d, L2]

    mfrp.plot_tomogram_statistics(data=data, group=group)
    return (data)


def get_longest_ind(
        data_list, t0, t1, N=200):
    t1 = (t1 - t0) / 1e9 + 2.5
    t0 = -0.75  # 0.0

    x1 = 0
    for i, v in enumerate(data_list):
        ind = np.where((v[0] > t0) & (v[0] < t1))[0]
        data_list[i] = np.array([v[0][ind], v[1][ind]])
        x1 = ind[-1] - ind[0] if (ind[-1] - ind[0]) >= x1 else x1

    int_list = np.linspace(.0, 1., x1)
    for i, v in enumerate(data_list):
        old_int = np.linspace(.0, 1., np.shape(v[0])[0])
        T = np.interp(int_list, old_int, v[0])
        data_list[i] = np.array([
            T, np.convolve(np.interp(T, v[0], v[1]),
                           np.ones((N,)) / N, mode='same')])
    return (data_list)


def power_balance_tomogram(
        program='20181010.032',
        plot=False,
        passVals=False):

    def sort_tomogram_power(tomos):
        T, PC, PT, PE = [], [], [], [[], []]
        for i, t in enumerate(tomos['time']):
            T.append(t)
            PT.append(tomos['power']['total']['tomogram'][i])
            PC.append(tomos['power']['core']['tomogram'][i])

            try:
                PE[0].append(tomos['power']['total']['tomogram'][i])
                PE[1].append(tomos['power']['error'][i])
            except Exception:
                pass  # print('\\\ failed errors')
        return (np.array([T, PT, PC]), np.array(PE))

    def tomograms_power_balance(power, a, d, e):
        balance = np.zeros((3, np.shape(power)[1]))
        for i, t in enumerate(power.transpose()):
            j1 = np.abs(a[0] - t[0]).argmin()
            j2 = np.abs(d[0] - t[0]).argmin()
            j3 = np.abs(e[0] - t[0]).argmin()

            balance[0, i] = t[0]
            balance[1, i] = a[1, j1] - t[1] - d[1, j2] - e[1, j3]
            balance[2, i] = a[1, j1] - t[2] - d[1, j2] - e[1, j3]
        return (balance)

    print('\\\ loading tomograms')
    tomograms = load_files(
        group='xp_' + program, verbose=False)
    tomo_powers, errors = sort_tomogram_power(tomograms)

    try:
        pinfo = api.xpid_info(program=program)[0]
        t0 = pinfo['programs'][0]['trigger']['1'][0]  # in ns
        t4 = pinfo['programs'][0]['trigger']['4'][0]  # in ns

        start = str(pinfo['programs'][0]['from'])
        stop = str(pinfo['programs'][0]['upto'])

    except Exception:
        print('\\\ no info, stop')
        return (None)

    try:  # ECRH
        ECRH = api.download_single(
            api.download_link(name='ECRH'),
            program_info=pinfo,
            start_POSIX=start, stop_POSIX=stop)
        ECRH = np.array([  # s and MW from T0
            [(t - t0) / 1e9 for t in ECRH['dimensions']],
            [v * 1e3 for v in ECRH['values']]])

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print('\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n\\\ ecrh failed')
        return (None)

    try:  # my calc
        print('\\\ using own prad')
        # links and labels
        bolo_list = (44, 45)
        data_req, data_labels = dat_lists.link_list()
        names = [data_labels[k] for k in bolo_list]

        HBC = api.download_single(
            api.download_link(name=names[0]), program_info=pinfo,
            start_POSIX=start, stop_POSIX=stop)
        VBC = api.download_single(
            api.download_link(name=names[1]), program_info=pinfo,
            start_POSIX=start, stop_POSIX=stop)

    except Exception:  # archive
        print('\\\ failed, using archive prad')
        HBC = api.download_single(
            api.download_link(name='Bolo_HBCmPrad'),
            program_info=pinfo,
            start_POSIX=start, stop_POSIX=stop)
        VBC = api.download_single(
            api.download_link(name='Bolo_VBCPrad'),
            program_info=pinfo,
            start_POSIX=start, stop_POSIX=stop)

    time = [(x - t0) / 1e9 for x in HBC["dimensions"]]
    HBC = np.array([time, [x for x in HBC['values']]])
    VBC = np.array([time, [x for x in VBC['values']]])

    try:
        wdia = api.download_single(
            api.download_link(name='W_dia'),
            program_info=pinfo,
            start_POSIX=start, stop_POSIX=stop)
        wdia = np.array([
            [(t - t0) / 1e9 for t in wdia['dimensions']],
            [x * 1.e3 for x in wdia['values']]])
        dwdia_dt = np.array([
            wdia[0], np.gradient(wdia[1]) / (wdia[0, 1] - wdia[0, 0])])

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print('\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n\\\ failed wdia')
        return (None)

    if (program == '20181010.032'):
        L = np.loadtxt('../files/20181010.032_total_divertor_load.txt')
        time = np.array([(t - t0) / 1e9 for t in L[:, 0]])
        div = np.array([time, np.array([x for x in L[:, 1]])])
        divertors = []  # empty

    else:
        [L0, Ls, LY] = \
            ['AEF', [
             '10', '20', '30', '40', '50',
             '11', '21', '31', '41', '51'],
             'heat load [MW]']

        missingU, missingD = 0, 0
        div, divU, divD = np.zeros((2, 0)), \
            np.zeros((2, 0)), np.zeros((2, 0))
        divertors = []

        for i, tag in enumerate(Ls):
            divertor = api.download_single(
                api.download_link(name='DivAEF' + tag),
                program_info=pinfo,
                start_POSIX=start, stop_POSIX=stop)
            divertor = np.array([
                [(t - t0) / 1.e9 for t in divertor['dimensions']],
                divertor['values']])

            if np.shape(divertor)[1] == 0:
                print('\\\ AEF' + tag, 'not found')
                if tag[-1] == '1':
                    missingU += 1
                elif tag[-1] == '0':
                    missingD += 1
                continue

            if np.shape(div)[1] == 0:
                div = np.zeros((2, len(divertor[1])))
                divU = np.zeros((2, len(divertor[1])))
                divD = np.zeros((2, len(divertor[1])))
                div[0], divU[0], divD[0] = divertor[0], \
                    divertor[0], divertor[0]

            lin = np.linspace(
                np.mean(divertor[1][:5]),
                np.mean(divertor[1][-5:]),
                np.shape(divertor[1])[0])
            divertor[1] += -lin

            if tag[-1] == '1':
                divU[1][:np.shape(divertor[1])[0]] += divertor[1]
            elif tag[-1] == '0':
                divD[1][:np.shape(divertor[1])[0]] += divertor[1]
            divertors.append([
                divertor[0], divertor[1]])

        div[1][:np.shape(divU[1])[0]] += ((5 - missingU) / 5) * divU[1]
        div[1][:np.shape(divD[1])[0]] += ((5 - missingD) / 5) * divD[1]

        lin = np.linspace(
            np.mean(div[1][:5]),
            np.mean(div[1][-5:]),
            np.shape(div)[1])
        div[1] += -lin
        print('\\\ divertor updown asym:', missingU, '/', missingD)

    if np.shape(div)[1] == 0:
        print('\\\ div power is empty, no power target load')
    if passVals or np.shape(div)[1] == 0:
        return (
            ECRH, HBC, VBC, wdia,
            dwdia_dt, div, divertors)

    [ECRH, HBC, VBC, wdia, dwdia_dt, div] = get_longest_ind(
        [ECRH, HBC, VBC, wdia, dwdia_dt, div], t0, t4)

    # power balance
    # 0 = P_ecrh - P_rad - dWdia_dt - P_load
    balance = np.array([
        ECRH[0],
        ECRH[1] - HBC[1] - dwdia_dt[1] - div[1],
        ECRH[1] - VBC[1] - dwdia_dt[1] - div[1],
        ECRH[1] - (HBC[1] + VBC[1]) / 2. - dwdia_dt[1] - div[1]])

    tomo_balance = tomograms_power_balance(
        tomo_powers, ECRH, dwdia_dt, div)

    if plot:
        mfrp.power_balance_plot(
            program=program, ecrh=ECRH, hbc=HBC, vbc=VBC, dwdia_dt=dwdia_dt,
            div_load=div, balance=balance, tomograms=tomo_balance)

    return (
        ECRH, HBC,
        VBC, wdia,
        dwdia_dt, div,
        balance, tomo_balance)


def power_balance(
        program='20181010.032',
        debug=False):

    program_info = api.xpid_info(program=program)[0]
    t0 = program_info['programs'][0]['trigger']['1'][0]  # in ns
    t4 = program_info['programs'][0]['trigger']['4'][0]  # in ns

    try:  # ECRH
        data = api.download_single(
            api.download_link(name='ECRH'),
            program_info=None, filter=None,
            start_POSIX=t0, stop_POSIX=t4)
        ECRH = np.array([  # s and MW from T0
            [(t - t0) / 1.e9 for t in data['dimensions']],
            [v * 1.e3 for v in data['values']]])
    except Exception:
        print('\\\ ecrh failed')
        return (None, None, None, None, None, None, None, None)

    try:  # my calc
        dHBC = api.download_single(
            api.download_link(name='PradHBCm'),
            program_info=None, filter=None,
            start_POSIX=t0, stop_POSIX=t4)
        dVBC = api.download_single(
            api.download_link(name='PradVBC'),
            program_info=None, filter=None,
            start_POSIX=t0, stop_POSIX=t4)
        f = 1.e-6

    except Exception:  # archive
        print('\\\ self prad failed')

    if ((np.shape(dHBC['values'])[0] == 0) or
            (np.shape(dVBC['values'])[0] == 0)):
        try:
            dHBC = api.download_single(
                api.download_link(name='Prad HBC'),
                program_info=None, filter=None,
                start_POSIX=t0, stop_POSIX=t4)
            dHBC['values'] = dHBC['values'][0]
            dVBC = api.download_single(
                api.download_link(name='Prad VBC'),
                program_info=None, filter=None,
                start_POSIX=t0, stop_POSIX=t4)
            dVBC['values'] = dVBC['values'][0]
            f = 1.e-3

        except Exception:
            print('\\\ archive prad failed')
            return (None, ECRH, None, None, None, None, None, None)

    time = [(x - t0) / 1e9 for x in dHBC["dimensions"]]
    HBC = np.array([time, [x / f for x in dHBC['values']]])
    VBC = np.array([time, [x / f for x in dVBC['values']]])

    try:  # Wdia
        wdia = api.download_single(
            api.download_link(name='W_dia'),
            program_info=None, filter=None,
            start_POSIX=t0, stop_POSIX=t4)

        wdia = np.array([
            [(t - t0) / 1e9 for t in wdia['dimensions']],
            [x * 1.e3 for x in wdia['values']]])
        dwdia_dt = np.array([
            wdia[0], np.gradient(wdia[1]) / (wdia[0, 1] - wdia[0, 0])])

    except Exception:
        print('\\\ failed wdia')
        return (None, ECRH, HBC, VBC, None, None, None, None)

    if (program == '20181010.032'):
        L = np.loadtxt('../files/20181010.032_total_divertor_load.txt')
        time = np.array([(t - t0) / 1e9 for t in L[:, 0]])
        div = np.array([time, np.array([x for x in L[:, 1]])])
        divertors = np.array([])  # empty

    else:
        [L0, Ls, LY] = \
            ['AEF', [
             '10', '20', '30', '40', '50',
             '11', '21', '31', '41', '51'],
             'heat load [MW]']

        missingU, missingD = 0, 0
        div, divU, divD = np.zeros((2, 0)), \
            np.zeros((2, 0)), np.zeros((2, 0))
        divertors = []

        for i, tag in enumerate(Ls):
            divertor = api.download_single(
                api.download_link(name='DivAEF' + tag),
                program_info=pinfo,
                start_POSIX=start, stop_POSIX=stop)
            divertor = np.array([
                [(t - t0) / 1.e9 for t in divertor['dimensions']],
                divertor['values']])

            if np.shape(divertor)[1] == 0:
                print('\\\ AEF' + tag, 'not found')
                if tag[-1] == '1':
                    missingU += 1
                elif tag[-1] == '0':
                    missingD += 1
                continue

            if np.shape(div)[1] == 0:
                div = np.zeros((2, len(divertor[1])))
                divU = np.zeros((2, len(divertor[1])))
                divD = np.zeros((2, len(divertor[1])))
                div[0], divU[0], divD[0] = divertor[0], \
                    divertor[0], divertor[0]

            lin = np.linspace(
                np.mean(divertor[1][:5]),
                np.mean(divertor[1][-5:]),
                np.shape(divertor[1])[0])
            divertor[1] += -lin

            if tag[-1] == '1':
                divU[1][:np.shape(divertor[1])[0]] += divertor[1]
            elif tag[-1] == '0':
                divD[1][:np.shape(divertor[1])[0]] += divertor[1]
            divertors.append([
                divertor[0], divertor[1]])

        div[1][:np.shape(divU[1])[0]] += ((5 - missingU) / 5) * divU[1]
        div[1][:np.shape(divD[1])[0]] += ((5 - missingD) / 5) * divD[1]

        lin = np.linspace(
            np.mean(div[1][:5]),
            np.mean(div[1][-5:]),
            np.shape(div)[1])
        div[1] += -lin

    if np.shape(div)[1] == 0:
        print('\\\ div power empty')
        return (None, ECRH, HBC, VBC, wdia, dwdia_dt, None, None)

    # try:
    [ECRH, HBC, VBC, wdia, dwdia_dt, div] = get_longest_ind(
        [ECRH, HBC, VBC, wdia, dwdia_dt, div], t0, t4)
    # except Exception as e:
    #     print('\\\ failed balance')
    #     return (None, ECRH, HBC, VBC, wdia, dwdia_dt, div, divertors)

    balance = np.array([  # power balance
        ECRH[0],
        ECRH[1] - HBC[1] - dwdia_dt[1] - div[1],
        ECRH[1] - VBC[1] - dwdia_dt[1] - div[1],
        ECRH[1] - (HBC[1] + VBC[1]) / 2. - dwdia_dt[1] - div[1]])
    return (balance, ECRH, HBC, VBC, wdia, dwdia_dt, div, divertors)
