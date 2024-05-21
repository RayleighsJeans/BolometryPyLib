""" **************************************************************************
    so header """

import warnings
import numpy as np
import requests
import statistics as stats
import json

import mClass
import webapi_access as api
import dat_lists as lists
import prad_calculation as prad

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")

Z = np.zeros
ones = np.ones
M, N = 10000, 100000

""" eo header
************************************************************************** """


def op_statistics(
        year='2018',
        vers='V4',
        debug=False):
    base = 'http://archive-webapi.ipp-hgw.mpg.de/' + \
        'Test/raw/W7XAnalysis/QSB_Bolometry/'
    suffix = '_DATASTREAM/' + vers + '/'

    op_days = lists.operation_days()[1]
    dates = [x for x in op_days if year in x]
    D = {}

    for d, day in enumerate(dates):
        req = requests.get(  # number of experiments that day
            url='https://w7x-logbook.ipp-hgw.mpg.de/api/' +
            'search.html?&q=id:XP_' + day + '.*').json()

        for shot in range(1, req['hits']['total'] + 1):  # experiments
            P = day + '.' + str(shot).zfill(3)
            print('\t>> ' + P)

            D[P] = {
                'std_dev': Z((128)),
                'abs_offs': Z((128)),
                'slope': Z((128)),
                'raw_max': Z((128)),
                'raw_min': Z((128)),
                'adj_max': Z((128)),
                'adj_median': Z((128)),
                'Kappam': Z((128)),
                'Rohm': Z((128)),
                'Taum': Z((128))}

            try:
                program_info = api.xpid_info(program=P)[0]
                start = str(program_info['programs'][0]['from'])
                stop = str(program_info['programs'][0]['upto'])
            except Exception:
                print('\t\t\t\\\ failed program info on ' + P)
                continue

            try:
                dat = {'BoloSignal': api.download_single(
                    api.download_link(name='BoloSignal'),
                    program_info=program_info,
                    start_POSIX=start, stop_POSIX=stop)}
            except Exception:
                print('\t\t\t\\\ failed Ud download on ' + P)
                continue

            try:
                prio = api.do_before_running(
                    program_info=program_info, program=P,
                    date=day, data_object=dat, indent_level='\t\t')[0]

                n_raw = dat['BoloSignal']['dimensions'][prio['t_it2']:-1]
                matrix = np.vstack([n_raw, np.ones(len(n_raw))]).T
            except Exception:
                print('\t\t\t\\\ failed prio obj on ' + P)
                continue

            for ch in range(0, 128):
                try:
                    vadj, _, _, D[P]['std_dev'][ch], _, _, D[P]['slope'][ch], \
                        D[P]['abs_offs'][ch] = prad.major_function(
                            voltage=dat['BoloSignal']['values'][ch],
                            t_it=prio['t_it'], t_it2=prio['t_it2'],
                            U0=prio['U0'], rc=prio['RC'],
                            f_bridge=prio['f_bridge'],
                            c_cab=prio['C_cab'], dt=prio['dt'],
                            n_raw=n_raw, f_tran=prio['f_tran'][ch],
                            M=matrix, make_linoffs=True)
                except Exception:
                    print('\t\t\t\\\ failed major function on ' +
                          P + ' ch' + str(ch))
                    continue

                try:
                    D[P]['raw_max'][ch] = max(dat['BoloSignal']['values'][ch])
                    D[P]['raw_min'][ch] = min(dat['BoloSignal']['values'][ch])
                    D[P]['adj_max'][ch] = max(vadj)
                    D[P]['adj_median'][ch] = stats.median(vadj[
                        prio['t_it2']:prio['t_it3']])
                except Exception:
                    print('\t\t\t\\\ failed Ud stats on ' + P + ' ch' + str(ch))
                    continue

            try:
                D[P]['Kappam'] = np.array(api.download_single(
                    base + 'MKappa' + suffix, program_info=program_info,
                    start_POSIX=start, stop_POSIX=stop)['values'][0])
                D[P]['Rohm'] = np.array(api.download_single(
                    base + 'MRes' + suffix, program_info=program_info,
                    start_POSIX=start, stop_POSIX=stop)['values'][0])
                D[P]['Taum'] = np.array(api.download_single(
                    base + 'MTau' + suffix, program_info=program_info,
                    start_POSIX=start, stop_POSIX=stop)['values'][0])
            except Exception:
                print('\t\t\t\\\\ failed calib params on ' + P)
                continue

        try:
            with open('op_' + year + '_statistics.json', 'w') as outfile:
                outdict = mClass.dict_transf(D, to_list=True)
                json.dump(outdict, outfile, indent=4, sort_keys=False)
            outfile.close()
        except Exception:
            print('\t\t\t\\\ failed saving')

    return (D)


def get_op_statistics(
        file='op_2018_statistics_v3.json',
        debug=False):

    with open('../files/' + file, 'r') as infile:
        data = json.load(infile)
    infile.close()

    return (mClass.dict_transf(data, to_list=False))
