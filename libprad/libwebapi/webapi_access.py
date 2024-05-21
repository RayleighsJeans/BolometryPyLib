""" **************************************************************************
    so header """

import os
import sys
import warnings
import requests
import numpy as np
import pprint

import datetime
import time as ttt

import dat_lists as dat_lists

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")

Z = np.zeros
one = np.ones

stdflush = sys.stdout.flush
stdwrite = sys.stdout.write

""" eo header
************************************************************************** """


def xpid_info(
        base_URI='http://archive-webapi.ipp-hgw.mpg.de/',
        program='20181010.032',
        debug=False):
    """ returns program info
    Args:
        base_URI (str, optional):
        program (str, optional):
        shot (int, optional):
        date (str, optional):
        debug (bool, optional):
    Returns:
        program_info (dict):
        req (http.request):
    """
    if debug:
        print('\t>> Looking for ' + program)
    req = requests.get(base_URI + 'programs.json' +
                       '?from=' + program)
    return (req.json(), req)


def download_link(
        base_URI='http://archive-webapi.ipp-hgw.mpg.de/',
        name='BoloSignal'):
    """ returns download link for given tag/name
    Args:
        base_URI (str, optional):
        name (str, optional):
    Returns:
        data_req (str): Link
    """
    data_req, data_labels = \
        dat_lists.link_list()
    # search and return
    for i, tag in enumerate(data_labels):
        if tag == name:
            return (data_req[i])
    return (None)


def return_data(
        indent_level='\t',
        req='<Response [200]>',
        date='20181010',
        shotno='032',
        base_URI='http://archive-webapi.ipp-hgw.mpg.de/',
        POSIX_from='1512651791100887241',  # in ns
        POSIX_upto='1512651807099287241',  # in ns
        epoch_time=False,
        printing=False):
    """ data needed from the archive after trying very hard.
    Args:
        indent_level (0, str): Printing indentation.
        req (1, str): HTTP request format for shot and date.
        date (2, str): Specified date.
        shotnot (3, str): Spec. experiment ID.
        POSIX_from (4, int): Epoch time [ns] from up to.
        POSIX_upto (5, int): Epoch time [ns] up to.
        epoch_time (6, bool): Bool whether POSIX time or XP ID.
        base_URI (7, str): Link to get stuff from.
    Returns:
        data_object (0, list): Full list of downloaded archive entries.
        params_object (1, list): param object list
    Notes:
        None.
    """
    [reqStat, i] = [False, 0]  # pre-set looper
    while not reqStat and (i <= 10):  # time limit ~3mins
        i += 1
        program_info = xpid_info(program=date + '.' + shotno)[0]
        data_object, params_object = \
            check_bolo_data(
                waiting_iterator=i, epoch_time=epoch_time,
                POSIX_from=POSIX_from, POSIX_upto=POSIX_upto,
                program=date + '.' + shotno, program_info=program_info,
                printing=printing)
        try:
            reqStat = True
            print(data_object.keys()[0])
            for k, tag in enumerate(data_object.keys()):  # only core dat
                if data_object[tag]['dimensions'] == []:
                    reqStat = False
                    break

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
                  'stat ', reqStat, ' -- i -- ', i)

        if not reqStat:
            ttt.sleep(1)
        elif reqStat:
            break

    if printing:
        print(indent_level + '>> Finished check_bolo_data at ' + str(i))
    return data_object, params_object


def download_single(
        data_req='http://archive-webapi.ipp-hgw.mpg.de/Test',
        program_info={'none': None},
        start_POSIX=1512651791100887241,  # in ns
        stop_POSIX=1512651807099287241,  # in ns
        filter=None,
        debug=False):
    """  single donwload link func
    Args:
        data_req (string, required): Link to download from
        program_info (dict, required): Program info req
        headers (dict, optional): Default filter info
        start_POSIX (int, optional): Start time
        stop_POSIX (int, optional): Stop time
        debug (bool, optional): debugging decision
    Returns:
        dat (dict): Data object downloaded
    """
    if program_info is None:  # 20171207.024
        try:
            filterstart = str(start_POSIX)  # in ns
            filterstop = str(stop_POSIX)  # in ns

        except Exception:
            filterstart = str(1512651791100887241)  # in ns
            filterstop = str(1512651807099287241)  # in ns
    else:
        filterstart = str(program_info['programs'][0]['from'])  # in ns
        filterstop = str(program_info['programs'][0]['upto'])  # in ns

    if data_req is None:  # 20171207.024, PradHBCm
        data_req = \
            'http://archive-webapi.ipp-hgw.mpg.de/Test/raw/' + \
            'W7XAnalysis/QSB_Bolometry/PradHBCm_DATASTREAM/V2/' + \
            '0/PradHBCm/' + \
            '_signal.json?from=1512651791100887241&upto=1512651807099287241'

    elif ('_signal.json?from=' not in data_req):
        if filter is None:
            data_req += '_signal.json?from=' + filterstart + \
                '&upto=' + filterstop
        else:
            data_req += '_signal.json?filterstart=' + filterstart + \
                '&filterstop=' + filterstop
    else:
        pass
    dat = requests.get(data_req).json()

    if debug:
        print('\nURL: ', data_req, '\n')
        pprint.pprint(dat, depth=1)
    return dat


def get_thomson_traces(
        XPID='20181010.032'):
    # links to the location of thomson scattering data
    base_URI = 'http://archive-webapi.ipp-hgw.mpg.de/'
    base0 = 'Test/raw/W7X/QTB_Profile/volume_'
    base1 = '_DATASTREAM/V'
    # profiles
    base200 = '/1/ne_map/'
    base210 = '/0/Te_map/'
    # errors
    base201 = '/1/ne_s/'
    base211 = '/0/Te_s/'

    # load experiment information
    program_info, req = xpid_info(program=XPID)
    start = str(program_info['programs'][0]['from'])
    stop = str(program_info['programs'][0]['upto'])

    N, T = [], []  # lists of available results
    for volume in range(1, 17):  # total available volumes
        n, t = np.zeros((2, 0)), np.zeros((2, 0))

        for version in range(21, 0, -1):  # from highest possible version down
            link_ne = base_URI + base0 + str(volume) + \
                base1 + str(version) + base200
            foo = download_single(
                link_ne, program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)
            # if nothing is in there, skip, else
            if np.shape(foo['values'])[0] > 0:
                n = np.array([
                    foo['dimensions'],
                    foo['values']])

                link_ne = base_URI + base0 + str(volume) + \
                    base1 + str(version) + base201
                ns = download_single(
                    link_ne, program_info=program_info,
                    start_POSIX=start, stop_POSIX=stop)['values']
                break  # found version, highest equals best, stop!

        for version in range(21, 0, -1):  # same for temperature
            link_Te = base_URI + base0 + str(volume) + \
                base1 + str(version) + base210
            foo = download_single(
                link_Te, program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)
            # highes is best, if containing data, stop
            if np.shape(foo['values'])[0] > 0:
                t = np.array([
                    foo['dimensions'],
                    foo['values']])

                link_Te = base_URI + base0 + str(volume) + \
                    base1 + str(version) + base211
                ts = download_single(
                    link_Te, program_info=program_info,
                    start_POSIX=start, stop_POSIX=stop)['values']
                break

        # write names and errors next to it
        if np.shape(n)[1] > 0:
            N.append([str(volume), n, ns])
        if np.shape(t)[1] > 0:
            T.append([str(volume), t, ts])

    print('>> loaded ne/Te QTB volumes:', end=' ')
    for d in N:
        print(d[0], end=' ')
    print('\n')
    return (N, T)


def load_non_XP_ids(
        YY=[2020, 2020],
        MM=[9, 9],
        DD=[25, 25],
        hh=[0, 23],
        mm=[0, 59],
        ss=[0, 59],
        start_search=None,
        stop_search=None,
        location='http://archive-webapi.ipp-hgw.mpg.de/Test/raw/W7X' +
                 '/QSB_Bolometry/BoloSignal_DATASTREAM/',
        debug=False):
    """ find filters for time frame given by utc timestamps
    Args:
        YY (list, optional): Years. Defaults to [2020, 2020].
        MM (list, optional): Months. Defaults to [9, 9].
        DD (list, optional): Days. Defaults to [25, 25].
        hh (list, optional): Hours. Defaults to [0, 23].
        mm (list, optional): Minutes. Defaults to [0, 59].
        ss (list, optional): Seconds. Defaults to [0, 59].
        start_search ([type], optional): POSIX start search. Defaults to None.
        stop_search ([type], optional): POSIX stop search. Defaults to None.
        location (str, optional): Where to look.
            Defaults to 'http://archive-webapi.ipp-hgw.mpg.de/Test/raw/W7X'+
            '/QSB_Bolometry/BoloSignal_DATASTREAM/'.
        debug (bool, optional): Debugging. Defaults to False.
    Returns:
        Filters (ndarray): Starts and stops of entries.
        dat (dict): Original return
    """

    if (start_search is None) or (stop_search is None):
        filterstart = int(datetime.datetime(
            YY[0], MM[0], DD[0], hh[0], mm[0], ss[0]).timestamp() * 1e9)
        filterstop = int(datetime.datetime(
            YY[1], MM[1], DD[1], hh[1], mm[1], ss[1]).timestamp() * 1e9)
    else:
        filterstart = int(start_search)
        filterstop = int(stop_search)

    # get entries at location and return filter-list?
    dat = download_single(
        data_req=location, program_info=False,
        start_POSIX=filterstart, stop_POSIX=filterstop,
        filter=True, debug=debug)

    try:
        filters = np.zeros((2, len(dat['_links']['children'])), dtype='int64')
        for i, entry in enumerate(dat['time_intervals']):
            filters[0, i], filters[1, i] = entry['from'], entry['upto']
        return (filters, dat)

    except Exception:
        print('\\\ failed loading filterstarts and stops for ' +
              'YY-MM-DD: ' + str(YY) + '-' + str(MM) + '-' + str(DD))

    if 'filters' not in locals():
        return (None, None)
    else:
        return (filters, dat)


def check_bolo_data(
        waiting_iterator=1,
        epoch_time=False,
        POSIX_from='1539179650333392400',  # in ns
        POSIX_upto='1539179729933392400',  # in ns
        base_URI='http://archive-webapi.ipp-hgw.mpg.de/',
        program='20181010.032',
        program_info={'none': None},
        printing=False,
        indent_level='\t'):
    """ Routine that looks at the archive DB and grabs whatever
        is or has been specified in the lists of names and URLS
        ('Earls') prior to that. Uses a very bad hack to put those
        data in a single big dictionary that can be passed on to
        the top level function
    Args:
        waiting_iterator (0, int): Integer of waiting round currently at
        epoch_time (1, bool): bool to chose program or specified epoch time
        POSIX_from (2, int): Nanosecond timing since 01.01.1970 for start
        POSIX_upto (3, int): Nanosecond timing since 01.01.1970 for stop
        base_URI (4, str): Archive DB link base
        program (6, str): Said program info and dictionary to date
        program_info (7, dict): Info of said program in dictionary
        indent_level (8, str): Printing indentation level for clean looks
    Returns:
        data (0, dict): arrays of previously specified links and properties.
    Notes:
        None.
    """

    if waiting_iterator == 1 and printing:
        print(indent_level + '>> Get bolometer data and everything else...')

    # Get start / stop nanoseconds by using the program id
    data = []
    params = []

    if not (epoch_time):
        try:
            start = str(program_info['programs'][0]['from'])  # in ns
            stop = str(program_info['programs'][0]['upto'])  # in ns

        except Exception:
            program_info = xpid_info(program=program)[0]
            start = str(program_info['programs'][0]['from'])  # in ns
            stop = str(program_info['programs'][0]['upto'])  # in ns

    else:
        start = str(POSIX_from)  # in ns
        stop = str(POSIX_upto)  # in ns

    # links and labels
    bolo_list = (0, 1, 2, 3, 4, 5, 6, 16, 17, 25, 26)
    data_req, data_labels = dat_lists.link_list()
    names = [data_labels[k] for k in bolo_list]

    data = {}
    for i, tag in enumerate(names):
        data[tag] = download_single(
            download_link(name=tag), program_info=program_info,
            start_POSIX=start, stop_POSIX=stop)

    params = {
        'BoloSignal PARLOG': download_single(
            download_link(name='BoloSignal PARLOG'),
            program_info=program_info,
            start_POSIX=start, stop_POSIX=stop)}

    try:  # loading the time
        if int(program[:8]) >= 20180718:
            raw_time = data['BoloSignal']["dimensions"]  # time POSIX in ns
            new_time, nsT, sT, arg, mode, params = dat_lists.timing_fix(
                raw_time=raw_time, params=params, printing=printing,
                program=program, mode='lab', indent_level=indent_level)
            data['BoloSignal']['dimensions'] = new_time  # in ns

        else:
            print(indent_level + '\t\\\ old campaing, not timing fix needed')

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname,
              exc_tb.tb_lineno, '\n', indent_level +
              '\t\\\ failed to find meas.time, no fix')

    return data, params  # see dat.lists.py


def do_before_running(
        program_info={'none': None},
        program='20171207.24',
        date='20171207',
        data_object={'none': {'none': None}},
        geom_input=None,
        strgrid=None,
        magconf=None,
        printing=False,
        indent_level='\t'):
    """ Lists all constants, variables and necessary quantities for
        the current configuration of the machine and the bolometer
        itself
    Args:
        program_info (dict, optional): python request results from the
            archive/logbook api with "all" the necessary info and timestamps
            on the shot
        program (string, optional): Shot/physics program info to the
            request query
        data_object (list, optional): all physics relevant (or even
            irrelephant) data from archive
        indent_level (string, optional): Printing indentation leven for
            inline output
    Returns:
        priority_data : Dictionary of the full data gathered/specified in
            in this routine
        bad_key_error: Since the server response is kinda finnicky, this bool
            triggers a repeat of True in the host routine after
            a certain down time
    Notes:
        None.
    """

    # constant values
    C_cab = 2.0e-9  # in F, that's nF
    RC = 41.  # in Ohm
    f_bridge = 2500.  # in Hz, 2.5kHz
    nn = 300  # number of points

    if (int(date) <= 20171207):
        # torus plasma volume
        volume_torus = 52.381772  # in m^3
        # volume_torus_FTM_beta0 = 39831410.
    elif (int(date) > 20171207):
        # for high mirror and standard EIM/EJM
        volume_torus = 45.  # 52.4542  # from cell integration off of MFR
        # 41.028  # EIM ref 1 x 1.3 (31.56m^3)
        # DAZ says 45.  # in m^3, 1e6 cm^3

    U0 = 5.0  # in V
    f_tran0 = 0.53  # unitless, %
    f_tran = np.ones((128))
    f_tran[:32] = [  # changed transmission because mesh, other cams parallel
        0.886, 0.886, 0.886, 0.886, 0.943, 0.943, 0.943, 0.943, 0.981,
        0.981, 0.981, 0.981, 1., 1., 1., 1., 1., 1., 1., 1., 0.980,
        0.980, 0.980, 0.980, 0.944, 0.944, 0.944, 0.944, 0.882, 0.882,
        0.882, 0.882]
    f_tran = f_tran * f_tran0

    # debugging when dimensions is somehow not found
    priority_data = {'none': None}
    if printing:
        print(indent_level + '>> Dump Run-Once stuff to priority_data...')

    try:
        trgt = 'http://archive-webapi.ipp-hgw.mpg.de/programs.json'
        bar = requests.get(trgt, params={'from': program}).json()
        t0 = bar['programs'][0]['trigger']['1'][0]  # in ns
        t4 = bar['programs'][0]['trigger']['4'][0]  # in ns

    except Exception:  # time or program info not available/filled
        print(indent_level + '\t\\\  cant find T0/T4 in new p.info')

        try:
            t0 = program_info["programs"][0]["trigger"]["1"][0]  # in ns
            t4 = program_info["programs"][0]["trigger"]["4"][0]  # in ns

        except Exception:
            print(indent_level + '\t\\\  cant find T0/T4 in old p.info')

            try:
                t0 = data_object['BoloSignal'][
                    'dimensions'][0] + 1e9 * 10  # T0, in ns, 10s into meas
                t4 = data_object['BoloSignal'][
                    'dimensions'][0] + 1e9 * 20  # T4, in ns, 10s into meas

            except Exception:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(indent_level + '\t\\\ ', exc_type, fname,
                      exc_tb.tb_lineno, '\n', indent_level +
                      '\t\\\  failed T0/T4 search')

                return priority_data, True

    try:  # catch bad performance of program info and trigger timings
        time = [(x - t0) / (1e9) for x in data_object[
            'BoloSignal']["dimensions"]]  # time from start in s

    except Exception:  # time or program info not available/filled
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname,
              exc_tb.tb_lineno, '\n', indent_level,
              '\t\\\  Failed at time')
        return None, True

    # TIME TIME TIME
    try:  # get ECRH off time from program info
        ecrh_off = (program_info["programs"][0]["trigger"]["4"][0] -
                    program_info['programs'][0]['trigger']['1'][0]) / \
                   (1e9) + 1.0  # time ECRH is off in s

        # the ECRH off time is beyond the sampling of the bolometer
        if (ecrh_off > time[-1]):
            ecrh_off = time[-1]  # in s

    except Exception:  # couldnt get the ECRH off time
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname,
              exc_tb.tb_lineno, '\n',
              indent_level, '\t\\\  ECRH_OFF time not in program info')
        return priority_data, True

    # read and input from geometry files
    v = dat_lists.geom_dat_to_json(
        saving=False, geom_input=geom_input,
        strgrid=strgrid, magconf=magconf, printing=printing)

    try:
        # screening for T1/T2/T3/T4? also, flip time of signal
        t_it = 0
        t_it2 = 0
        t_it3 = 0
        while time[t_it3] < ecrh_off:  # ECRH off
            t_it3 += 1
            if time[t_it2] < 0.0:  # ECRH on
                t_it2 += 1
            if time[t_it] < 0.2:  # check polarity of channel
                t_it += 1
        dt = (time[1] - time[0])  # in s
        f_bridge = 1.0 / dt  # in Hz

        # dump the long list of constants to the priority data
        priority_data = {
            'C_cab': C_cab,  # in F
            'RC': RC,  # in Ohm
            'nn': nn,  # in points
            'volume_torus': volume_torus,  # in m^3
            'U0': U0,  # in V
            'f_tran': f_tran,  # unitless, %
            'f_bridge': f_bridge,  # in Hz
            'time': time,  # in s
            't_it': t_it,  # in points
            't_it2': t_it2,  # in points
            't_it3': t_it3,  # in points
            'ecrh_off': ecrh_off,  # in s
            'dt': dt,  # in s
            't0': t0,  # in ns
            't4': t4,  # in ns
            'geometry': v}

        return priority_data, False

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level,
              '\t\\\  Exception at do_before_running with error\n' +
              indent_level + '\t\\\ ',
              exc_type, fname, exc_tb.tb_lineno)

        return priority_data, True
