""" **************************************************************************
    so HEADER """

import sys
import os
import time as ttt
import json
import numpy as np
import requests
import mClass

stdwrite = sys.stdout.write
stdflush = sys.stdout.flush

Z = np.zeros
one = np.ones

""" eo HEADER
************************************************************************** """


def operation_days(
        date='20181010',
        shot=32,
        indent_level='\t'):
    """ Returns list of all operation days (OP11/OP12)
    Args:
        indent_level (0, str): Indentation level
        date_nrboolean (1, str): Nr. or bool of date/for date selection
        shot_nrboolean (2, int): XP ID or bool
    Returns:
        dateno (0, str): Date number in list
        op_days(1, list): list of operation days as strings
        date (2, str): Date selected
        shot (3, int): XP ID selected
    Notes:
        None.
    """

    op11_ls = \
        ['20151209', '20151210', '20151211', '20151214', '20151215',
         '20160112', '20160113', '20160114', '20160119', '20160120',
         '20160121', '20160126', '20160127', '20160128', '20160202',
         '20160203', '20160204', '20160209', '20160210', '20160216',
         '20160217', '20160218', '20160223', '20160224', '20160225',
         '20160301', '20160302', '20160308', '20160309', '20160310'
         ]
    op12a_ls = \
        ['20170906', '20170907', '20170912', '20170913', '20170914',
         '20170919', '20170920', '20170921', '20170926', '20170927',
         '20171004', '20171005', '20171010', '20171011', '20171012',
         '20171017', '20171018', '20171019', '20171024', '20171025',
         '20171026',
         '20171101', '20171107', '20171108', '20171109',
         '20171114', '20171115', '20171121', '20171122', '20171123',
         '20171124', '20171129',
         '20171205', '20171206', '20171207'
         ]
    op12b_ls = \
        ['20180718', '20180719', '20180724', '20180725', '20180726',
         '20180731', '20180801', '20180807', '20180808', '20180809',
         '20180814', '20180815', '20180821', '20180822', '20180823',
         '20180906', '20180911', '20180912', '20180913',
         '20180918', '20180919', '20180920', '20180925', '20180926',
         '20180927', '20181002', '20181004', '20181009', '20181010',
         '20181011', '20181016', '20181017', '20181018']

    # SKIP, where possible flips happened
    # '20180828', '20180829', '20180830', '20180904', '20180905',

    op11_ls.extend(op12a_ls)
    op11_ls.extend(op12b_ls)
    op_days = op11_ls
    dateno = [0, False]

    if not date:
        date = ttt.strftime("%Y%m%d")
        date = '20170913'
    else:
        pass

    if not shot:
        shot = 1
    else:
        pass

    try:
        dateno[0] = op_days.index(date)
        dateno[1] = True
    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print('\t\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n\t\t\\\  Error on date specification')

        dateno[0] = 0
        dateno[1] = False

    return (dateno, op_days, date, shot)


def overview_time_limits(
        program='20181010.032',
        prio={'none': {'none': None}}):

    if program == '20180822.020':
        time_limits = [2.0940688 - 0.1, 2.4896 + 0.1]
    elif program == '20180822.022':
        time_limits = [2.5970688 - 0.1, 3.1008 + 0.1]
    elif program == '20180823.005':
        time_limits = [0.9351945 - 0.1, 1.2816 + 0.1]
    elif program == '20180823.006':
        time_limits = [2.0207398 - 0.1, 2.0416 + 0.1]
    elif program == '20180823.014':
        time_limits = [2.222693 - 0.1, 2.7888 + 0.1]
    elif program == '20180823.037':
        time_limits = [2.1890736 - 0.1, 2.88 + 0.1]
    elif program == '20180906.036':
        time_limits = [1.9642773 - 0.1, 2.4432 + 0.1]
    elif program == '20180911.009':
        time_limits = [1.6662916 - 0.1, 3.1456 + 0.1]
    elif program == '20180911.012':
        time_limits = [1.383377 - 0.1, 1.792 + 0.1]
    elif program == '20180911.014':
        time_limits = [1.4771382 - 0.1, 1.8288 + 0.1]
    elif program == '20180912.013':
        time_limits = [2.160065 - 0.1, 2.8208 + 0.1]
    elif program == '20180912.022':
        time_limits = [2.062021 - 0.1, 2.8528 + 0.1]
    elif program == '20180912.023':
        time_limits = [2.2269933 - 0.1, 2.9792 + 0.1]
    elif program == '20180918.012':
        time_limits = [2.247584 - 0.1, 3.3424 + 0.1]
    elif program == '20180822.006':
        time_limits = [2.0610688 - 0.1, 2.0608 + 0.1]
    elif program == '20180822.008':
        time_limits = [2.2090688 - 0.1, 2.2544 + 0.1]
    elif program == '20180822.011':
        time_limits = [2.1230688 - 0.1, 2.1888 + 0.1]
    elif program == '20180823.020':
        time_limits = [2.6324167 - 0.1, 3.256 + 0.1]
    elif program == '20180823.022':
        time_limits = [3.8308952 - 0.1, 4.3216 + 0.1]
    elif program == '20180823.039':
        time_limits = [2.4718862 - 0.1, 3.5472 + 0.1]
    elif program == '20180906.025':
        time_limits = [5.007814 - 0.1, 5.6088 + 0.1]
    elif program == '20180911.008':
        time_limits = [0.9111296 - 0.1, 3.3 + 0.1]
    elif program == '20180911.012':
        time_limits = [1.383377 - 0.1, 3.3952 + 0.1]
    elif program == '20180911.014':
        time_limits = [1.4202236 - 0.1, 3.2928 + 0.1]
    else:
        # adjust and scale values of dimensions
        time_limits = [-1., (prio['t4'] - prio['t0']) / (1e9) + 1.5]

    return time_limits


def triggerC_infos(
        prio={'none': {'none': None}},
        program='20181010.032'):
    """ trigger info for trigger plot
    Args:
        priority_object (list, optional): predefined stuff
        date (str, optional): XP date
        shotno (str, optional): XP id
    Returns:
        lists (list): list of timings where to look
    Notes:
        None
    """
    lists = []
    with open('../files/tespel_relevant_data.json', 'r') as f:
        tespel_data = json.load(f)
    f.close()

    try:
        T4 = prio['t4']
        T0 = prio['t0']
        ecrh_off = (T4 - T0) / 1.e9  # s
    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        stdwrite('\t\\\ ' + str(exc_type) + ' ' + str(fname) + ' ' +
                 str(exc_tb.tb_lineno) + '\n\t\\\  ecrh off failed')
        stdflush()
        ecrh_off = 10.0  # s

    ecrh_off = ecrh_off  # + 0.15
    if (program == '20181011.015'):  # LBO timings
        lists = [  # s
            0.0, 1.303, 1.903, 2.453, 3.503, 3.703, 4.303,
            4.853, 5.453, 6.003, 6.603, ecrh_off]
    elif (program == '20181010.032'):
        lists = [0.0, 5.0, ecrh_off]  # s
    elif (program == '20181016.016'):
        lists = [0.0, 22.0, 31.101]  # s
    elif (program == '20180809.012'):
        lists = [0.0, 2.053, 2.054, 3.008, ecrh_off]  # s
    elif (program == '20171012.011'):
        lists = [0.0, 0.551, 1.052, ecrh_off]  # s
    elif (program == '20180919.004'):
        lists = [0.0, 7.8, ecrh_off]  # s
    elif (program == '20180919.018'):
        lists = [0.0, 1.3, 3.75, ecrh_off]  # s

    elif ((program in tespel_data['xpids']) and (int(
            program[0:8]) not in range(20180828, 20180905))):
        it = np.where(np.array(
            tespel_data['xpids']) == program)[0][0]
        lists = [0.0, tespel_data['ablation_times_ms'][it] * 1e-3,  # s
                 tespel_data['tracer_times_ms'][it] * 1e-3, ecrh_off]  # s
    else:
        lists = [0.0, ecrh_off]  # s

    return lists  # s


def timing_fix(
        raw_time=Z((10000)),
        params={'none': {'none': None}},
        program='20181010.032',
        mode='manual',
        printing=False,
        indent_level='\t'):
    """ calculate time fix and shift at T1/T4
    Args:
        raw_time (list, required): Orig time
        params (dict, required): Parlog that holds info of exp
        program (str, required): XP ID
        mode (str, optional): Timing fix scale mode
        indent_level (str, optional): Printing indentation
    Returns:
        new_time (list): Fixed time
        nsT (float): New sample time
        sT (float): Sample time
        arg (list): List of changes regarding T1/T4
        mode (str): Scaling mode of time trace
        params (dict): Appended parameters with new times
    Notes:
        Doesn't apply as expected
    """
    sT = (raw_time[1] - raw_time[0]) * 1e-9  # sample time in s
    sN = len(raw_time)  # sample number

    # for diagnostics
    URL = 'http://archive-webapi.ipp-hgw.mpg.de/programs.json'
    res = requests.get(URL, params={'from': program}).json()
    T1 = res['programs'][0]['trigger']['1'][0]  # ns
    T4 = res['programs'][0]['trigger']['4'][0]  # ns

    arg = Z((2), dtype='int')
    for i, trig in enumerate([T1, T4]):
        bar = np.abs([t - trig for t in raw_time])
        arg[i] = int(np.where(np.min(bar) == bar)[0][0])

    if mode is not None:  # change timebase
        if (('Measured Time for DAQ [ms]' in
             params['BoloSignal PARLOG']['values'][0].keys())):
            skS = 1000  # skipped samples before start of raw_time
        else:
            skS = 0

        # all properties in s
        if (mode == 'lab'):  # bin sT, lab results
            if (1.55e-3 < sT < 1.65e-3):
                nsT, slope, offS, GSF = 1.625e-3, 0.0, 0.0, 1.0
            elif (0.75e-3 < sT < 0.85e-3):
                nsT, slope, offS, GSF = 0.821e-3, 1.005e-09, 0.0, 1.0
            else:
                nsT, slope, offS, GSF = sT, 0.0, 0.0, 1.0

        elif (mode == 'manual'):  # bin the sT, from HEXOS timings
            if (1.55e-3 < sT < 1.65e-3):
                nsT, slope, offS, GSF = sT, 0.0, 84.00e-3, 1.0110
            elif (0.75e-3 < sT < 0.85e-3):
                nsT, slope, offS, GSF = sT, 0.0, 23.891e-3, 1.0  # , 0.999147
            else:
                nsT, slope, offS, GSF = sT, 0.0, 0.0, 1.0

        elif (mode == 'meas'):  # measured CPU timing from LabVIEW

            if (('loop DAQ time [ms]' in
                 params[0]['values'][0].keys())):
                nsT = params[0]['values'][0]['loop DAQ time [ms]']
                slope, skS, offS, GSF = 0.0, 1000, 0.0, 1.0

            else:
                mT = params['BoloSignal PARLOG'][
                    'values'][0]['Measured Time for DAQ [ms] ']
                pT = params['BoloSignal PARLOG'][
                    'values'][0]['predicted time [ms] to finalize DAQ']
                # measure time of acquisition as fix
                nsT, slope, skS, offS, GSF = \
                    mT / pT * sT, 0.0, 1000, 0.0, 1.0

        preDAQ_skip = ((nsT - sT) + slope) * skS + offS  # shifted, s
        new_time = \
            [raw_time[0] + (preDAQ_skip + (nsT + slope) * n) * 1e9 * GSF
             for n, t in enumerate(np.linspace(0, 1, sN))]  # in ns

    else:  # no mode given, take the OG timebase
        nsT = sT
        new_time = raw_time

    iL = indent_level
    if printing:
        print(iL + '\t\\\ timing error, mode: ' + str(mode) + '\n' +
              iL + '\t\\\ ' + str(round(((nsT / sT) - 1) * 100, 3)) +
              '%, new sample time:' + str(round(nsT * 1.e3, 3)) + 'ms\n' +
              iL + '\t\\\ shift around T1/T4: ' +
              str(round((new_time[arg[0]] - raw_time[arg[0]]) / 1e6, 3)), '/',
              str(round((new_time[arg[1]] - raw_time[arg[1]]) / 1e6, 3)), 'ms')

    params['raw time'] = raw_time  # in ns
    params['new time'] = new_time  # in ns
    params['new sample time'] = nsT  # s
    params['old sample time'] = (raw_time[1] - raw_time[0]) / 1e9  # s
    params['T1 difference'] = (new_time[arg[0]] - raw_time[arg[0]]) / 1e9  # s
    params['T4 difference'] = (new_time[arg[1]] - raw_time[arg[1]]) / 1e9  # s

    return new_time, nsT, sT, arg, mode, params


def freq_QSB(
        SF_decimal=None,
        SF_hex=None,
        f_master=5e6,  # Hz
        chopmode=True):
    """ short function to understand filter settings
    Args:
        SF_decimal (None, required): filter register decimal
        SF_hex (None, required): filter register hex
        f_master (float, optional): mast clock
        chopmode (bool, required): chop mode on/off
    Returns:
        sample_req (float): frequency in mhz
        sample_time (float): time in ms
    Notes:
        None
    """
    if SF_hex is not None:
        print('>> hex, chopmode:', chopmode)
        SF_decimal = int(str(SF_hex), 16)

    print(SF_decimal)

    if SF_decimal is not None:
        if SF_decimal.__class__ == 'str':
            SF_decimal = int(SF_decimal, 16)
        print('>> decimal, chopmode:', chopmode)

        if chopmode:
            f_in = (f_master) / 16 * 1 / (SF_decimal)
        elif not chopmode:
            f_in = (f_master) / 16 * 1 / (3 * SF_decimal)

        return (f_in, 1 / f_in)

    else:
        print('>> given nothing, returning nothing')
        return (None, None)


def geom_dat_to_json(
        saving=False,
        geom_input=None,
        strgrid=None,
        printing=False,
        magconf=None):
    """ setting up geometry and effective radii
    Args:
        saving (bool, optional): Should write file?
    Returns:
        v (dict): Completed dictionary with all info
    """

    v = {
        'channels': {'eChannels': {}, 'gChannels': {}},
        'geometry': {},
        'radius': {}
    }

    v['channels']['droplist'] = [
        6, 20, 33, 37, 39, 49, 53, 55, 77, 87, 99, 111]

    """                            EChan        GChan          NChans """
    hbcm_range = \
        np.linspace(0, 31, 32)     # 1 ,...,32  1 ,...,32      #32
    hbcs1_range = \
        np.linspace(32, 47, 16)    # 33,...,48  1 ,...,16      #16
    hbcs2_range = \
        np.linspace(97, 112, 16)   # 97,...,113 17,...,32      #16
    vbcl1_range = \
        np.linspace(48, 63, 16)    # 49,...,64  9 ,...,24      #16
    vbcl2_range = \
        np.linspace(88, 95, 8)     # 89,...,96  1 ,...,8       #8
    vbcr_range = \
        np.linspace(64, 87, 24)    # 65,...,88  1 ,...,24      #24
    sxrv_range = \
        np.linspace(88, 95, 8)     # 89,...,96  1 ,...,8       #8
    sxrh1_range = \
        np.linspace(44, 47, 4)     # 45,...,48  13,...,16      #4
    sxrh2_range = \
        np.linspace(96, 107, 12)   # 97,...,108 17,...,28      #12
    Al_filter1_range = \
        np.linspace(32, 43, 11)    # 33,...,44  1 ,...,11      #11
    Al_filter2_range = \
        np.linspace(108, 112, 5)   # 109,..,113 28,...32       #5
    artf_range = \
        np.linspace(113, 127, 15)  # 113,..,127 1,...,15       # 15

    # concatenating/appending ranges
    hbcs_range = np.append(hbcs1_range, hbcs2_range)
    vbcl_range = np.append(vbcl2_range, vbcl1_range)
    vbc_range = np.append(vbcr_range, np.linspace(54, 63, 10))
    sxrh_range = np.append(sxrh1_range, sxrh2_range)
    Alf_range = np.append(Al_filter1_range, Al_filter2_range)

    # make them iterable
    v['channels']['eChannels']['HBCm'] = [int(h) for h in hbcm_range]
    v['channels']['gChannels']['HBCm'] = [
        int(ch + 1) for ch in hbcm_range]

    v['channels']['eChannels']['HBCs'] = [int(h) for h in hbcs_range]
    v['channels']['gChannels']['HBCs'] = \
        [int(i + 1) for i, ch in enumerate(hbcs_range)]

    v['channels']['eChannels']['VBC'] = [int(v) for v in vbc_range]
    v['channels']['gChannels']['VBC'] = \
        [int(i + 1) if int(ch) <= 88 else int(ch - 39)
         for i, ch in enumerate(vbc_range)]

    v['channels']['eChannels']['VBCr'] = [int(v) for v in vbcr_range]
    v['channels']['gChannels']['VBCr'] = \
        [int(i + 1) for i, ch in enumerate(vbcr_range)]

    v['channels']['eChannels']['VBCl'] = [int(v) for v in vbcl_range]
    v['channels']['gChannels']['VBCl'] = \
        [int(i + 1) for i, ch in enumerate(vbcl_range)]

    v['channels']['eChannels']['SXRv'] = [int(x) for x in sxrv_range]
    v['channels']['gChannels']['SXRv'] = \
        [int(i + 1) for i, ch in enumerate(sxrv_range)]

    v['channels']['eChannels']['SXRh'] = [int(x) for x in sxrh_range]
    v['channels']['gChannels']['SXRh'] = \
        [int(i + 13) for i, ch in enumerate(sxrh_range)]

    v['channels']['eChannels']['ALF'] = [int(a) for a in Alf_range]
    v['channels']['gChannels']['ALF'] = \
        [int(i + 1) if int(ch) < 108 else int(i + 17)
         for i, ch in enumerate(Alf_range)]

    v['channels']['eChannels']['ARTf'] = [int(a) for a in artf_range]
    v['channels']['gChannels']['ARTf'] = [
        i + 1 for i, ch in enumerate(artf_range)]

    ALL = [0.0] * 128
    for i in range(129):
        for cam in v['channels']['eChannels'].keys():
            if i in v['channels']['eChannels'][cam]:
                ALL[i] = cam
    v['channels']['eChannels']['ALL'] = ALL

    if (geom_input is None):
        loc = '../files/geom/camgeo/'

        # effective radii along the fluxsurfaces
        v['radius']['reff'] = [0.0] * 128  # in m
        v['radius']['rho'] = [0.0] * 128  # in m

        for file in ['standard_camHBCm_reffLoS.dat',  # in m
                     'standard_camVBCl_reffLoS.dat',  # in m
                     'standard_camVBCr_reffLoS.dat',  # in m
                     'standard_camARTf_reffLoS.dat']:  # in m
            reff = np.genfromtxt(
                loc + file, comments='#',
                names=['channel', 'reff [m]', 'roh [r_eff/a]'],
                dtype=[int, float, float])
            cam = file.replace('standard_cam', '').replace('_reffLoS.dat', '')

            for i in range(np.shape(reff)[0]):
                try:
                    eChannel_ID = v['channels']['gChannels'][
                        cam].index(reff[i][0])  # in m
                except Exception:
                    print('reff', reff[i][0],
                          'not found in', cam, 'gChannels')
                    continue

                eChannel = v['channels']['eChannels'][cam][eChannel_ID]
                # loaded in cm
                v['radius']['reff'][eChannel] = reff[i][1]  # in m
                v['radius']['rho'][eChannel] = reff[i][2]  # in units of above

        # lines of sight and dectector geometries as well as volumes and k's
        # get volume and bolo data from files for selected hbc and vbc
        v['geometry']['vbolo'] = np.zeros((128))  # in m^3
        v['geometry']['kbolo'] = np.zeros((128))  # in m^3

        for file in ['standard_camHBCm_kbolott_and_volum_PV_new_.dat',
                     'standard_camVBCl_kbolott_and_volum_PV_new_.dat',
                     'standard_camVBCr_kbolott_and_volum_PV_new_.dat',
                     'standard_camARTf_kbolott_and_volum_PV_new_.dat']:
            geom = np.genfromtxt(
                loc + file, names=['kbolo [m^3]', 'vbolo [m^3]'],
                dtype=[float, float])
            cam = file.replace('standard_cam', '').replace(
                '_kbolott_and_volum_PV_new_.dat', '')

            for i in range(np.shape(geom)[0]):
                try:
                    eChannel_ID = v['channels']['gChannels'][
                        cam].index(i + 1)
                except Exception:
                    print('geom', i, 'not found in', cam, 'gChannels')
                    continue
                eChannel = v['channels']['eChannels'][cam][eChannel_ID]
                v['geometry']['kbolo'][eChannel] = geom[i][0]  # in m^3
                v['geometry']['vbolo'][eChannel] = geom[i][1]  # in m^3

    elif (geom_input == 'self'):
        print('\t\t\\\ using own geometry input:', magconf, strgrid)
        reff, rho, kbolott, volume = \
            get_reff_kbolott_volume_self(
                magconf=magconf, strgrid=strgrid,
                channels=v['channels'])

        v['geometry']['kbolo'] = kbolott  # in m^3
        v['geometry']['vbolo'] = volume  # in m^3
        v['radius']['reff'] = reff  # in m
        v['radius']['rho'] = rho  # in in units of above

    else:
        print('\t\t\\\ failed loading geometry, wrong keyword set')

    if saving:
        with open('../files/geom/geometry_combined.json', 'w') as outfile:
            outdict = mClass.dict_transf(v, to_list=True)
            json.dump(outdict, outfile, indent=4, sort_keys=False)
        outfile.close()

    return (v)


def get_reff_kbolott_volume_self(
        base_location='../results/INVERSION/MFR/',
        magconf='EIM_beta000',
        strgrid='tN4_50x30x100_1.4',
        channels={'none': None},
        debug=False):
    name = base_location + strgrid + '/' + magconf

    cams = ['HBCm', 'VBCl', 'VBCr']
    reff_LoS = np.zeros((128))  # in m
    rho_LoS = np.zeros((128))  # in m

    kbolott = np.zeros((128))  # in m^3
    volume = np.zeros((128))  # in m^3

    print(os.getcwd())

    for cam in cams:
        eCh, gCh = \
            channels['eChannels'][cam], \
            channels['gChannels'][cam]

        data_R = np.loadtxt(name + '_cam' + cam + '_reffLoS.dat')
        data_k = np.loadtxt(
            name + '_3D_cam' + cam + '_kbolott_and_volume_PV_pih.dat')
        if debug:
            print(cam, eCh, gCh, np.shape(data_R))

        for i, ch in enumerate(gCh):
            reff_LoS[eCh[i]] = data_R[ch - 1, 1]  # in m
            rho_LoS[eCh[i]] = data_R[ch - 1, 2]  # in units of reff

            kbolott[eCh[i]] = data_k[ch - 1, 0]  # in m^3
            volume[eCh[i]] = data_k[ch - 1, 1]  # in m^3
    return (reff_LoS, rho_LoS, kbolott, volume)


def link_list(
        base_URI='http://archive-webapi.ipp-hgw.mpg.de/'):
    """ Returns links and reqs for data/parlogs
    Args:
        base_URI (str, optional): Bare Uniform Resource Identifier
    Returns:
        URLs (list): List of links for data
        parlog_req (list): List of http requests for parlogs
    Notes:
        None
    """
    # URLS
    w7xanalysis = 'ArchiveDB/raw/W7XAnalysis/'
    adb_raw = "ArchiveDB/raw/W7X/"
    base_test = 'Test/raw/W7X/'
    test = "Test/raw/W7X/QSB_Bolometry/"
    test_analysis = "Test/raw/W7XAnalysis/"
    codac_raw = "ArchiveDB/codac/W7X/"
    miner_base = base_URI + "Test/raw/Minerva1/Minerva.Magnetics15."
    miner_suff = "_DATASTREAM/V1/0/"

    # names the variables get
    data_urls = [
        base_URI + test + s + '_DATASTREAM/' for s in
        ["BoloSignal", "MKappa", "MRes", "MTau", "RKappa", "RRes", "RTau"]]
    debug_urls = [
        base_URI + test + s + '_DATASTREAM/' for s in
        ["BoloSingleChannelFeedback", "BoloRealTime_P_rad", "Bolo_HBCmPrad",
         "Bolo_VBCPrad", "BoloPowerRaw", "BoloAdjusted"]]
    FoilCurrFits = [
        base_URI + test + s + '_DATASTREAM/' for s in
        ['BoloCalibMeasFoilCurrent', 'BoloCalibRefFoilCurrent',
         'BoloCalibMeasFoilFit', 'BoloCalibRefFoilFit']]

    # daihongs P_rad from HBC on BoloTest4
    hbc_prad = base_URI + 'ArchiveDB/raw/W7XAnalysis/QSB-Bolometry/' + \
        'Prad_HBC_DATASTREAM/V1/'
    vbc_prad = base_URI + 'ArchiveDB/raw/W7XAnalysis/QSB-Bolometry/' + \
        'Prad_VBC_DATASTREAM/V1/'

    # ECRH
    ecrh_total = base_URI + codac_raw + \
        "CBG_ECRH/TotalPower_DATASTREAM/V1/0/Ptot_ECRH/scaled/"

    # QME electron dens/temperature
    el_dens = base_URI + codac_raw + "CoDaStationDesc.16339/Data" + \
        "ModuleDesc.16341_DATASTREAM/0/Line integrated density/"

    # ECE Minerva new
    el_temp_core = base_URI + 'ArchiveDB/raw/Minerva/' + \
        'Minerva.ECE.DownsampledRadiationTemperatureTimetraces/' + \
        'signal_DATASTREAM/V4/12/QME-ch13/'
    el_temp_out = base_URI + 'ArchiveDB/raw/Minerva/' + \
        'Minerva.ECE.DownsampledRadiationTemperatureTimetraces/' + \
        'signal_DATASTREAM/V4/23/QME-ch24/'

    # QTB thomson eletron temperature/dens
    el_dens_map = base_URI + \
        'Test/raw/W7X/QTB_Profile/volume_2_DATASTREAM/V6/1/ne_map/'
    el_temp_map = base_URI + \
        'Test/raw/W7X/QTB_Profile/volume_2_DATASTREAM/V6/0/Te_map/'

    # Wdia
    dia_energy_v1 = miner_base + \
        "Wdia/Wdia_compensated_QXD31CE001x" + miner_suff

    # QSQ feedback signals
    EKS1_bolo1 = base_URI + adb_raw + 'QSQ_Hebeam/' + \
        'Feedback_signals_EKS1_DATASTREAM/V1/3/Bolometer%201/'
    EKS1_bolo2 = base_URI + adb_raw + 'QSQ_Hebeam/' + \
        'Feedback_signals_EKS1_DATASTREAM/V1/4/Bolometer%202/'
    feedback_Params = base_URI + adb_raw + 'QSQ_Hebeam/' + \
        'Feedback_coefficients_AEH51_DATASTREAM/V1/0/' + \
        'AEH51%20control%20parameter/'

    # DCH main gas inlet
    gasvalve_H2 = base_URI + adb_raw + \
        'CoDaStationDesc.10/DataModuleDesc.10' + \
        '_DATASTREAM/22/Actual Value Flow Valve BG011/'
    gasvalve_He = base_URI + adb_raw + \
        'CoDaStationDesc.10/DataModuleDesc.10' + \
        '_DATASTREAM/23/Actual Value Flow Valve BG031/'

    # QRT divertor totl loads
    [QRT1, QRT2] = [base_URI + w7xanalysis + 'QRT_IRCAM/AEF',
                    '_loads_DATASTREAM/V3/0/divertor_total_load/']

    divertor_targets = [
        AEF10, AEF11, AEF20, AEF21, AEF30, AEF31, AEF40, AEF41,
        AEF50, AEF51] = [
        QRT1 + str(x) + QRT2 for x in [
            10, 11, 20, 21, 30, 31, 40, 41, 50, 51]]

    # QSQ valve voltages
    [QSQ1, QSQ2, QSQ3] = ['QSQ_Hebeam/Feedback_coefficients_AEH',
                          '_DATASTREAM/V1/4/AEH', '%20valve%20voltage/']
    [AEH51, AEH30] = [base_URI + adb_raw + QSQ1 + x + QSQ2 + x + QSQ3
                      for x in ['51', '30']]

    # final P_rad from the analysis tab
    [prad_hbc, prad_vbc] = [
        test_analysis + 'QSB_Bolometry/' + a + '_DATASTREAM/V4/0/' + a + '/'
        for a in ['PradHBCm', 'PradVBC']]

    # prepare all the names
    other_urls = [ecrh_total, el_temp_core, el_temp_out, el_dens,
                  dia_energy_v1, el_dens_map, el_temp_map, hbc_prad, vbc_prad]
    feedback_urls = [EKS1_bolo1, EKS1_bolo2, feedback_Params]
    gasvalve_urls = [gasvalve_H2, gasvalve_He]
    hebeam_urls = [AEH51, AEH30]
    parlog_urls = [base_URI + test + 'BoloSignal_PARLOG/']
    prad_urls = [base_URI + prad_hbc, base_URI + prad_vbc]
    feedback_parlog_urls = [base_URI + test + s + '_PARLOG/' for s in [
        'BoloSingleChannelFeedback', 'BoloRealTime_P_rad']]

    # channel selection parlog
    selection_urls = [base_URI + test + s for s in [
        'BoloChannelSelection_PARLOG/', 'ChannelSelection_DATASTREAM/',
        'ChannelSelection_PARLOG/']]

    # QSQ PID settings
    pid_urls = []
    [QSQ1, QSQ2, QSQ3] = [
        'QSQ_Hebeam/Feedback_coefficients_AEH', '_DATASTREAM/V1/', '/AEH']
    for x in ['30', '51']:
        for i, coeff in enumerate([
                ' control parameter/', ' Kp/', ' Ki/', ' Kd/']):
            link = base_URI + adb_raw + QSQ1 + x + QSQ2 + \
                str(i) + QSQ3 + x + coeff.replace(' ', '%20')
            pid_urls.append(link)

    # filterscopes of C III
    c3_tubes = [5, 19, 24, 29, 36, 41, 53, 57]  # 8
    # tubes of C II, C IV
    c2_tubes, c4_tubes = [35], [20]  # 1, 1
    # H-alpha
    hA_tubes = [4, 8, 11, 13, 21, 28, 32,  # 14
                37, 45, 56, 60, 61, 65, 69]

    QSR02_a, QSR02_b, QSR02_c = 'Filterscopetest4/PMT', \
        '_DATASTREAM/V1/1/PMT', '_PhotonFlux/'
    c3_urls = [base_URI + base_test + QSR02_a + str(tube).zfill(3) + QSR02_b +
               str(tube).zfill(3) + QSR02_c for tube in c3_tubes]
    c2_urls = [base_URI + base_test + QSR02_a + str(tube).zfill(3) + QSR02_b +
               str(tube).zfill(3) + QSR02_c for tube in c2_tubes]
    c4_urls = [base_URI + base_test + QSR02_a + str(tube).zfill(3) + QSR02_b +
               str(tube).zfill(3) + QSR02_c for tube in c4_tubes]
    hA_urls = [base_URI + base_test + QSR02_a + str(tube).zfill(3) + QSR02_b +
               str(tube).zfill(3) + QSR02_c for tube in hA_tubes]

    # full links to compare to labels
    data_req = []
    for i, L in enumerate([
            data_urls, other_urls, debug_urls, feedback_urls,
            FoilCurrFits, gasvalve_urls, divertor_targets, hebeam_urls,
            parlog_urls, prad_urls, feedback_parlog_urls, pid_urls,
            selection_urls, c3_urls, c2_urls, c4_urls, hA_urls]):
        for j, n in enumerate(L):
            data_req.append(n)

    # labels to sort through and return link
    data_labels = [
        'BoloSignal', 'MKappa', 'MRes', 'MTau', 'RKappa', 'RRes', 'RTau',  # 0-6
        'ECRH', 'T_e ECE core', 'T_e ECE out', 'n_e lint',  # 7-10
        'W_dia', 'n_e QTB vol2', 'T_e QTB vol2', 'Prad HBC',  # 11-14
        'Prad VBC', 'BoloSingleChannelFeedback', 'BoloRealTime_P_rad',  # 15-17
        'Bolo_HBCmPrad', 'Bolo_VBCPrad', 'BoloPowerRaw',  # 18-20
        'BoloAdjusted', 'EKS1 Bolo1', 'EKS1 Bolo2', 'QSQ Params',  # 21-24
        'BoloCalibMeasFoilCurrent', 'BoloCalibRefFoilCurrent',  # 25-26
        'BoloCalibMeasFoilFit', 'BoloCalibRefFoilFit',  # 27-28
        'Main valve BG011', 'Main valve BG031', 'DivAEF10',  # 29-31
        'DivAEF11', 'DivAEF20', 'DivAEF21', 'DivAEF30',  # 32-35
        'DivAEF31', 'DivAEF40', 'DivAEF41', 'DivAEF50', 'DivAEF51',  # 36-40
        'QSQ Feedback AEH51', 'QSQ Feedback AEH30',  # 41-42
        'BoloSignal PARLOG', 'PradHBCm', 'PradVBC',  # 43-45
        'SingleChannel PARLOG', 'RealTime PARLOG',  # 46 - 47
        'AEH30 control', 'AEH30 Kp', 'AEH30 Ki', 'AEH30 Kd',  # 48 - 51
        'AEH51 control', 'AEH51 Kp', 'AEH51 Ki', 'AEH51 Kd',  # 52 - 55
        'BoloSelection Channels', 'Channel Selection Data',  # 56 - 57
        'Channel Selection Params', 'CIII 5', 'CIII 19', 'CIII 24',  # 58 - 61
        'CIII 29', 'CIII 36', 'CIII 41', 'CIII 53', 'CIII 57',  # 62 - 66
        'CII 35', 'CIV 20', 'HAlpha 4', 'HAlpha 8', 'HAlpha 11',  # 67 - 71
        'HAlpha 13', 'HAlpha 21', 'HAlpha 28', 'HAlpha 32',  # 72 - 75
        'HAlpha 37', 'HAlpha 45', 'HAlpha 56', 'HAlpha 60',  # 76 - 79
        'HAlpha 61', 'HAlpha 65', 'HAlpha 69']  # 80 - 82
    return data_req, data_labels


def compare(
        printing=False,
        base='../results/',
        indent_level='\t'):
    """ Queue for above fcnt.
    Args:
        base: Base directory to look for.
        indent_level: Indentation level.
    Returns:
        compare_shots: XP IDs to compare at.
        compare_data_names: File names and locations to compare.
        comparison: Bool list of what and how to compare stuff.
    Notes:
        None.
    """

    if printing:
        print(indent_level +
              '>> Comparison info for shots ...')
    compare_shots, compare_data_names, comparison = \
        introduce_comparison_data(base, indent_level='\t\t')

    return compare_shots, compare_data_names, comparison


def introduce_comparison_data(
        base,
        indent_level):
    """ Sets up comparison list for file locations, names, XP IDs and dates.
    Args:
        base: Base directory to look for.
        indent_level: Indentation level.
    Returns :
        compare_shots: XP IDs to compare at.
        compare_data_names: File names and locations to compare.
        comparison: Bool list of what and how to compare stuff.
    Notes:
        None.
    """
    compare_shots = [
        "20171115.039", "20171122.014", "20171207.010", "20171109.045"]

    compare_data_names = \
        [["20171115/vbc_powch_20171115039.dat",
          "20171115/vbc_udch_20171115039.dat",
          "20171115/hbc_powch_20171115039.dat",
          "20171115/hbc_udch_20171115039.dat",
          "20171115/prad_vbc_20171115039.dat",
          "20171115/prad_hbc_20171115039.dat"
          ],
         ["20171122/vbc_powch_20171122014.dat",
          "20171122/vbc_udch_20171122014.dat",
          "20171122/hbc_powch_20171122014.dat",
          "20171122/hbc_udch_20171122014.dat",
          "20171122/prad_vbc_20171122014.dat",
          "20171122/prad_hbc_20171122014.dat"
          ],
         ["20171207/vbc_powch_20171207010.dat",
          "20171207/vbc_udch_20171207010.dat",
          "20171207/hbc_powch_20171207010.dat",
          "20171207/hbc_udch_20171207010.dat",
          "20171207/prad_vbc_20171207010.dat",
          "20171207/prad_hbc_20171207010.dat"
          ],
         [None, None, "20171109/hbc_powch_20171109045.dat",
          None, None, "20171109/prad_hbc_20171109045.dat"
          ]]

    start_str = '../results/COMPARISON/CROSSCHECK/'
    [i, j] = [0, 0]
    for j in range(0, len(compare_data_names)):
        for i in range(0, len(compare_data_names[0])):
            if isinstance(compare_data_names[j][i], str):
                compare_data_names[j][i] = start_str + compare_data_names[j][i]

    comparison = \
        [[[True, True, True],      # 20171115.039 vbc
          [True, True, True],      # 20171115.039 hbc
          [True, True]             # 20171115.039 prads
          ],
         [[True, True, True],      # 20171122.014 vbc
          [True, True, True],      # 20171122.014 hbc
          [True, True]             # 20171122.014 prads
          ],
         [[True, True, True],      # 20171207.010 vbc
          [True, True, True],      # 20171207.010 hbc
          [True, True]             # 20171207.010 prads
          ],
         [[False, False, False],   # 20171207.010 vbc
          [False, False, True],    # 20171207.010 hbc
          [False, True]            # 20171207.010 prads
          ]]

    return compare_shots, compare_data_names, comparison
