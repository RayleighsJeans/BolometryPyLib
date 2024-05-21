""" **************************************************************************
    so header """

import os
import warnings
import numpy as np
from scipy.optimize import curve_fit

import mClass
import dat_lists as dat_lists
import webapi_access as api

import magic_function as magic
import fit_procedures as fitting
import auxilary_funcs_fix as aux


warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")

# backup for later to fix channels in range of dates
# flpd_chan = [10, 16, 18, 21, 24, 25]
# if (int(date) >= 20180828) and (int(date) <= 20180905):
# channel_volt = aux.aux_voidTest(
#     channel_volt, v, data_object[0]['dimensions'],
#     priority_object[36], priority_object[37], True,
#     indent_level + '\t', date, shotno)
# if (int(date) >= 20180828) and (int(date) <= 20180905):
#     print(indent_level + '\t>> Skipping flipped channels from failed DAQ')
#     brk_chan.append(flpd_chan)

Z = np.zeros
ones = np.ones
mn = np.mean
M, N = 10000, 100000

""" eo header
************************************************************************** """


def calculation(
        prio={'none': {'none': None}},
        dat={'none': {'none': None}},
        date='20181010',
        shotno=32,
        program_info={'none': None},
        make_linoffs=False,
        filter_method='raw',
        printing=False,
        indent_level='\t'):

    N = len(dat['BoloSignal']['dimensions'])  # sample amount
    v_adjusted = Z((128, N))  # final full power array
    power = Z((128, N))  # final full power array
    volscaled = Z((128, N))  # final full power array
    stddiv = Z((128))  # raw chan sig
    power_stddiv = Z((128))
    a = Z((128))  # lin offs fit results
    b = Z((128))  # ...
    spikes = 0  # No of spikes found in quick fix
    # for linear trickery later
    n_raw = dat['BoloSignal']['dimensions'][prio['t_it2']:-1]  # in ns
    matrix = np.vstack([n_raw, np.ones(len(n_raw))]).T
    # broken channels
    brk_chan = prio['geometry']['channels']['droplist']

    # in kappa [A^2], R [Ohm], tau [s], Imax [A]
    status, kappam, rohm, taum, kappar, rohr, taur, fit_results, Imax = \
        fitting_wrapper(dat=dat, shotno=shotno, date=date, printing=printing)

    if (20180828 <= int(date) <= 20180905):
        dat['BoloSignal']['values'] = aux.clicky_thingy(
            dat['BoloSignal'], date=date, shotno=str(shotno).zfill(3))

    if printing:
        print(indent_level + '>> power and voltage:', end=' ')
    for ch in range(0, 128):
        if ((ch in brk_chan) or
            ((int(date) in [20180905]) and
             ((int(shotno) in [13]) and (ch in [-1]))) or
            ((int(date) in [20180904]) and
             ((int(shotno) in [23]) and (ch in [33]))) or
            ((int(date) in [20180905]) and
             (((int(shotno) in [14, 15]) and (ch in [16])) or
              ((int(shotno) in [16, 17]) and (ch in [16, 21])) or
              ((int(shotno) in [21]) and (ch in [18, 21])) or
              ((int(shotno) in [24]) and (ch in [4, 7, 21, 24])))) or
            ((int(date) in [20180904]) and
             ((int(shotno) in [23]) and (ch in [29])) or
             ((int(shotno) in [25]) and (ch in [16, 18, 21])))):
            continue  # skip broken uncorrectable

        if printing:
            print(ch, end='... ')
            if ((ch != 0) and (ch % 11 == 0.0)):
                print('\n' + indent_level + '\t', end=' ')

        # in Volts, Watts, Volts, Watts, Watts, V/s, Volts
        v_adjusted[ch], power[ch], SN, stddiv[ch], power_stddiv[ch], \
            volscaled[ch], a[ch], b[ch] = major_function(
            voltage=dat['BoloSignal']['values'][ch],  # in V
            kappa_ch=kappam[ch],  # in A^2
            r_ch=rohm[ch],  # in Ohm
            tau_ch=taum[ch],  # in s
            t_it=prio['t_it'], t_it2=prio['t_it2'],
            U0=prio['U0'],  # in V
            rc=prio['RC'],  # in Ohm
            f_bridge=prio['f_bridge'],  # in Hz
            c_cab=prio['C_cab'],  # in F
            dt=prio['dt'],  # in s
            n_raw=n_raw,
            f_tran=prio['f_tran'][ch],
            volume=prio['geometry']['geometry']['vbolo'][ch],  # in m^3
            k_bolo=prio['geometry']['geometry']['kbolo'][ch],  # in m^3
            ch=ch, M=matrix, make_linoffs=make_linoffs,
            filter_method=filter_method)
        spikes += SN

    if printing:
        print('... done!')
        print(indent_level + ">> No. of caching glitches spikes found:", spikes)

        print(indent_level + '\t>> Calculation of P_rad\'s:', end=' ')

    save_voltage_dat(
        time=dat['BoloSignal']['dimensions'], voltage=v_adjusted,
        date=int(date), shotno=int(shotno), method=filter_method,
        indent_level=indent_level, debug=printing)

    P_rad_hbc, volume_sum_hbc = calculate_prad(
        time=dat['BoloSignal']['dimensions'],  # ns
        power=power,  # Watts
        volume=prio['geometry']['geometry']['vbolo'],  # m^3
        k_bolo=prio['geometry']['geometry']['kbolo'],  # m^3
        volume_torus=prio['volume_torus'],  # m^3
        channels=prio['geometry']['channels']['eChannels']['HBCm'],
        camera_list=prio['geometry']['channels']['eChannels']['HBCm'],
        date=int(date), shotno=int(shotno), brk_chan=brk_chan,
        camera_tag='HBCm', saving=True, method=filter_method, debug=printing)

    if printing:
        print(indent_level + '\t                            ', end=' ')
    P_rad_vbc, volume_sum_vbc = calculate_prad(
        time=dat['BoloSignal']['dimensions'],  # ns
        power=power,  # Watts
        volume=prio['geometry']['geometry']['vbolo'],  # m^3
        k_bolo=prio['geometry']['geometry']['kbolo'],  # m^3
        volume_torus=prio['volume_torus'],  # m^3
        channels=prio['geometry']['channels']['eChannels']['VBC'],
        camera_list=prio['geometry']['channels']['eChannels']['VBC'],
        date=int(date), shotno=int(shotno), brk_chan=brk_chan,
        camera_tag='VBC', saving=True, method=filter_method, debug=printing)

    return ({
        'status': status,
        'time': [(t - program_info["programs"][0]["trigger"]["1"][0]) / 1e9
                 for t in dat['BoloSignal']['dimensions']],  # in s
        'voltage adjusted': v_adjusted,  # in V
        'stddiv': stddiv,  # in V
        'power': power,  # in W
        'power_stddiv': power_stddiv,  # in W
        'P_rad_hbc': P_rad_hbc,  # in W
        'volume_sum_hbc': volume_sum_hbc,  # m^3
        'P_rad_vbc': P_rad_vbc,  # in W
        'volume_sum_vbc': volume_sum_vbc,  # in m^3
        'kappam': kappam,  # in A^2
        'rohm': rohm,  # in Ohm
        'taum': taum,  # in s
        'a': a,  # in V/s
        'b': b,  # in V
        'kappar': kappar,  # in A^2
        'rohr': rohr,  # in Ohm
        'taur': taur,  # in s
        'fit_results': fit_results,
        'Imax': Imax,  # in A
        'volscaled': volscaled})  # in W / ???


def save_voltage_dat(
        time=np.zeros((10000)),
        voltage=np.zeros((128, 10000)),
        date=20181010,
        shotno=32,
        method='raw',
        indent_level='\t',
        debug=False):

    dat = np.zeros((1 + np.shape(voltage)[0], np.shape(voltage)[1]))
    # concatenate dat object
    dat[0], dat[1:] = time, voltage

    location = r"../results/PRAD/" + str(date)
    if not os.path.exists(location):
        os.makedirs(location)

    # saving in text two colums with time on first
    # vbc and hbc data in kW in second column
    datafile = '../results/PRAD/' + str(date) + '/' + \
        str(date) + '.' + str(shotno).zfill(3) + \
        "_voltage_adjusted_" + method + ".dat"
    if debug:
        print(indent_level + '\t\\\ saving voltage')
    np.savetxt(datafile, dat)
    return


def calculate_prad(
        time=Z((10000)),  # in ns
        power=Z((128, 10000)),  # Watts
        volume=Z((128)),  # in m^3
        k_bolo=Z((128)),  # in m^3
        volume_torus=45.,  # m^3
        channels=[0, 1, 2, 3, 4, 5],
        date=20181010,
        shotno=32,
        brk_chan=[6, 20, 32, 87],
        saving=False,
        camera_tag='HBCm',
        camera_list=[int(v) for v in np.linspace(0, 31, 32)],
        method='raw',
        debug=False,
        indent_level='\t'):
    """ Mother of all functions, calculates a proper voltage signal
        and hence the power of each channel, the standard deviation,
        errors and so much more (drifts, offsets and so on).
    Args:
        priority_object (0, list): Object precalculated variables
        data_object (1, list): Downloaded data from the archive
        date (2, str): Current date
        shotno (3, int): Current experiment ID
        make_linoffs (4, bool): linear drift be fitted and subtraction
        indent_level (5, str): Indentation Level
    Returns:
        prad_object (0, list): Concatenated list of all
            the power properties and results.
    Notes:
        None.
    """
    # set up stuff for all the needed vars/space
    P_rad = Z((len(power[0])), dtype=float)  # final prad
    volume_sum = 0.0  # cam vol

    for i, ch in enumerate(channels):
        if ((ch in brk_chan) or
            ((date in [20180905]) and
             ((shotno in [13]) and (ch in [-1]))) or
            ((date in [20180904]) and
             ((shotno in [23]) and (ch in [33]))) or
            ((date in [20180905]) and
             (((shotno in [14, 15]) and (ch in [16])) or
              ((shotno in [16, 17]) and (ch in [16, 21])) or
              ((shotno in [21]) and (ch in [18, 21])) or
              ((shotno in [24]) and (ch in [4, 7, 21, 24])))) or
            ((date in [20180904]) and
             ((shotno in [23]) and (ch in [29])) or
             ((shotno in [25]) and (ch in [16, 18, 21])))):
            continue  # skip broken uncorrectable

        if debug:
            print(ch, "...", end=' ')
        if ((i != 0) and (i % 5 == 0.0)):
            if debug:
                print('\n' + indent_level + '\t', end=' ')

        P_rad += volume[ch] * power[ch] / \
            k_bolo[ch] if k_bolo[ch] != 0.0 else 0.0
        # m^3 * Watts / m^3 -> Watts

    for i, ch in enumerate(camera_list):
        if ch in brk_chan:
            continue
        if debug:
            print(ch, "...", end=' ')
        volume_sum += volume[ch]  # in m^3

    if saving:
        location = r"../results/PRAD/" + str(date)
        if not os.path.exists(location):
            os.makedirs(location)

        # saving in text two colums with time on first
        # vbc and hbc data in kW in second column
        datafile = '../results/PRAD/' + str(date) + '/' + \
            str(date) + '.' + str(shotno).zfill(3) + "prad_" + camera_tag + \
            '_' + method + ".dat"
        with open(datafile, 'w') as file:
            for k in range(len(P_rad)):
                file.write('{}\t{}\n'.format(time[k], P_rad[k] / 1e3))
        file.close()

    if debug:
        print('done!')
    # Watts / m^3 * m^3 -> Watts
    return (P_rad / volume_sum * volume_torus, volume_sum)


def lin_func(
        x=1.0,
        a=1.0,
        b=1.0):
    """ Template for exponential function used in fitting routine.
    Args:
        x (0, npfloat): Value put in.
        a (1, npfloat): Leading factor.
        b (2, npfloat): Exponential factor.
    Returns:
        y (0, npfloat): Result of function.
    Notes:
        None.
    """
    return a * x + b


def major_function(
        date='20181010',
        voltage=Z(M),
        kappa_ch=8e-4,  # in A^2
        r_ch=1e3,  # in Ohm
        tau_ch=1e-1,  # in s
        t_it=1000,
        t_it2=9000,
        U0=5.0,  # in V
        rc=1.e3,  # in Ohm
        f_bridge=2500.,  # in Hz
        c_cab=5.e-9,  # in F
        dt=1.6e-3,  # in s
        n_raw=Z(M),
        f_tran=0.53,
        volume=1.e-6,  # in m^3
        k_bolo=0.4,  # in m^3
        M=np.vstack([Z(M), np.ones(len(Z(M)))]).T,
        ch=0,
        debug=False,
        make_linoffs=False,
        filter_method='raw'):

    # voltage and std div
    raw_stddiv = np.std(voltage[-101:-1], dtype=np.float64)  # in V
    # correcting voltage by offset, polarity
    offset = np.mean(voltage[-101:-1])  # in V

    if debug:
        print(' offset fix ', end='')
    # fix offset taken when DAQ time is wrong and in discharge
    if (abs(offset) >= 5e-6):
        offset = np.mean(voltage[t_it2 - 200:t_it2 - 100])  # in V
    voltage = voltage - offset  # V

    if debug:
        print(' flip fix ', end='')
    if (np.mean(voltage[t_it:t_it + 200]) < 0):
        voltage = (-1) * voltage  # in V

    if (make_linoffs):  # and \
        # (abs(np.mean(voltage[-101:-1])) <= 5e-1):
        if debug:
            print(' linear fix ', end='')

        # offset drift correction
        try:
            ydata = np.linspace(  # in V
                np.mean(voltage[t_it2 - 50:t_it2]),
                np.mean(voltage[-51:-1]), len(n_raw))
            [a, b] = curve_fit(  # V / s, V
                lin_func, [(t - n_raw[0]) / 1e9 for t in n_raw], ydata,
                p0=(0.0, 0.0), sigma=None, absolute_sigma=None,
                check_finite=True, method='lm')[0]

        except Exception:
            a, b = 0.0, 0.0

        # if we are supposed to make the offset is right
        # adjusted in V
        voltage[t_it2:-1] = voltage[t_it2:-1] - \
            ([a * x + b for x in [(t - n_raw[0]) / 1e9 for t in n_raw]])

    else:
        a, b = 0.0, 0.0

    # magic function where pow_chn is calculated
    if debug:
        print(' magic func ', end='')
    power, SN = magic.magic_function(  # Watts
        voltage=voltage, dt=dt, U0=U0, kappa_ch=kappa_ch,  # V, s, V, A^2
        r_ch=r_ch, tau_ch=tau_ch,  # Ohm, s
        rc=rc, f_bridge=f_bridge, c_cab=c_cab,  # Ohm, Hz, F
        filter_method=filter_method)

    # same offset fix as above but for pow
    if debug:
        print(' power offset ', end='')
    power_offset = np.mean(power[-101:-1])
    if (power_offset >= 5e-7):
        power_offset = np.mean(power[t_it2 - 200:t_it2 - 100])

    # calc prad from volumes, geom and with offs
    if debug:
        print(' power scaling ', end='')
    power = (power - power_offset) / f_tran  # Watts
    power_stddiv = np.abs(np.std(power[-101:-1]))  # Watts

    volscaled = power / k_bolo if volume > 1.e-5 else 0.0  # in Watts/m^3

    return (voltage,  # V
            power,  # Wats
            SN,
            raw_stddiv,  # V
            power_stddiv,  # Watts
            volscaled,  # Watts
            a, b)  # V/s, V


def fitting_wrapper(
        dat={'none': {'none': None}},
        shotno=32,
        date='20181010',
        printing=False,
        indent_level='\t',
        kappar=np.zeros((128)),
        rohr=np.zeros((128)),
        taur=np.zeros((128)),
        kappam=np.zeros((128)),
        rohm=np.zeros((128)),
        taum=np.zeros((128)),
        fit_results=[np.zeros((128, 2)), np.zeros((128, 2))],
        Imax=[np.zeros((128)), np.zeros((128))]):

    status = 'valid'
    # fitting mode to find best parameters
    [kappam, rohm, taum, kappar, rohr, taur, fit_results, Imax] = fitting.fit_parameters(  # A^2, Ohm, s, ...
        indent_level=indent_level + '\t', dat=dat,
        shotno=shotno, date=date)

    trgts = [np.mean(kappam[0:85]), np.mean(kappar[0:85]),
             np.mean(taum[0:85]), np.mean(taur[0:85]),
             np.mean(rohm[0:85]), np.mean(rohr[0:85])]

    if printing:
        print(indent_level + '\t>> Mean fitted calibration values are:\n',
              indent_level + '\t\tkappa_m=', format(trgts[0], '.3e'), 'A^2;',
              'kappa_r=', format(trgts[1], '.3e'),
              'A^2\n', indent_level + '\t\ttau_m=',
              format(trgts[2] * 1e3, '.2f'), 'ms;',
              'tau_r=', format(trgts[3] * 1e3, '.2f'), 'ms;\n',
              indent_level + '\t\troh_m=', format(trgts[4], '.2f'), 'Ohm;',
              'roh_r=', format(trgts[5], '.2f'), 'Ohm;')

    # catch failed fitting routine with questionaire
    if ([len(kappam), len(rohm), len(taum)] != [128, 128, 128]) or not \
            (3e-4 < trgts[0] < 9e-4) or not (  # A^2
                100e-3 < trgts[2] < 130e-3) or not (  # s
                    .6e3 < trgts[4] < 1.3e3):  # Ohm
        # incomplete or out of range
        status = 'invalid'
        ans = mClass.query_yes_no(
            indent_level + '\t>> Calibration values failed,' +
            ' use mean from past XPIDs?', "yes")

        if ans:  # use fixed
            status = 'mean'
            if printing:
                print(indent_level + '\t>> Use mean calibration values from ' +
                      'past two XPIDs, status: ', status)

            # downloading and combining calibration values from past two XPIDs
            kappam, rohm, taum, kappar, rohr, taur = mean_fit_parameters(debug=False, programs=[
                date + '.' + str(shotno - 1).zfill(3),
                date + '.' + str(shotno - 2).zfill(3)])
            print(indent_level + '\t>> Mean downloaded calibration' +
                  ' values are\n', indent_level + '\t\tkappa_m=',
                  format(np.mean(kappam[:85]), '.3e'), 'A^2;', 'tau_m=',
                  format(np.mean(taum[:85]) * 1e3, '.2f'), 'ms;', 'roh_m=',
                  format(np.mean(rohm[:85]), '.2f'), 'Ohm')

        elif not ans:  # nope, look in the archive
            status = 'constant Data'
            if printing:
                print(indent_level + '\t>> Use archive saved calibration ' +
                      'values, status: ', status)
            [kappam, rohm, taum] = [[8e-4] * 128, [1.e3] * 128, [1e-1] * 128]
            [kappar, rohr, taur] = [[8e-4] * 128, [1.e3] * 128, [1e-1] * 128]
            # in A^2, Ohm and s

    return (status, kappam, rohm, taum, kappar, rohr, taur, fit_results, Imax)


def mean_fit_parameters(
        programs=['20181010.032', '20181010.033'],
        kappar=np.zeros((128)),
        rohr=np.zeros((128)),
        taur=np.zeros((128)),
        kappam=np.zeros((128)),
        rohm=np.zeros((128)),
        taum=np.zeros((128)),
        debug=False):

    if debug:
        print(programs)
    D = [kappam, rohm, taum, kappar, rohr, taur]
    data_labels = dat_lists.link_list()[1]
    tags = [data_labels[k] for k in (1, 2, 3, 4, 5, 6)]
    for x, program in enumerate(programs):
        # links and labels
        program_info = api.xpid_info(program=program)[0]
        start = str(program_info['programs'][0]['from'])
        stop = str(program_info['programs'][0]['upto'])

        for i, tag in enumerate(tags):
            D[i] = api.download_single(
                api.download_link(name=tag), program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)['values']

    if debug:
        print(np.shape(D), np.shape(D[0]), D[0])
    for q, f in enumerate(D):
        D[q] = [v[0] / np.shape(programs)[0] for v in f]

    return (D[0], D[1], D[2], D[3], D[4], D[5])
