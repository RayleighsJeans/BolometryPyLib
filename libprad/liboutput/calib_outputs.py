""" **************************************************************************
    so header """

import os
import numpy as np
import requests
import webapi_access as webapi

import matplotlib.pyplot as p
import matplotlib
import sys
import warnings
import plot_funcs as pf

matplotlib.use('Qt5agg')
warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")
NoneType = type(None)

""" eo header
************************************************************************** """


def output(
        prio={'none': {'none': None}},
        dat={'none': {'none': None}},
        power={'none': {'none': None}},
        program='20181010.032',
        program_info={'none': {'none': None}},
        indent_level='\t'):
    """ Queue of calibration value outputs
    Args:
        prio: Pre-defined data object/list with constants
        dat: List object with all archive date loaded
        power: List object of calculated radiation properties
        program: Combined date and XP ID
        program_info: HTTP return of logbook concerning the program
        indent_level: Indentation level
    Returns:
        None
    """
    NChannels = 98
    [minc, maxc] = [-0.5, NChannels + .5]
    channels = np.linspace(0, NChannels, NChannels + 1)

    print(indent_level + '>> calibration values plot ...')
    indent_level = indent_level + '\ngt'
    # import comparison calibration values from archive
    # ideally made by old version

    std_program = '20171207.020'
    std_vals = standard_values(std_program)
    [stdKap, stdTau, stdRes] = \
        [std_vals[0]['values'], std_vals[1]['values'], std_vals[2]['values']]
    fig, (kappa, tau, res, fit) = p.subplots(nrows=4, sharex=True)
    for ax in [kappa, tau, res, fit]:
        ax.grid(b=True, which='major', linestyle='-.')

    kappa_out(
        std_program=std_program, comp_Kap=stdKap,
        NChannels=NChannels, minc=minc, maxc=maxc, channels=channels,
        fig=fig, kappa=kappa, tau=tau, res=res, fit=fit,
        prio=prio, dat=dat, power=power, program=program)
    tau_out(
        comp_Tau=stdTau,
        NChannels=NChannels, minc=minc, maxc=maxc, channels=channels,
        fig=fig, kappa=kappa, tau=tau, res=res, fit=fit,
        prio=prio, dat=dat, power=power)
    res_out(
        comp_Res=stdRes,
        NChannels=NChannels, minc=minc, maxc=maxc, channels=channels,
        fig=fig, kappa=kappa, tau=tau, res=res, fit=fit,
        prio=prio, dat=dat, power=power)
    fit_results_out(
        NChannels=NChannels, minc=minc, maxc=maxc, channels=channels,
        fig=fig, kappa=kappa, tau=tau, res=res, fit=fit,
        prio=prio, dat=dat, power=power)

    fig.set_size_inches(7., 12.)
    pf.fig_current_save('calibs', fig)

    location = '../results/CALIBS/' + program[0:8]
    if not os.path.exists(location):
        os.makedirs(location)

    fig.savefig('../results/CALIBS/' + program[0:8] + '/calibration_values.' +
                program + '.png', bbox_inches='tight', dpi=169.0)
    p.close('all')

    return


def standard_values(
        program='20181010.032'):
    """ Gets the calibration values to compare to from the archive
    Args:
        program (str):
    Returns:
        data: List object including the std calib values
    """

    base_URI = 'http://archive-webapi.ipp-hgw.mpg.de/'
    req = requests.get(base_URI + 'programs.json' +
                       '?from=' + program).json()

    start = str(req['programs'][0]['from'])
    stop = str(req['programs'][0]['upto'])

    data_name = ['MKappa', 'MTau', 'MRes',
                 'BoloCalibMeasFoilCurrent', 'BoloCalibMeasFoilFit']
    # requests in json format and with time step
    test = "Test/raw/W7X/QSB_Bolometry/"
    links = [base_URI + test + n +
             "_DATASTREAM/_signal.json?from=" + start + "&upto=" + stop
             for n in data_name]

    data = []
    for i, L in enumerate(links):
        data.append(webapi.download_single(L, program_info=req))
    return data


def kappa_out(
        std_program='20181018.041',
        comp_Kap=[[], []],
        NChannels=98,
        minc=-0.5,
        maxc=98.5,
        channels=[],
        fig=None,
        kappa=None,
        tau=None,
        res=None,
        fit=None,
        prio={'none': {'none': None}},
        dat={'none': {'none': None}},
        power={'none': {'none': None}},
        program='20181010.032',
        indent_level='\t'):
    """ Plots the heat capacities of meas., ref. and std. foils
    Args:
        std_program: XP id
        comp_Kap: Standard head capacity
        NChannels: Plot width/channel number to show
        minc: Minimum channel number
        maxc: Maximum channel number
        channels: Channel vector
        fig: Figure canvas reference
        kappa: Subplot heat capacity ref
        tau: Subplot cooling time
        res: Subplot electrical resistance
        fit: Fit parameter subplot ref
        prio: Pre-defined list object with fixed values/constants
        dat: Data list object with everything from the archive
        power: Radiation power data object list from calc
        program (str): XP ID
        indent_level: Indentation level
    Returns:
        fig, kappa, tau, res, fit: Figure references
    """
    print(indent_level + 'kappas ...', end=' ')

    try:
        [kappaMeas, kappaRef, stdKap] = \
            [[0] * (NChannels + 1), [0] * (NChannels + 1),
             [0] * (NChannels + 1)]

        for ch in range(0, NChannels + 1):
            kappaMeas[ch] = power['kappam'][ch]
            kappaRef[ch] = power['kappar'][ch]
            stdKap[ch] = comp_Kap[ch][0]

        try:
            maxk = max(max(kappaMeas),
                       max(kappaRef), max(stdKap)) * 1.1
            mink = min(min(kappaMeas), min(kappaRef), min(stdKap))
            kappa.set_ylim(mink, maxk)

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\  kappa lim failed')

        kappa.plot(channels, [x for x in kappaMeas],
                   ':', marker='o', color='r',
                   label='$\\kappa$ (meas.)')
        kappa.plot(channels, [x for x in kappaRef],
                   '--', marker='^', color='b',
                   label='$\\kappa$ (ref.)')
        kappa.plot(channels, [x if x is not None else 0.0
                              for x in stdKap],
                   '-.', marker='*', color='orange',
                   label='$\\kappa$ (std.)')

        kappa.set_xlim(minc, maxc)
        kappa.set_title('shot #' + program)
        kappa.set_ylabel('norm. heat cap. [A$^{2}$]')
        kappa.legend()

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  kappa plot failed')

    return fig, kappa, tau, res, fit


def tau_out(
        std_program='20181018.041',
        comp_Tau=[[], []],
        NChannels=98,
        minc=-0.5,
        maxc=98.5,
        channels=[],
        fig=None,
        kappa=None,
        tau=None,
        res=None,
        fit=None,
        prio={'none': {'none': None}},
        dat={'none': {'none': None}},
        power={'none': {'none': None}},
        program='20181010.032',
        indent_level='\t'):
    """ Plots the cooling time of meas., ref. and std. foils
    Args:
        comp_Tau: Standard cooling time
        NChannels: Plot width/channel number to show
        minc: Minimum channel number
        maxc: Maximum channel number
        channels: Channel vector
        fig: Figure canvas reference
        kappa: Subplot heat capacity ref
        tau: Subplot cooling time
        res: Subplot electrical resistance
        fit: Fit parameter subplot ref
        prio: Pre-defined list object with fixed values/constants
        dat: Data list object with everything from the archive
        power: Radiation power data object list from calc
        program (str): XP ID
        indent_level: Indentation level
    Returns:
        fig, kappa, tau, res, fit: Figure references
    """
    print(indent_level + 'taus ...', end=' ')

    try:
        [tauMeas, tauRef, stdTau] = \
            [[0] * (NChannels + 1), [0] * (NChannels + 1),
             [0] * (NChannels + 1)]
        for ch in range(0, NChannels + 1):
            tauMeas[ch] = power['taum'][ch]
            tauRef[ch] = power['taur'][ch]
            stdTau[ch] = comp_Tau[ch][0]

        try:
            maxt = max(max(tauMeas), max(tauRef), max(stdTau)) * 1.05
            mint = min(min(tauMeas), min(tauRef), min(stdTau))
            tau.set_ylim(mint, maxt)

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\  tau lims failed')

        tau.plot(channels, [x for x in tauMeas],
                 ':', marker='o', color='r',
                 label='$\\tau$ (meas.)')
        tau.plot(channels, [x for x in tauRef],
                 '--', marker='^', color='b',
                 label='$\\tau$ (ref.)')
        tau.plot(channels, [x if x is not None else 0.0
                            for x in stdTau],
                 '-.', marker='*', color='orange',
                 label='$\\tau$ (std.)')

        tau.set_xlim(minc, maxc)
        # tau.set_title('meas. and ref. cooling time of foils')
        tau.set_ylabel('cooling time [s]')
        tau.legend()

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  tau plot failed')

    return fig, kappa, tau, res, fit


def res_out(
        comp_Res=[[], []],
        NChannels=98,
        minc=-0.5,
        maxc=98.5,
        channels=[],
        fig=None,
        kappa=None,
        tau=None,
        res=None,
        fit=None,
        prio={'none': {'none': None}},
        dat={'none': {'none': None}},
        power={'none': {'none': None}},
        program='20181010.032',
        indent_level='\t'):
    """ Plots the electrical resistance of meas., ref. and std. foils
    Args:
        comp_Res: Standard electrical resistance
        NChannels: Plot width/channel number to show
        minc: Minimum channel number
        maxc: Maximum channel number
        channels: Channel vector
        fig: Figure canvas reference
        kappa: Subplot heat capacity ref
        tau: Subplot cooling time
        res: Subplot electrical resistance
        fit: Fit parameter subplot ref
        prio: Pre-defined list object with fixed values/constants
        dat: Data list object with everything from the archive
        power: Radiation power data object list from calc
        program (str): XP ID
        indent_level: Indentation level
    Returns:
        fig, kappa, tau, res, fit: Figure references
    """
    print(indent_level + 'res ...', end=' ')

    try:
        [resMeas, resRef, stdRes] = \
            [[0] * (NChannels + 1), [0] * (NChannels + 1),
             [0] * (NChannels + 1)]
        for ch in range(0, NChannels + 1):
            resMeas[ch] = power['rohm'][ch]
            resRef[ch] = power['rohr'][ch]
            stdRes[ch] = comp_Res[ch][0]

        try:
            maxr = max(max(resMeas), max(resRef), max(stdRes)) * 1.05 / 1e3
            minr = min(min(resMeas), min(resRef), min(stdRes)) / 1e3
            res.set_ylim(minr, maxr)

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\  res lims failed')

        res.plot(channels, [x / 1e3 for x in resMeas],
                 ':', marker='o', color='r',
                 label='R (meas.)')
        res.plot(channels, [x / 1e3 for x in resRef],
                 '--', marker='^', color='b',
                 label='R (ref.)')
        res.plot(channels, [x / 1e3 if x is not None else 0.0
                            for x in stdRes],
                 '-.', marker='*', color='orange',
                 label='R (std.)')

        res.set_xlim(minc, maxc)
        # res.set_title('meas. and ref. resistance of foils')
        res.set_ylabel('R [kOhm]')
        res.legend()

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  res plot failed')

    return fig, kappa, tau, res, fit


def fit_results_out(
        comp_Kap=[[], []],
        NChannels=98,
        minc=-0.5,
        maxc=98.5,
        channels=[],
        fig=None,
        kappa=None,
        tau=None,
        res=None,
        fit=None,
        prio={'none': {'none': None}},
        dat={'none': {'none': None}},
        power={'none': {'none': None}},
        program='20181010.032',
        indent_level='\t'):
    """ Plots the fit results of meas., ref. foils
    Args:
        NChannels: Plot width/channel number to show
        minc: Minimum channel number
        maxc: Maximum channel number
        channels: Channel vector
        fig: Figure canvas reference
        kappa: Subplot heat capacity ref
        tau: Subplot cooling time
        res: Subplot electrical resistance
        fit: Fit parameter subplot ref
        prio: Pre-defined list object with fixed values/constants
        dat: Data list object with everything from the archive
        power: Radiation power data object list from calc
        program (str): XP ID
        indent_level: Indentation level
    Returns:
        fig, kappa, tau, res, fit: Figure references
    """
    print(indent_level + 'fit results ...', end=' ')

    try:
        [measAmp, measDamp] = \
            [[0] * (NChannels + 1), [0] * (NChannels + 1)]
        try:
            for ch in range(0, NChannels + 1):
                measDamp[ch] = power['fit_results'][0][ch, 1]
                measAmp[ch] = power['Imax'][0][ch]

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno)

        amps = fit.twinx()
        dampp, = fit.plot(
            channels, [x for x in measDamp], ':', marker='o', color='r',
            label='fit damping')
        ampp, = amps.plot(
            channels, [x * 1e3 for x in measAmp], '--', marker='^', color='b',
            label='fit amplitude')

        fit.set_xlim(minc, maxc)
        # fit.set_title('fit results')
        fit.set_ylabel('damping [1/ms]')
        fit.set_xlabel('Channel No.#')

        amps.yaxis.label.set_color(ampp.get_color())
        amps.tick_params(axis='y', colors=ampp.get_color())
        amps.spines["right"].set_edgecolor(ampp.get_color())
        amps.set_ylabel('amplitude [$\\mu$A]')

        lines = [dampp, ampp]
        fit.legend(lines, [l.get_label() for l in lines])

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  fit results plot failed')

    print('done!...')

    return fig, kappa, tau, res, fit
