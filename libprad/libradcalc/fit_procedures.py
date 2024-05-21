""" **************************************************************************
    so header """

import sys
import os
import warnings
import numpy as np
import matplotlib.pyplot as p
from scipy.optimize import curve_fit

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")

Z = np.zeros
ones = np.ones
M = 10000

""" eo header
************************************************************************** """


def exponential_func(
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
    return a * np.exp(b * x)


def plot_fits(
        base='FirstStepFitMeas.',
        date='20181010',
        shotno=32,
        fit_results=[1, 1],
        ch=1,
        measTime=Z(M),
        measCurr=ones(M),
        XX=Z(8000),
        YY=Z(8000),
        EXP=Z(8000)):
    """ Plots the fitting results in comparison to the used data.
    Args:
        base (0, str): Base portion of the file name.
        date (1, str): Date of shot.
        shotno (2, int): Experiment number/ID.
        fit_results (3, list): Results a,b from exponential function fit.
        ch (4, int): Channel number.
        measTime (5, list): Time vector of data.
        measCurr (6, list): Real current measured.
        XX (7, list): Selected time interval.
        YY (8, list): Data of fit.
        EXP (9, list): Exponential fit result.
    Returns:
        None
    Notes:
        None
    """

    location = r'../results/FITSTEPS/' + date + '/' + 'channel_' + str(ch)
    if not os.path.exists(location):
        os.makedirs(location)
    fig = p.figure()
    fig.set_size_inches(4., 4.)
    ax = p.subplot(111)
    ax.plot(
        measTime, [1e6 * z for z in measCurr],
        label='meas. current')
    ax.plot(XX, [1e6 * y for y in YY], label='fit data')
    ax.plot(XX, [1e6 * x for x in EXP], label='fit')
    ax.text(
        0.2, 1e6 * 0.5 * max(EXP) + 20,
        'exponential fit')
    ax.text(
        0.2, 1e6 * 0.5 * max(EXP) + 10,
        'amplitude: ' + str(round(fit_results[0], 7)))
    ax.text(
        0.2, 1e6 * 0.5 * max(EXP),
        'damping: ' + str(round(fit_results[1], 3)))
    ax.set_ylabel('current [$\\mu$A]')
    ax.set_xlabel('rel. T - T$_{heat pulse}$ [ms]')
    ax.set_title(
        'shot #' + date + '.' + shotno +
        ' ch. ' + '[' + str(ch) + ']')
    ax.legend()
    fig.savefig(
        '../results/FITSTEPS/' + date + '/' + 'channel_' + str(ch) +
        '/' + base + '_' + date + '_' + shotno + '_' + str(ch) + '.pdf')
    if (ch == 0):
        fig.savefig('../results/CURRENT/' + base + '_' + str(ch) + '.pdf')
    p.close('all')
    return


def fit_parameters(
        indent_level='\t',
        dat={'none': {'none': None}},
        shotno=32,
        date='20181010',
        printing=False,
        debug=False):
    """ Fitting routine to get best results from current calibration of
        measurement and reference foils of the detectors.
    Args:
        indent_level (0, str): Indentation printing level.
        data_object (1, list): Bolometer data.
        shotno (2, int): Experiment ID.
        date (3, str): Experiment date.
    Returns:
        kappam (0, list): Meas. heat capacity.
        rohm (1, list): Meas. electrical resistance.
        taum (2, list): Meas. cooling time.
        kappar (3, list): ...
        rohr (4, list): ...
        taur (5, list): ...
        fit_results (6, list): A, B from the exponential function.
        Imax (7, list): max current at the bridge through meas./ref. foils
    Notes:
        None.
    """
    [kappam, rohm, taum, kappar, rohr, taur, fit_results, Imax] = \
        [[], [], [], [], [], [], [], []]

    # stuff for inside loop
    NR = 4000  # points
    NShift = 55  # points
    i = 0
    default = [6e-2, -2.1]  # fit parameters (a, b); a * exp(b * x)
    titles = ['FirstStepFitMeas.', 'FirstStepFitRef.']

    if printing:
        print(indent_level + '>> Fitting foil parameters :')
    measCurr = dat['BoloCalibMeasFoilCurrent']['values']  # in A
    refCurr = dat['BoloCalibRefFoilCurrent']['values']  # in A

    # time vectors
    measTime = [(x - dat['BoloCalibMeasFoilCurrent']['dimensions'][0]) / 1e9
                for x in dat['BoloCalibMeasFoilCurrent']['dimensions']]  # in s
    measTime = np.linspace(measTime[0], measTime[0] + (
        0.4e-3 * 9999), 10000)  # fixing the time to s, 0.4e-3 s sampling

    refTime = measTime  # in s
    dts = [measTime[1] - measTime[0], refTime[1] - refTime[0]]  # in s
    # specified properties
    Rload = 10.  # in Ohm
    Rcable = 40.  # in Ohm
    # second derivative of current to find turning point
    d2Idt2 = np.zeros(len(measCurr[0]))  # second deriv of current A/s^2
    # positions of slope maximums
    positions_meas = np.zeros((128))  # in points
    positions_ref = np.zeros((128))  # in points
    # slope heights
    Imax_meas = np.zeros((128))  # in A
    Imax_ref = np.zeros((128))  # in A
    # different step heights
    I01_meas = np.zeros((128))  # in A
    I02_meas = np.zeros((128))  # in A
    I03_meas = np.zeros((128))  # in A

    I01_ref = np.zeros((128))  # in A
    I02_ref = np.zeros((128))  # in A
    I03_ref = np.zeros((128))  # in A
    # fit results
    fit_results_meas = np.zeros((128, 2))
    fit_results_ref = np.zeros((128, 2))

    # final fit params
    RB01 = np.zeros((128))  # in Ohm
    RB02 = np.zeros((128))  # in Ohm
    taum = np.zeros((128))  # in s
    kappam = np.zeros((128))  # in
    RB01_ref = np.zeros((128))  # in Ohm
    RB02_ref = np.zeros((128))  # in Ohm
    taum_ref = np.zeros((128))   # in s
    kappam_ref = np.zeros((128))  # in A^2

    # fitting results to check afterwards
    XX = np.zeros((2, 128, NR))  # in s
    YY = np.zeros((2, 128, NR))  # in A
    EXP = np.zeros((2, 128, NR))  # in A

    # needs to be done for meas and ref curr
    currs = [measCurr, refCurr]  # in A
    times = [measTime, refTime]  # in s
    positions = [positions_meas, positions_ref]
    Imax = [Imax_meas, Imax_ref]  # in A
    I0 = [[I01_meas, I02_meas, I03_meas],  # in A
          [I01_ref, I02_ref, I03_ref]]
    fit_results = [fit_results_meas, fit_results_ref]
    fit_parameters = [[RB01, RB02, taum, kappam],
                      [RB01_ref, RB02_ref, taum_ref, kappam_ref]]

    for curr in currs:  # for meas and ref foils
        title = titles[i]  # names
        if printing:
            print(indent_level + '   ' + title, end='')

        for ch in range(0, len(curr) - 1):
            if (np.remainder(ch, 10) == 0) and printing:
                print(ch, "...", end="")
            if (ch == 90) and printing:
                print('\n\t                              ', end='')

            try:
                # pre adjusting the spot where the max is
                d2Idt2 = np.array(np.gradient(np.gradient(
                    curr[ch], dts[i]), dts[i]))  # in A/s^2
                # step heights in calib
                I0[i][0][ch] = np.mean(curr[ch][900:1900])  # in A
                I0[i][1][ch] = np.mean(curr[ch][4900:5900])  # in A
                I0[i][2][ch] = np.mean(curr[ch][8900:9900])  # in A

                # first step in voltage signal
                foo = max(np.array(d2Idt2)[1900:NR])
                bar = int(np.where(np.array(d2Idt2) == foo)[0])

                # getting closer on the max of the slope and shift
                bar = max(np.array(curr[ch][bar - 10:bar + 10]))  # in A
                foo = int(np.where(np.array(curr[ch]) == bar)[0][0]) + NShift
                positions[i][ch] = foo
                Imax[i][ch] = curr[ch][foo - NShift]  # in A

                # fit data used on first step
                T0 = times[i][foo]  # in s

                # arranging fitting stuff
                XX[i][ch][0:NR - foo] = [x - T0 for x in times[i][foo:NR]]  # s
                YY[i][ch][0:NR - foo] = np.abs(  # in A
                    [x - I0[i][1][ch] for x in curr[ch][foo:NR]])
                # fitting using the least square method, first step
                fit_results[i][ch], tmp = \
                    curve_fit(exponential_func,
                              XX[i][ch][0:NR - foo],  # in s
                              YY[i][ch][0:NR - foo],  # in A
                              p0=(default[0], default[1]),
                              sigma=None, absolute_sigma=None,
                              check_finite=True, method='lm')

                # dump it
                EXP[i][ch][0:NR - foo] = \
                    exponential_func(np.array(XX[i][ch][0:NR - foo]),  # in A
                                     fit_results[i][ch, 0],
                                     fit_results[i][ch, 1])
                # make better guess for next time
                default = [(default[0] + fit_results[i][ch, 0]) / 2,
                           (default[1] + fit_results[i][ch, 1]) / 2]

                # plot to tmp directory
                if False:  # ch in [32, 36, 40, 44, 48, 52, 56, 60]:
                    # plotting fit results
                    plot_fits(
                        base=title, date=date, shotno=shotno,
                        fit_results=fit_results[i][ch], ch=ch,
                        measTime=[t - T0 for t in times[i][foo - NShift:NR]],
                        measCurr=np.abs([x - I0[i][1][ch] for x in
                                         curr[ch][foo - NShift:NR]]),
                        XX=XX[i][ch][0:NR - foo], YY=YY[i][ch][0:NR - foo],
                        EXP=exponential_func(
                            np.array(XX[i][ch][0:NR - foo]),
                            fit_results[i][ch][0], fit_results[i][ch][1]))

                # RB01; 2 * (2.5V / (Imax - I0)) - Rl - Rcab) -> Ohm
                fit_parameters[i][0][ch] = \
                    2 * (((2.5) / (Imax[i][ch] - I0[i][0][ch])) -
                         (Rload + Rcable))
                # RB02; (1.2v / (Imax - I0) - Rl - Rcab) -> Ohm
                fit_parameters[i][1][ch] = \
                    2 * (((1.2) / (I0[i][2][ch] - I0[i][1][ch])) -
                         (Rload + Rcable))
                # taum; 1 / (1 / t) -> s
                fit_parameters[i][2][ch] = \
                    1 / (fit_results[i][ch][1]) * (-1)
                # kappam; Rb * (Idiff^4 / (4 * U * dI)) -> A^2
                fit_parameters[i][3][ch] = \
                    fit_parameters[i][0][ch] * \
                    (((((I0[i][1][ch] - I0[i][0][ch])**4) / 4) / 2.5) /
                     (Imax[i][ch] - I0[i][1][ch]))
                # adjusted R_OH; (Rb + sqrt(rb^2 + 25 / kappa))* 1/2 -> Ohm
                fit_parameters[i][0][ch] = \
                    (fit_parameters[i][0][ch] +
                     np.sqrt(fit_parameters[i][0][ch]**2 +
                             5**2 / fit_parameters[i][3][ch])) / 2

            except Exception:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                if debug:
                    print(indent_level + '\t\\\ ', exc_type, fname,
                          exc_tb.tb_lineno, '\n' + indent_level +
                          '\t\\\  error in fit at ch#' + str(ch) +
                          ' using average of last three')

                # dirty fix for what ?!?!
                fit_parameters[i][0][ch] = \
                    np.mean(fit_parameters[i][0][ch - 4:ch - 1])  # RB01, Ohm
                fit_parameters[i][1][ch] = \
                    np.mean(fit_parameters[i][1][ch - 4:ch - 1])  # RB02, Ohm
                fit_parameters[i][2][ch] = \
                    np.mean(fit_parameters[i][2][ch - 4:ch - 1])  # taum, s
                fit_parameters[i][3][ch] = \
                    np.mean(fit_parameters[i][3][ch - 4:ch - 1])  # kappam, A^2
                # adjusted R_OH
                fit_parameters[i][0][ch] = \
                    np.mean(fit_parameters[i][0][ch - 4:-1])  # adj R_OH, Ohm

        if printing:
            print('\n', end='')
        i += 1
    if printing:
        print(indent_level + '   ... Done!', end='\n')

    # all the results
    [kappam, rohm, taum, kappar, rohr, taur] = [
        fit_parameters[0][3], fit_parameters[0][0], fit_parameters[0][2],
        fit_parameters[1][3], fit_parameters[1][0], fit_parameters[1][2]]

    # dump to file
    location = r'../results/FITPARAMS/' + date + '/'
    if not os.path.exists(location):
        os.makedirs(location)
    file = r'../results/FITPARAMS/' + date + '/' + date + \
        '.' + str(int(shotno)).zfill(3) + '.dat'

    with open(file, 'wb') as outfile:
        np.savetxt(outfile,
                   np.c_[positions[0], [NR - x for x in positions[0]],
                         fit_results[0][:, 0], fit_results[0][:, 1],
                         Imax[0], I0[0][0], I0[0][1], I0[0][2]],
                   delimiter="\t\t", newline="\n")

    # debugging
    if printing:
        print(indent_level + '>> fit result bounds: max ampl.',
              format(max(Imax[0][0:98]), '.3e'), 'A; min ampl.',
              format(min(Imax[0][0:98]), '.3e'), 'A\n' + indent_level,
              '                     max damp.',
              format(max(fit_results[0][0:98, 1]), '.3e'), 's; min damp.',
              format(min(fit_results[0][0:98, 1]), '.3e'), 's')

    return [kappam, rohm, taum, kappar, rohr, taur, fit_results, Imax]
