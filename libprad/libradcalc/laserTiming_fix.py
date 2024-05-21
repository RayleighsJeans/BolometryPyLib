""" *************************************************************************
    so HEADER """

import json
import numpy as np
import sys
import warnings
import glob
import re
import time
import datetime as dt
import os

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")

Z = np.zeros
stdwrite = sys.stdout.write
stdflush = sys.stdout.flush

"""
20181011.015
HEXOS/LBO [ms]       Prad [ms] @ 0.8ms sT
1.305                1.282
1.904                1.882
2.454                2.430
3.704                3.688
4.303                4.286
4.853                4.828
6.603                6.586
5.453                5.430
6.005                5.988

20181016.016
HEXOS/LBO [ms]       Prad [ms] @ 1.6ms sT
1 =========================
0.00381213           -0.114
22.164               21.834
31.188               31.281
2 =========================
0.00381213           -0.122
21.868               21.485
31.151               30.6484


def lin_func(x, a, b):      def lin_func2(x, b):
    return a + b * x            return b * x

curve_fit(lin_fun(2), HEXOS, Prad, p0=(1., 1.), method='lm')

Out:[0.8ms sT]
    (array([0.023891  , 0.99914783]),
     array([[ 8.37481270e-06, -1.74109009e-06],
            [-1.74109009e-06,  4.30489243e-07]]))

    [1.6ms sT]
    1, lin_func ==============================================
    (array([0.18588567, 0.99617281]),
     array([[ 0.07653357, -0.00278742],
            [-0.00278742,  0.00015778]]))
    2, lin_func ==============================================
    (array([0.12585453, 1.01219451]),
     array([[ 3.48556785e-05, -1.29406258e-06],
            [-1.29406258e-06,  7.46416591e-08]]))

    1, lin_func2 =============================================
    (array([1.00294293]),
     array([[4.08267228e-05]]))
    2, lin_func2 =============================================
    (array([1.02221603]),
     array([[2.74971501e-05]]))

    [1.6ms]
    nsT, slope, offS, GSF = sT, 0.0, 89.71722, 1.01353062
    nsT, slope, offS, GSF = sT, 0.0, 185.88567, 0.99617281
    nsT, slope, offS, GSF = sT, 0.0, 125.85453, 1.01219451
    nsT, slope, offS, GSF = sT, 0.0, 0.0, 1.00294293
    nsT, slope, offS, GSF = sT, 0.0, 0.0, 1.02221603

    eo header
************************************************************************** """


def loadDatafiles(
        filename='../test.json'):
    with open(filename, 'r') as f:
        loaded = json.load(f)
    f.close()
    return loaded


def laserEval_main(
        use_dt=False,
        dateT=None,
        S=None,
        E=None,
        indent_level='\t'):
    """ evaluate laser experiments
    Args:
        use_dt (bool, optional):
        dateT (None, optional):
        S (None, optional):
        E (None, optional):
        indent_level (str, optional):
    Returns:
        None
    """

    base = 'X:/E5-IMP/Bolometer/DATA/OP1.2b_Lab/'
    fileL = glob.glob(base + 'BoloSignal_DATASTREAM_*')
    timeL = np.array([int(re.findall(r'\d+', v)[-1]) for v in fileL])
    S, E = \
        time.mktime(dt.datetime(2019, 7, 4).timetuple()) * 1e9, \
        time.mktime(dt.datetime(2019, 9, 4).timetuple()) * 1e9

    filterL = timeL[np.where(timeL[np.where(timeL > S)] < E)]

    for i, t in enumerate(filterL[2:]):
        try:
            inData = loadDatafiles(
                filename=base + 'BoloSignal_DATASTREAM_' + str(t) + '.json')

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(indent_level + '\t\\\ ', exc_type, '\n' + indent_level +
                  '\t\\\  file load failed: ' + str(t))

            continue

        try:
            values = inData['values'][2]
            T = inData['dimensions']
            evalChannelSignal(data=values, time=T)
            return

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
                  '\n' + indent_level + '\t\\\  time fix failed')

    return


def evalChannelSignal(
        data=Z((10000)),
        time=Z((10000)),
        indent_level='\t'):
    """ for a channel signal, eval timings
    Args:
        data (TYPE, optional):
        time (TYPE, optional):
        indent_level (str, optional):
    Returns:
        None
    """
    N = 33
    dt = time[1] - time[0]
    deriv = np.gradient(data, dt)
    scndDeriv = np.convolve(deriv, np.ones((N,)), mode='same')
    Xovers = np.where(np.diff(np.signbit(scndDeriv)))[0]

    thresh = int((0.1 / ((dt) / 1e9)) / 3)
    print('thresh:', thresh, 'Xovers:', Xovers[::2][0])

    # find out whether first is max or min
    find = np.max if (data[Xovers[0]] > data[Xovers[1]]) else np.min

    prev_val, new_time = None, []
    for i, val in enumerate(Xovers[::2]):

        if prev_val is None:
            new_time.extend(time[0:val])
            prev_val = val

        else:
            max_id = np.where(
                data[val - thresh:val + thresh] == find(
                    data[val - thresh:val + thresh]))[0][0] + (val - thresh)

            i, j = prev_val, max_id

            print('prev_val:', i, 'and val:', j)
            t_diff = (time[j] - time[i]) / 1e9
            print('t_diff =', str(t_diff) + 's', 'but should be 0.1s')

            new_time.extend(
                [new_time[-1] + i * dt * (0.1 / t_diff)
                 for i in range(j - i)])

            prev_val = max_id

    val = max_id

    if val < len(time):
        print(
            'too short, add up: val=', val, 'length - new_time=',
            len(new_time), 'to length - dimensions=', len(time))
        new_time.extend(time[val:])

    print('length now:', len(new_time), ' by adding',
          len(time[val:]))

    return
