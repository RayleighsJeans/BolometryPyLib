""" **************************************************************************
    so header """

import warnings
import numpy as np
import matplotlib.pyplot as p
from scipy.optimize import curve_fit

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")

""" eo header
************************************************************************** """


def lin_fun(
        t=1.0,
        a=1.0,
        b=1.0,
        c=1.0):
    """ a * u^b * v^c
    Args:
        t (float):
        a (float):
        b (float):
        c (float):
    Returns:
        a * u**(b) * v**(c)
    """
    u, v = t
    return a * u**(b) * v**(c)


def fit_lin(
        u=np.linspace(0, 1, 10),
        v=np.linspace(0, 1, 10),
        y=np.linspace(1, np.pi, 10)):
    """ fit wrapper for lines u,v and targets
    Args:
        u (list): Description
        v (list): Description
        y (list): Description
    Returns:
        None.
    """
    return curve_fit(
        lin_fun, (u, v), y, p0=(1, 1, 1), absolute_sigma=None,
        check_finite=True, method='lm')


def sudo_scaling_plot():
    """ Loads the previously, continuously created list of shot information
        and plots a selection of similar quantities like gas type and fueling
        method in graphs of n(ECRH) or P_rad(ECRH) and the alike.
    Args:
        None.
    Returns:
        None.
    Notes:
        0 date
        1 XP ID
        2 magconf
        3 ecrh_duration           0
        4 ecrh_power              1
        5 energy                  2
        6 gas1 type
        7 gas1 approach
        8 gas2 type
        9 gas2 approach
        10 gas3 type
        11 gas3 approach
        12 pellet desc
        13 HBC max Prad           3
        14 VBC max Prad           4
        15 max ECRH               5
        16 max Wdia               6
        17 max Eldens             7
        18 max Te out             8
        19 max Te core            9
        20 ECRH at max Prad      10
        21 Wdia at max Prad      11
        22 ne at max Prad        12
        23 Te out at max Prad    13
        24 Te in at max Prad     14
    """
    file = '../files/datafiles/power_diagnostic.dat'
    # load the files that have been written
    tmp_floats = np.loadtxt(
        file, delimiter='\t', dtype=np.float,
        usecols=[3, 4, 5, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

    # magnetic configurations lists
    big_frad = [
        i for i, x in enumerate(tmp_floats[:, 12])
        if 1.2 > tmp_floats[i, 3] / tmp_floats[i, 10] > .8 or
        1.2 > tmp_floats[i, 4] / tmp_floats[i, 10] > .8 and
        tmp_floats[i, 3] * tmp_floats[i, 4] < 12.]

    p.ioff()
    f, axes = p.subplots(ncols=4, nrows=2)

    axes[0, 0].plot(
        [tmp_floats[i, 10] for i in big_frad],
        [tmp_floats[i, 3] for i in big_frad],
        linestyle='', marker='<',
        color='orange', label='HBC f$_{rad}$>85%')
    axes[0, 0].plot(
        [tmp_floats[i, 10] for i in big_frad],
        [tmp_floats[i, 4] for i in big_frad],
        linestyle='', marker='>', color='cyan',
        label='VBC f$_{rad}$>85%')

    axes[0, 0].set_xlabel('ECRH [MW]')
    axes[0, 0].set_ylabel('P$_{rad}$ [MW]')
    axes[0, 0].legend()

    axes[0, 1].plot(
        [tmp_floats[i, 10] for i in big_frad],
        [tmp_floats[i, 12] for i in big_frad],
        linestyle='', marker='s', color='green',
        label='n$_{e}$ f$_{rad}$>85%')

    axes[0, 1].set_ylim(0, 15.)
    axes[0, 1].set_ylabel('n$_{e}$ [$10^{19}$ m$^{-3}$]')
    axes[0, 1].set_xlabel('ECRH [MW]')
    axes[0, 1].legend()

    axes[0, 2].plot(
        [tmp_floats[i, 10] for i in big_frad],
        [tmp_floats[i, 11] for i in big_frad],
        linestyle='', marker='h', color='r',
        label='W$_{dia}$ f$_{rad}$>85%')

    axes[0, 2].set_ylabel('W$_{dia}$ LCFS [kJ]')
    axes[0, 2].set_xlabel('ECRH [MW]')
    axes[0, 2].legend()

    axes[0, 3].plot(
        [tmp_floats[i, 10] for i in big_frad],
        [tmp_floats[i, 11] / tmp_floats[i, 10] for i in big_frad],
        linestyle='', marker='>', color='k',
        label='$\\tau_{e}$ f$_{rad}$>85%')

    axes[0, 3].set_ylabel('$\\tau_{e}$ [s]')
    axes[0, 3].set_xlabel('P$_{ECRH}$ [MW]')
    axes[0, 3].legend()

    axes[1, 0].plot(
        [tmp_floats[i, 12] for i in big_frad],
        [tmp_floats[i, 4] for i in big_frad],
        linestyle='', marker='>', color='cyan',
        label='VBC f$_{rad}$>85%')
    axes[1, 0].plot(
        [tmp_floats[i, 12] for i in big_frad],
        [tmp_floats[i, 3] for i in big_frad],
        linestyle='', marker='<', color='orange',
        label='HBC f$_{rad}$>85%')

    axes[1, 0].set_xlim(0, 15.)
    axes[1, 0].set_xlabel('n$_{e}$ [10$^{19}$ m$^{-3}$]')
    axes[1, 0].set_ylabel('P$_{rad}$ [MW]')
    axes[1, 0].legend()

    axes[1, 1].plot(
        [tmp_floats[i, 8] for i in big_frad],
        [tmp_floats[i, 4] for i in big_frad],
        linestyle='', marker='>', color='cyan',
        label='VBC f$_{rad}$>85%')
    axes[1, 1].plot(
        [tmp_floats[i, 8] for i in big_frad],
        [tmp_floats[i, 3] for i in big_frad],
        linestyle='', marker='<', color='orange',
        label='HBC f$_{rad}$>85%')

    axes[1, 1].set_xlim(0, 3.)
    axes[1, 1].set_xlabel('T$_{e}$ [keV]')
    axes[1, 1].set_ylabel('power [MW]')
    axes[1, 1].legend()

    vprad = [tmp_floats[i, 4] for i in big_frad]
    hprad = [tmp_floats[i, 3] for i in big_frad]
    dens = [tmp_floats[i, 12] for i in big_frad]
    ecrh = [tmp_floats[i, 10] for i in big_frad]

    [a, b, c], fitres = fit_lin(ecrh, dens, vprad)
    axes[1, 2].plot(
        [a * v**(b) * u**(c) for u, v in zip(ecrh, dens)], vprad,
        linestyle='', marker='>', color='cyan',
        label='P$_{rad, VBC}$; a=' + str(round(a, 3)) + '; b=' +
        str(round(b, 3)) + '; c=' + str(round(c, 3)))

    [a, b, c], fitres = fit_lin(ecrh, dens, hprad)
    axes[1, 2].plot(
        [a * v**(b) * u**(c) for u, v in zip(ecrh, dens)], hprad,
        linestyle='', marker='<', color='orange',
        label='P$_{rad, VBC}$; a=' + str(round(a, 3)) + '; b=' +
        str(round(b, 3)) + '; c=' + str(round(c, 3)))

    axes[1, 2].legend()
    axes[1, 2].set_ylabel('P$_{rad}$')
    axes[1, 2].set_xlabel('a$\\cdot$P$_{ECRH}$$^{c}\\cdot$n$_{e}$$^{b}$')

    f.set_size_inches(20, 10)
    f.savefig('../results/power_diagnostic.pdf')
    p.close('all')
    return
