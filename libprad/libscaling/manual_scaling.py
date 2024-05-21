""" ********************************************************************** """

import warnings
import numpy as np
import matplotlib.pyplot as p
from scipy.optimize import curve_fit

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")
warnings.filterwarnings("ignore", "Reloaded modules")

""" ********************************************************************** """


def lin_fun(
        t=[[1.0], [1.0]],
        a=[1.0],
        b=[1.0],
        c=[1.0]):
    """Summary
    Args:
        t (list, optional): Input vectors
        a (list, optional): Proportio factor
        b (list, optional): First exponent
        c (list, optional): Second exponent
    Returns:
        a * u**(b) * v**(c) (list): Calculated results
    """
    u, v = t
    return a * u**(b) * v**(c)


def fit_lin(
        u=[1.0],
        v=[1.0],
        y=[1.0]):
    """Summary
        Fit of u,v to y
    Args:
        u (list, optional): Arg one
        v (list, optional): Arg two
        y (list, optional): Dataset to fit
    Returns:
        Scipy fit like lin_fun based off of (1, 1, 1)
    """
    return curve_fit(lin_fun, (u, v), y, p0=(1, 1, 1), absolute_sigma=None,
                     check_finite=True, method='lm')


def scaling_quick():
    """Summary
        Scales and fits according to manually collected data
    Notes:
        0 vbc
        1 hbc
        2 ne
        3 Te
        4 ECRH
        5 main gas inlet
        6 HeBeam
        7 Wdia

        f(a,b,c) = a*ne^b*P_ECRH^c == P_rad
        --> fit via steady state database
        plot fit vs. P_rad/d_Prad
    Returns:
        None
    """
    filecols = \
        [[
         '../files/datafiles/peak_XPIDs.dat',
         '../files/datafiles/peak_before_times.dat',
         '../files/datafiles/peak_before_values.dat',
         '../files/datafiles/peak_after_times.dat',
         '../files/datafiles/peak_after_values.dat'],
         [
         '../files/datafiles/steady_XPIDs.dat',
         '../files/datafiles/steady_before_times.dat',
         '../files/datafiles/steady_before_values.dat',
         '../files/datafiles/steady_after_times.dat',
         '../files/datafiles/steady_after_values.dat'],
         [
         [0, 1, 2],
         [0, 1, 2, 3, 4, 5, 6, 7],
         [0, 1, 2, 3, 4, 5, 6, 7],
         [0, 1, 2, 3, 4, 5, 6, 7],
         [0, 1, 2, 3, 4, 5, 6, 7]
         ]]

    [foo, j] = [[], 0]
    res = [None] * (2 * len(filecols[0]))
    # load the files that have been written
    for k in [0, 1]:
        for i in range(0, len(filecols[0])):
            cols = filecols[2][i]
            res[j + i] = np.loadtxt(filecols[k][i], delimiter='\t',
                                    dtype=np.float, usecols=cols)
        j = len(filecols[0])

    p.ioff()
    f, axes = p.subplots(nrows=2, ncols=2)
    f.set_size_inches(12, 12)

    """ PEAKING CASES """
    [hpbef, hpaf] = [[t[0] for t in res[2][:]], [t[0] for t in res[4][:]]]
    [vpbef, vpaf] = [[t[1] for t in res[2][:]], [t[1] for t in res[4][:]]]
    [densbef, densaf] = [[t[2] for t in res[2][:]], [t[2] for t in res[4][:]]]
    [tebef, teaf] = [[t[3] for t in res[2][:]], [t[3] for t in res[4][:]]]
    [ecrhbef, ecrhaf] = [[t[4] for t in res[2][:]], [t[4] for t in res[4][:]]]
    [flowbef, flowaf] = [[t[5] for t in res[2][:]], [t[5] for t in res[4][:]]]

    print('peaks\n# \t' + 'flow\t' + 'dHBC\t' + 'dVBC\t' + 'dne')
    for i in range(0, len(flowaf)):
        print(str(i).zfill(2), '\t', round(flowaf[i] - flowbef[i], 2), '\t',
              round(hpaf[i] - hpbef[i], 2), '\t',
              round(vpaf[i] - vpbef[i], 2), '\t',
              round(densaf[i] - densbef[i], 2))

    # plot raw data for puffs, VBC&HBC
    axes[0, 0].plot(
        [x - y for x, y in zip(flowaf, flowbef)],
        [x - y for x, y in zip(hpaf, hpbef)],
        linestyle='', marker='^',
        color='r', label='peak $\\Delta$P$_{HBC}$')
    axes[0, 0].plot(
        [x - y for x, y in zip(flowaf, flowbef)],
        [x - y for x, y in zip(vpaf, vpbef)],
        linestyle='', marker='',
        color='b', label='peak $\\Delta$P$_{VBC}$')

    # resort all according to VBC prad data, so definitively increasing
    # and fit afterwards, to get rough estimates for a,b,c
    vpaf, densaf, ecrhaf, hpaf, teaf, flowaf, \
        vpbef, densbef, ecrhbef, hpbef, tebef, flowbef = \
        zip(*sorted(zip(
            vpaf, densaf, ecrhaf, hpaf, teaf, flowaf,
            vpbef, densbef, ecrhbef, hpbef, tebef, flowbef)))
    [av, bv, cv], fitres = fit_lin(densaf, ecrhaf, vpaf)
    [k, l, m], fitres = fit_lin(flowaf, ecrhaf, vpaf)

    # plot results against fit
    axes[0, 1].plot(
        [av * u**(bv) * v**(cv) for u, v in zip(densaf, ecrhaf)], vpaf,
        label='peak P$_{rad, VBC}$', markersize=8.0,
        linestyle='', marker='v', color='b')
    axes[1, 1].plot(
        [k * u**(l) * v**(m) for u, v in zip(flowaf, ecrhaf)], vpaf,
        label='peak P$_{rad, VBC}$', markersize=8.0,
        linestyle='', marker='v', color='b')

    hpaf, densaf, ecrhaf, vpaf, teaf, flowaf, \
        vpbef, densbef, ecrhbef, hpbef, tebef, flowbef = \
        zip(*sorted(zip(
            hpaf, densaf, ecrhaf, vpaf, teaf, flowaf,
            vpbef, densbef, ecrhbef, hpbef, tebef, flowbef)))
    [ah, bh, ch], fitres = fit_lin(densaf, ecrhaf, hpaf)
    [r, s, q], fitres = fit_lin(flowaf, ecrhaf, hpaf)

    # plot results against fit
    axes[0, 1].plot(
        [ah * u**(bh) * v**(ch) for u, v in zip(densaf, ecrhaf)], hpaf,
        label='peak P$_{rad, HBC}$', markersize=8.0,
        linestyle='', marker='^', color='r')
    axes[1, 1].plot(
        [r * u**(s) * v**(q) for u, v in zip(flowaf, ecrhaf)], hpaf,
        label='peak P$_{rad, HBC}$', markersize=8.0,
        linestyle='', marker='^', color='r')

    # plot line as guess for factors so it looks math/nice
    foo, bar = zip(*sorted(zip(
        [ah * u**(bh) * v**(ch) for u, v in zip(densaf, ecrhaf)],
        [1 / 5 * u**(4 / 5) * v**(2 / 5) for u, v in zip(densaf, ecrhaf)])))
    axes[0, 1].plot(
        [foo[0], foo[-1]], [bar[0], bar[-1]],
        label='peak: a=1/5, b=4/5, c=2/5', linestyle='-.', color='k')

    foo, bar = zip(*sorted(zip(
        [r * u**(s) * v**(q) for u, v in zip(flowaf, ecrhaf)],
        [0.05 * u**(0.6) * v**(0.8) for u, v in zip(flowaf, ecrhaf)])))
    axes[1, 1].plot(
        [foo[0], foo[-1]], [bar[0], bar[-1]],
        label='peak: a=1/20, b=3/5, c=4/5', linestyle='-.', color='k')

    # write fit results
    print('\n>> fit 1, peaky:\n' +
          '\tah=' + str(round(ah, 3)) + ', bh=' + str(round(bh, 3)) +
          ', ch=' + str(round(ch, 3)) + '\n' +
          '\tav=' + str(round(av, 3)) + ', bv=' + str(round(bv, 3)) +
          ', cv=' + str(round(cv, 3)) + '\n' +
          '\tk=' + str(round(k, 3)) + ', l=' + str(round(l, 3)) +
          ', m=' + str(round(m, 3)) + '\n' +
          '\tr=' + str(round(r, 3)) + ', s=' + str(round(s, 3)) +
          ', q=' + str(round(q, 3)) + '\n')

    """ STEADY STATE CASES """
    [hpbef, hpaf] = [[t[0] for t in res[7][:]], [t[0] for t in res[9][:]]]
    [vpbef, vpaf] = [[t[1] for t in res[7][:]], [t[1] for t in res[9][:]]]
    [densbef, densaf] = [[t[2] for t in res[7][:]], [t[2] for t in res[9][:]]]
    [tebef, teaf] = [[t[3] for t in res[7][:]], [t[3] for t in res[9][:]]]
    [ecrhbef, ecrhaf] = [[t[4] for t in res[7][:]], [t[4] for t in res[9][:]]]
    [flowbef, flowaf] = [[t[5] for t in res[7][:]], [t[5] for t in res[9][:]]]

    print('steady state\n# \t' + 'flow\t' + 'dHBC\t' + 'dVBC\t' + 'dne')
    for i in range(0, len(flowaf)):
        print(str(i).zfill(2), '\t', round(flowaf[i] - flowbef[i], 2), '\t',
              round(hpaf[i] - hpbef[i], 2), '\t',
              round(vpaf[i] - vpbef[i], 2), '\t',
              round(densaf[i] - densbef[i], 2))

    # plot raw data for puffs, VBC&HBC
    axes[0, 0].plot(
        [x - y for x, y in zip(flowaf, flowbef)],
        [x - y for x, y in zip(hpaf, hpbef)],
        linestyle='', marker='p',
        color='orange', label='steady $\\Delta$P$_{HBC}$')
    axes[0, 0].plot(
        [x - y for x, y in zip(flowaf, flowbef)],
        [x - y for x, y in zip(vpaf, vpbef)],
        linestyle='', marker='p',
        color='cyan', label='steady $\\Delta$P$_{VBC}$')

    # resort all according to VBC prad data, so definitively increasing
    # and fit afterwards, to get rough estimates for a,b,c
    vpaf, densaf, ecrhaf, hpaf, teaf, flowaf, \
        vpbef, densbef, ecrhbef, hpbef, tebef, flowbef = \
        zip(*sorted(zip(
            vpaf, densaf, ecrhaf, hpaf, teaf, flowaf,
            vpbef, densbef, ecrhbef, hpbef, tebef, flowbef)))
    [av, bv, cv], fitres = fit_lin(densaf, ecrhaf, vpaf)
    [k, l, m], fitres = fit_lin(flowaf, ecrhaf, vpaf)
    # plot results against fit
    axes[0, 1].plot(
        [av * u**(bv) * v**(cv) for u, v in zip(densaf, ecrhaf)], vpaf,
        label='steady P$_{rad, VBC}$', markersize=8.0,
        linestyle='', marker='h', color='cyan')
    axes[1, 1].plot(
        [k * u**(l) * v**(m) for u, v in zip(flowaf, ecrhaf)], vpaf,
        label='steady P$_{rad, VBC}$', markersize=8.0,
        linestyle='', marker='h', color='cyan')

    hpaf, densaf, ecrhaf, vpaf, teaf, flowaf, \
        vpbef, densbef, ecrhbef, hpbef, tebef, flowbef = \
        zip(*sorted(zip(
            hpaf, densaf, ecrhaf, vpaf, teaf, flowaf,
            vpbef, densbef, ecrhbef, hpbef, tebef, flowbef)))
    [ah, bh, ch], fitres = fit_lin(densaf, ecrhaf, hpaf)
    [r, s, q], fitres = fit_lin(flowaf, ecrhaf, hpaf)
    # plot results against fit
    axes[0, 1].plot(
        [ah * u**(bh) * v**(ch) for u, v in zip(densaf, ecrhaf)], hpaf,
        label='steady P$_{rad, HBC}$', markersize=8.0,
        linestyle='', marker='p', color='orange')
    axes[1, 1].plot(
        [r * u**(s) * v**(q) for u, v in zip(flowaf, ecrhaf)], hpaf,
        label='steady P$_{rad, HBC}$', markersize=8.0,
        linestyle='', marker='p', color='orange')

    foo, bar = zip(*sorted(zip(
        [ah * u**(bh) * v**(ch) for u, v in zip(densaf, ecrhaf)],
        [1 / 16 * u**(3) * v**(-2) for u, v in zip(densaf, ecrhaf)])))
    axes[0, 1].plot(
        [foo[0], foo[-1]], [bar[0], bar[-1]],
        label='steady: a=1/16, b=3, b=-2', linestyle='-.', color='m')

    foo, bar = zip(*sorted(zip(
        [r * u**(s) * v**(q) for u, v in zip(flowaf, ecrhaf)],
        [40 * u**(-0.8) * v**(-0.15) for u, v in zip(flowaf, ecrhaf)])))
    axes[1, 1].plot(
        [foo[0], foo[-1]], [bar[0], bar[-1]],
        label='steady: a=40, b=-4/5, c=-1/7', linestyle='-.', color='m')

    # write fit results
    print('\n>> fit 1, steady:\n' +
          '\tah=' + str(round(ah, 3)) + ', bh=' + str(round(bh, 3)) +
          ', ch=' + str(round(ch, 3)) + '\n' +
          '\tav=' + str(round(av, 3)) + ', bv=' + str(round(bv, 3)) +
          ', cv=' + str(round(cv, 3)) + '\n' +
          '\tk=' + str(round(k, 3)) + ', l=' + str(round(l, 3)) +
          ', m=' + str(round(m, 3)) + '\n' +
          '\tr=' + str(round(r, 3)) + ', s=' + str(round(s, 3)) +
          ', q=' + str(round(q, 3)) + '\n')

    axes[0, 0].legend()
    axes[0, 0].set_ylabel('$\\Delta$ radiation power [MW]')
    axes[0, 0].set_xlabel('$\\Delta$ flow rate [mbar$\\cdot$L/s]')

    axes[0, 1].set_title('fit au$^{c}$v$^{b}$')
    axes[0, 1].legend()
    axes[0, 1].set_ylabel('P$_{rad}$ [MW]')
    axes[0, 1].set_xlabel('a$\\cdot$n$_{e}$$^{b}$$\\cdot$P$_{ECRH}$$^{c}$')

    axes[1, 1].set_title('fit au$^{c}$v$^{b}$')
    axes[1, 1].legend()
    axes[1, 1].set_ylabel('P$_{rad}$ [MW]')
    axes[1, 1].set_xlabel('a$\\cdot$f$_{H2}$$^{b}$$\\cdot$P$_{ECRH}$$^{c}$')

    f.savefig('../results/scaling_test.pdf')
    p.close('all')

    return
