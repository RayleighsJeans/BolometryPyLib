""" *************************************************************************
    so HEADER """

import os
import sys
import glob
import warnings
import json
from sklearn.gaussian_process import kernels
from tqdm import tqdm

import numpy as np
import scipy as sc
import requests
import random as rand
import math

import matplotlib.pyplot as p
from matplotlib.pyplot import cm
from itertools import cycle
from fractions import Fraction as frac

import mClass
import dat_lists as dat_lists
import archivedb
import webapi_access as api

import thomsonS_access as TS
import read_calculation_corona as rcc
# import hexos_getter as hexos_get

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")

Z = np.zeros
stdwrite = sys.stdout.write
stdflush = sys.stdout.flush

""" eo header
************************************************************************** """


def fig_current_save(
        name='test',
        figure=None):

    for i, ftype in enumerate(['.pdf', '.png']):
        figure.savefig('../results/CURRENT/' + name + ftype, dpi=169.0)
    return


def autoscale_x(
        ax,
        margin=0.1,
        debug=False):
    """ This function rescales the x-axis based on the data that is visible
        given the current ylim of the axis.
    Args:
        ax: a matplotlib axes object
        margin: the fraction of the total height of the y-data to pad the
            upper and lower ylims
    Returns:
        left: bottom limit
        right: top limit
    Note:
        None.
    """
    def get_left_right(line, old_left, old_right):
        left, right = 1e18, -1e18
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo, hi = ax.get_ylim()
        if debug:
            print('xd=', xd, '\nyd=', yd, '\nlo=', lo, 'hi=', hi)

        if (xd[0] == xd[-1]) or (yd[0] == yd[-1]) and \
                (xd[0] == xd[1]) or (yd[0] == yd[1]):
            if debug:
                print('first xd/yd is last and second')
            return old_left, old_right

        try:
            L = [i for i, v in enumerate(yd) if v > lo and v < hi]
            if debug:
                print('L=', L)
            x_displayed = xd[L]
            h = np.max(x_displayed) - np.min(x_displayed)
            left = np.min(x_displayed) - margin * h
            right = np.max(x_displayed) + margin * h
            if debug:
                print('x_displayed=', x_displayed)
                print('left=', left, 'right=', right)

        except Exception:
            if debug:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print('\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
                      '\n\t\\\  failed get line data')
            left, right = np.inf, -np.inf

        return left, right

    left, right = np.inf, -np.inf
    line = None
    lines = ax.get_lines()
    if debug:
        print('lines=', lines, lines[0].__class__)

    for line in lines:
        if debug:
            print('line=', line.get_label())
        new_left, new_right = get_left_right(line, left, right)
        if debug:
            print('from line new_left=', new_left, 'new_right=', new_right)
        if (new_left != np.inf) and (new_right != -np.inf):
            if new_left < left:
                left = new_left
            if new_right > right:
                right = new_right

        if debug:
            print('new_left=', new_left, 'new_right=', new_right)

    if debug:
        print('left=', left, 'right=', right)

    ax.set_xlim(left, right)
    return


def autoscale_y(
        ax,
        margin=0.1,
        debug=False):
    """ This function rescales the y-axis based on the data that is visible
        given the current xlim of the axis.
    Args:
        ax: a matplotlib axes object
        margin: the fraction of the total height of the y-data to pad the
            upper and lower ylims
    Returns:
        bot: bottom limit
        top: top limit
    Note:
        None.
    """
    def get_bottom_top(line, old_bot, old_top):
        bot, top = 1e18, -1e18
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo, hi = ax.get_xlim()
        if debug:
            print('xd=', xd, '\nyd=', yd, '\nlo=', lo, 'hi=', hi)

        if (xd[0] == xd[-1]) or (yd[0] == yd[-1]) and \
                (xd[0] == xd[1]) or (yd[0] == yd[1]):
            if debug:
                print('first xd/yd is last and second')
            return old_bot, old_top

        try:
            L = [i for i, v in enumerate(xd) if v > lo and v < hi]
            if debug:
                print('L=', L)
            y_displayed = yd[L]
            h = np.max(y_displayed) - np.min(y_displayed)
            bot = np.min(y_displayed) - margin * h
            top = np.max(y_displayed) + margin * h
            if debug:
                print('bot=', bot, 'top=', top)

        except Exception:
            if debug:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print('\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
                      '\n\t\\\  failed get line data')
            bot, top = np.inf, -np.inf

        return bot, top

    bot, top = np.inf, -np.inf
    line = None
    lines = ax.get_lines()
    if debug:
        print('lines=', lines, lines[0].__class__)

    for line in lines:
        if debug:
            print('line=', line.get_label())
        new_bot, new_top = get_bottom_top(line, bot, top)
        if (new_bot != np.inf) and (new_top != -np.inf):
            if new_bot < bot:
                bot = new_bot
            if new_top > top:
                top = new_top

        if debug:
            print('new_bot=', new_bot, 'new_top=', new_top)

    try:
        ax.set_ylim(bot, top)
    except Exception:
        pass
    return


def autoscale_z(
        ax,
        margin=0.1,
        debug=False):
    """ This function rescales the z-axis based on the data that is visible
        given the current xlim of the axis.
    Args:
        ax: a matplotlib axes object
        margin: the fraction of the total height of the y-data to pad the
            upper and lower ylims
    Returns:
        inn: bottom limit
        out: top limit
    Note:
        None.
    """
    def get_inn_out(line, old_inn, old_out):
        xd, yd, zd = line._verts3d
        inn, out = 1e18, -1e18
        lo, hi = ax.get_xlim()
        if debug:
            print('xd=', xd, '\nzd=', zd, '\nlo=', lo, 'hi=', hi)

        if ((xd[0] == xd[-1]) or (zd[0] == zd[-1])) and \
                ((xd[0] == xd[1]) or (zd[0] == zd[1])):
            if debug:
                print('first xd/zd is last and second')
            return old_inn, old_out

        try:
            L = [i for i, v in enumerate(xd) if v > lo and v < hi]
            if debug:
                print('L=', L)
            z_displayed = zd[L]
            h = np.max(z_displayed) - np.min(z_displayed)
            inn = np.min(z_displayed) - margin * h
            out = np.max(z_displayed) + margin * h
            if debug:
                print('inn=', inn, 'out=', out)

        except Exception:
            if debug:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print('\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
                      '\n\t\\\  failed get line data')
            inn, out = np.inf, -np.inf

        return inn, out
        return

    inn, out = np.inf, -np.inf
    line = None
    lines = ax.get_lines()
    if debug:
        print('lines=', lines, lines[0].__class__)

    for line in lines:
        if (str(type(line)) == '<class \'mpl_toolkits.mplot3d.art3d.Line3D\'>'):
            if debug:
                print('line=', line.get_label())
            new_inn, new_out = get_inn_out(line, inn, out)

            if (new_inn != np.inf) and (new_out != -np.inf):
                if new_inn < inn:
                    inn = new_inn
                if new_out > out:
                    out = new_out

            if debug:
                print('new_inn=', new_inn, 'new_out=', new_out)

    ax.set_ylim(inn, out)
    return


def align_yaxis(
        ax1,
        v1,
        ax2,
        v2,
        y2max):
    """ adjust ax2 ylimit so that v2 in
        ax2 is aligned to v1 in ax1.

        where y2max is the maximum value in your secondary plot. I haven't
        had a problem with minimum values being cut, so haven't set this. This
        approach doesn't necessarily make for axis limits at nice near units,
        but does optimist plot space """
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1 - y2))
    miny, maxy = ax2.get_ylim()

    scale = 1.
    while scale * (maxy + dy) < y2max * 1.02:
        scale += 0.07
    ax2.set_ylim(scale * (miny + dy), scale * (maxy + dy))
    return


def smoothing(
        y,
        window_length=100):
    """ smoothes any given line, mean of window_length points
    Args:
        y: input list, array
        window_length (int, optional): smoothing window width
    Returns:
        y: smoothed unput line with given length
    """
    y = np.convolve(
        y, np.ones((window_length, )) / window_length, mode='same')
    return y


def strahl_info_plot(
        time=[1.0],
        ne_maps=[Z((100)), Z((100))],
        ne_rad=[Z((100)), Z((100))],
        te_maps=[Z((100)), Z((100))],
        te_rad=[Z((100)), Z((100))],
        ne_smooth='spline',
        Te_smooth='spline'):
    """ support grid plot for STRAHL, carries, both the fits and original grids
    Keyword Arguments:
        time {list} -- list of times (default: {[1.0]})
        ne_maps {list} -- denstiy 2XspaceXtime (default: {[]})
        ne_rad {list} -- density space dim (default: {[]})
        te_maps {list} -- temperat. 2XspaceXtime (default: {[]})
        te_rad {list} -- temperature (default: {[]})
        ne_smooth {str} -- dens. smoothing methods (default: {'spline'})
        Te_smooth {str} --  temp. smoothing methods (default: {'spline'})
    Returns:
        None.
    """
    fig, axes = p.subplots(2, 1, sharex=True)
    if ne_smooth is None or Te_smooth is None:
        smooth = 'no_action'

    for j, t in enumerate(time):
        a, = axes[0].plot(
            ne_rad[0], ne_maps[0][j], ls='', marker='+',
            markersize=4., label='n$_{e, meas}$ ' + format(t, '.2f') + 's')
        b, = axes[0].plot(
            ne_rad[1], ne_maps[1][j], a.get_color(),
            ls='-.', label='n$_{e, ' + ne_smooth + '}$', alpha=1.)
        axes[0].text(
            0.8, ne_maps[0][j][-5],
            'n$_{e, LCFS}$=' + str(format(
                ne_maps[1][j][-1] * 1e19, '.2e')) + 'm$^{-3}$')

        c, = axes[1].plot(
            te_rad[0], te_maps[0][j], ls='', marker='+',
            markersize=4., label='T$_{e, meas}$ ' + format(t, '.2f') + 's')
        d, = axes[1].plot(
            te_rad[1], te_maps[1][j], c=c.get_color(),
            ls='-.', label='T$_{e, ' + Te_smooth + '}$', alpha=1.0)
        axes[1].text(
            0.8, te_maps[0][j][-5],
            'T$_{e, LCFS}$=' + str(format(
                te_maps[1][j][-1] * 1e3, '.2f')) + 'eV')

    axes[0].legend()
    axes[1].legend()
    axes[1].set_xlabel('$r_{eff}$')
    axes[1].set_ylabel('T$_{e}$ [keV]')
    axes[0].set_ylabel('n$_{e}$ [10$^{19}$m$^{-3}$]')

    fig.savefig('../results/STRAHL/grids/strahl_grid_' + format(
        time[-1], '.3f') + '.png', bbox_inches='tight', dpi=169.0)
    fig_current_save('strahl_support_grids', fig)
    p.close('all')
    return


def compare_strahl(
        program='20181010.032',
        files=None,  # ['C_00009t2.421_3.421_1'],
        material='C_',  # 'C_',
        strahl_ids=['00093', '00094'],  # 0005
        names=['f$_{rad}$=90%', 'f$_{rad}$=100%'],
        figx=7.,
        figy=3.,
        mode='edge',
        debug=False):
    """ wrapper to plot and compare multiple STRAHL routines with eachother
    Keyword Arguments:
        files {[type]} -- file strings (default: {None})
        material {[type]} -- species specifier  (default: {None})
        strahl_ids {[type]} -- STRAHL IDs [description] (default: {None})
        debug {bool} -- debugging (default: {False})
    Returns:
        None.
    """
    run_ids, results, r_majs, datas, powers, grids, volumes, labels = \
        [], [], [], [], [], [], [], []

    if (material is None) and (strahl_ids is None):
        for file in files:
            # run_id
            (data, grid_data, label, r_maj, el_symb,
             mass, density, number_density, result, run_id) = \
                rcc.scale_impurity_radiation(file=file)
            (power, volume) = rcc.corona_core_v_sol(
                grid_data=grid_data, data=data, r_maj=r_maj)

            run_ids.append(run_id)
            results.append(results)
            r_majs.append(r_maj)
            datas.append(data)
            powers.append(power)
            grids.append(np.nan_to_num(grid_data))
            volumes.append(volume)
            labels.append(label)

    elif isinstance(material, str) and isinstance(strahl_ids, list):
        for strahl_id in strahl_ids:
            # run_id1
            (data, grid_data, label, r_maj, el_symb,
             mass, density, number_density, result, run_id) = \
                rcc.scale_impurity_radiation(
                    material=material, strahl_id=strahl_id)
            (power, volume) = rcc.corona_core_v_sol(
                grid_data=grid_data, data=data, r_maj=r_maj)

            run_ids.append(run_id)
            results.append(result)
            r_majs.append(r_maj)
            datas.append(data)
            powers.append(power)
            grids.append(np.nan_to_num(grid_data))
            volumes.append(volume)
            labels.append(label)

    else:
        print('\\\ failed looking for file, error in mat/id/loc')
        return (None)

    if np.shape(run_ids)[0] <= 2:
        compare_strahl_concentration(
            run_ids=run_ids, results=results, r_maj=r_majs, names=names,
            grids=grids, labels=labels, mode=mode, figx=figx, figy=figy)
        compare_strahl_anomal_transp(
            results=results, grids=grids, figx=figx, figy=figy,
            run_ids=run_ids, r_maj=r_majs, mode=mode, names=names)
        compare_strahl_tot_rad(
            run_ids=run_ids, results=results, r_maj=r_majs, datas=datas,
            grids=grids, labels=labels, mode=mode, figx=figx, figy=1.1 * figy,
            names=names)
        compare_strahl_rad_ratios(
            run_ids=run_ids, results=results, r_maj=r_majs, datas=datas,
            grids=grids, labels=labels, mode=mode, figx=figx, figy=figy,
            names=names)
        compare_strahl_nete(
            run_ids=run_ids, results=results, r_maj=r_majs, names=names,
            grids=grids, mode=mode, program=program,
            figx=figx, figy=1.25 * figy)
        compare_fract_abund(
            run_ids=run_ids, results=results, r_maj=r_majs, names=names,
            grids=grids, labels=labels, mode=mode, figx=figx, figy=figy)

    if np.shape(run_ids)[0] >= 3:
        compare_core_sol_strahl(
            powers=powers, volumes=volume, labels=labels, r_maj=r_majs,
            datas=datas, grids=grids, mode=mode, run_ids=run_ids,
            figx=figx, figy=2. * figy, xticklabels=names)
        pass

    return


def compare_strahl_concentration(
        run_ids=[],
        results=[],
        r_maj=[],
        grids=[],
        labels=[],
        names=['f$_{rad}$=90%', 'f$_{rad}$=100%'],
        figx=5.,
        figy=5.,
        mode='edge'):
    """ plotting the STRAHL impurity concentrations
    Keyword Arguments:
        run_ids {list} -- run ids (default: {[]})
        r_maj {list} -- minor radius (default: {[]})
        volumes {list} -- integration volumes (default: {[]})
        grids {list} -- radial grids (default: {[]})
        mode {str} -- plot mode (default: {'edge'})
    """
    colors = cm.brg(np.linspace(0, 1, 7))
    styles = ['-', '-.', '--', ':']

    # anomal drift and diffusion
    f_concen, cont = p.subplots(1, 1)

    file = '../results/STRAHL/concentration/compare_concentration'
    for j, run_id in enumerate(run_ids):
        name = str(int(run_id[:5]))
        file += name + '_'

        el_dens = rcc.return_attribute(
            netcdf=results[j], var_name='electron_density')[0]
        imp_dens = rcc.return_attribute(
            netcdf=results[j], var_name='impurity_density')[0].transpose()
        tot_imp_dens = np.sum(imp_dens, axis=0)

        if j == 0:
            cont.plot(
                grids[j] * (r_maj[0] / 1000), tot_imp_dens / el_dens * 100,
                c='k', ls=styles[j],
                label='c$_{tot}$')  # + names[j])
        else:
            cont.plot(
                grids[j] * (r_maj[0] / 1000), tot_imp_dens / el_dens * 100,
                c='k', ls=styles[j])

    for j, run_id in enumerate(run_ids):
        name = str(int(run_id[:5]))
        file += name + '_'

        for i, dens in enumerate(imp_dens[:]):
            if j == 0:
                if i == 0:
                    cont.plot(
                        grids[j] * (r_maj[0] / 1000), dens / el_dens * 100,
                        c=colors[i], alpha=1., ls=styles[j],
                        label=labels[j][i][:4])
                else:
                    cont.plot(
                        grids[j] * (r_maj[0] / 1000), dens / el_dens * 100,
                        c=colors[i], alpha=1., ls=styles[j],
                        label=labels[j][i][:8])
            else:
                cont.plot(
                    grids[j] * (r_maj[0] / 1000), dens / el_dens * 100,
                    c=colors[i], alpha=1., ls=styles[j])

    cont.set_ylabel('c$_{i}$ [%]')
    # fractional abundance
    for i, ax in enumerate([cont]):
        ax.legend(
            loc='upper center', bbox_to_anchor=(.5, 1.35),
            ncol=4, fancybox=True, shadow=False,
            handletextpad=0.1, labelspacing=0.2, handlelength=1.)

        ax.set_xlabel('$r_{eff}$ [m]')
        if mode == 'full':
            ax.set_xlim(0.0, np.max(grids[0] * (r_maj[0] / 1000)))
        elif mode == 'edge':
            ax.set_xlim(0.8 * (r_maj[0] / 1000),
                        np.max(grids[0] * (r_maj[0] / 1000)))
        ax.axvline(r_maj[0] / 1000, ls='--', lw=2.0, c='grey', alpha=0.5)

    # SAVING
    # fractional abundance
    text = '\n'.join((  # textstring
        names[0] + ': $-$',
        names[1] + ': $-\\cdot-\\cdot$'))
    # build box
    prop = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(
        0.05, 0.6, text, transform=ax.transAxes,
        verticalalignment='top', bbox=prop)

    f_concen.set_size_inches(figx, figy)
    fig_current_save('compare_strahl_concentration', f_concen)
    f_concen.savefig(file + mode + '.png', bbox_inches='tight', dpi=169.0)
    p.close('all')
    return


def compare_strahl_anomal_transp(
        run_ids=[],
        results=[],
        grids=[],
        r_maj=[],
        names=['f$_{rad}$=90%', 'f$_{rad}$=100%'],
        figx=5.,
        figy=5.,
        mode='edge'):
    """ transport info comparison for strahl runs
    Keyword Arguments:
        run_ids {list} -- strahl ids (default: {[]})
        r_maj {list} -- major radius list [description] (default: {[]})
        mode {str} -- plot window mode [description] (default: {'edge'})
    Returns:
        None.
    """
    colors = ['r', 'b']  # cm.brg(np.linspace(0, 1, 2))
    styles = ['-', '-.', '--', ':']

    # anomal drift and diffusion
    anomal_transp = p.figure()
    an_drift = anomal_transp.add_subplot(111)
    an_diff = an_drift.twinx()

    file = '../results/STRAHL/transport/compare_anomal_transp_'
    for j, run_id in enumerate(run_ids):
        name = str(int(run_id[:5]))
        file += name + '_'

        if j == 0:
            # anomal drift and diffusion
            drift, = an_drift.plot(
                grids[j] * (r_maj[j] / 1000),
                rcc.return_attribute(netcdf=results[j],
                                     var_name='anomal_drift')[0] / 1e4,
                c=colors[0], alpha=1., ls=styles[j],
                label='drift')  # names[j])
            diff, = an_diff.plot(
                grids[j] * (r_maj[j] / 1000),
                rcc.return_attribute(
                    netcdf=results[j], var_name='anomal_diffusion')[0] / 1e4,
                c=colors[1], ls=styles[j],
                label='diffusion')  # names[j])

        else:
            an_drift.plot(
                grids[j] * (r_maj[j] / 1000),
                rcc.return_attribute(
                    netcdf=results[j], var_name='anomal_drift')[0] / 1e4,
                c=colors[0], alpha=1., ls=styles[j])
            an_diff.plot(
                grids[j] * (r_maj[j] / 1000),
                rcc.return_attribute(
                    netcdf=results[j], var_name='anomal_diffusion')[0] / 1e4,
                c=colors[1], ls=styles[j])

    lines = [drift, diff]
    an_drift.legend(
        lines, [l.get_label() for l in lines],
        loc='upper center', bbox_to_anchor=(.5, 1.2),
        ncol=4, fancybox=True, shadow=False,
        handletextpad=0.1, labelspacing=0.2, handlelength=1.)
    # anomal diffusion and drift
    an_diff.set_ylabel('diffusion [m$^{2}$s$^{-1}$]')
    an_drift.set_ylabel('drift [m$^{2}$s$^{-1}$ / ms$^{-1}$ / $\\alpha$]')

    # total radiation and anomal diffusion
    for i, ax in enumerate([an_drift, an_diff]):
        ax.set_xlabel('$r_{eff}$ [m]')
        if mode == 'full':
            ax.set_xlim(0.0, np.max(grids[j] * (r_maj[0] / 1000)))
        elif mode == 'edge':
            ax.set_xlim(0.8 * (r_maj[0] / 1000),
                        np.max(grids[j] * (r_maj[0] / 1000)))
        ax.axvline(r_maj[0] / 1000, ls='--', lw=2.0, c='grey', alpha=0.5)

    text = '\n'.join((  # textstring
        names[0] + ': $-$',
        names[1] + ': $-\\cdot-\\cdot$'))
    # build box
    prop = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(
        0.5, 0.35, text, transform=ax.transAxes,
        verticalalignment='top', bbox=prop)

    # SAVING
    # anomal drift and diffusion
    anomal_transp.set_size_inches(figx, figy)
    fig_current_save('compare_anomal_transp', anomal_transp)
    anomal_transp.savefig(file + mode + '.png', bbox_inches='tight', dpi=169.0)
    p.close('all')
    return


def compare_strahl_tot_rad(
        run_ids=[],
        results=[],
        r_maj=[],
        datas=[],
        grids=[],
        labels=[],
        names=['f$_{rad}$=90%', 'f$_{rad}$=100%'],
        figx=5.,
        figy=5.,
        mode='edge'):
    """ comparing total strahl radiation and line radiation from ions
    Keyword Arguments:
        run_ids {list} -- list of run ids (default: {[]})
        results {list} -- result objects list (default: {[]})
        r_maj {list} -- major radius list (default: {[]})
        datas {list} -- data from radiation calculation (default: {[]})
        grids {list} -- grid data list (default: {[]})
        labels {list} -- plot labels with vol/line int (default: {[]})
        mode {str} -- plot window mode (default: {'edge'})
    Return:
        None.
    """
    colors = cm.brg(np.linspace(0, 1, 7))
    styles = ['-', '-.', '--', ':']

    # figure
    # line and total radiation
    p_tot_fig = p.figure()
    p_line = p_tot_fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[])
    p_total = p_tot_fig.add_axes([0.1, 0.1, 0.8, 0.4])

    N = np.shape(run_ids)[0]
    K, file = [], '../results/STRAHL/diag_lines/compare_strahl_rad_'
    for j, run_id in enumerate(run_ids):
        name = str(int(run_id[:5]))
        file += name + '_'

        # total radiation
        if np.shape(run_ids)[0] < 2:
            line, = p_total.plot(
                grids[j] * (r_maj[j] / 1000), datas[j][:, 9] / 1e3,
                alpha=1., c='k', ls=styles[j],
                label=labels[j][9] + ' ' + names[j])
        else:
            line, = p_total.plot(
                grids[j] * (r_maj[j] / 1000), datas[j][:, 9] / 1e3,
                alpha=1., c='k', ls=styles[j],
                label='total')  # 'total rad. ' + names[j])
        if j == 0:
            K.append(line)

        # total and line radiation
        for i, imp in enumerate(np.transpose(datas[j][:, :7])):
            if i == 0:
                line, = p_line.plot(
                    grids[j] * (r_maj[j] / 1000),
                    imp / 1e3, c=colors[i], alpha=1.,
                    ls=styles[j], label=labels[j][i][:4])
            else:
                line, = p_line.plot(
                    grids[j] * (r_maj[j] / 1000),
                    imp / 1e3, c=colors[i], alpha=1.,
                    ls=styles[j], label=labels[j][i][:8])
            if j == 0:
                K.append(line)

    # total and line radiation
    p_line.legend(
        K, [l.get_label() for l in K],
        loc='upper center', bbox_to_anchor=(.5, 1.45),
        ncol=4, fancybox=True, shadow=False,
        handletextpad=0.1, labelspacing=0.2, handlelength=1.)

    for ax in [p_line, p_total]:
        ax.set_xlabel('$r_{eff}$ [m]')
    p_total.set_ylabel('P$_{tot} $ [kWm$^{-3}$]')
    p_line.set_ylabel('P$_{diag} $ [kWm$^{-3}$]')

    # total radiation and anomal diffusion
    for i, ax in enumerate([p_line, p_total]):
        if mode == 'full':
            ax.set_xlim(0.0, np.max(grids[j] * (r_maj[j] / 1000)))
        elif mode == 'edge':
            ax.set_xlim(0.8 * (r_maj[j] / 1000),
                        np.max(grids[j] * (r_maj[j] / 1000)))
        # autoscale_y(ax)
        ax.axvline(r_maj[j] / 1000, ls='--', lw=2.0, c='grey', alpha=0.5)

    text = '\n'.join((  # textstring
        names[0] + ': $-$',
        names[1] + ': $-\\cdot-\\cdot$'))
    # build box
    prop = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(
        0.05, 0.9, text, transform=p_total.transAxes,
        verticalalignment='top', bbox=prop)

    # SAVING
    # total and line radiation
    p_tot_fig.set_size_inches(figx, figy)
    fig_current_save('compare_strahl_rad', p_tot_fig)
    p_tot_fig.savefig(file + mode + '.png', bbox_inches='tight', dpi=169.0)
    p.close('all')
    return


def compare_strahl_rad_ratios(
        run_ids=[],
        results=[],
        r_maj=[],
        datas=[],
        grids=[],
        labels=[],
        names=['f$_{rad}$=90%', 'f$_{rad}$=100%'],
        figx=5.,
        figy=5.,
        mode='edge'):
    """ comparing strahl line radiation ratios
    Keyword Arguments:
        run_ids {list} -- list of run ids (default: {[]})
        results {list} -- result objects list (default: {[]})
        r_maj {list} -- major radius list (default: {[]})
        datas {list} -- data from radiation calculation (default: {[]})
        grids {list} -- grid data list (default: {[]})
        labels {list} -- plot labels with vol/line int (default: {[]})
        mode {str} -- plot window mode (default: {'edge'})
    Return:
        None.
    """
    colors = cm.brg(np.linspace(0, 1, 7))
    styles = ['-', '-.', '--', ':']

    # figure
    # line and total radiation
    p_fig, p_line = p.subplots(1, 1)
    N = np.shape(run_ids)[0]

    L, file = [], '../results/STRAHL/diag_lines/' + \
        'compare_strahl_rad_ratios'
    for j, run_id in enumerate(run_ids):
        name = str(int(run_id[:5]))
        file += name + '_'

        # total and line radiation
        for i, imp in enumerate(np.transpose(datas[j][:, :7])):
            if i == 0:
                line, = p_line.plot(
                    grids[j] * (r_maj[j] / 1000),
                    (imp / datas[j][:, 9]),
                    c=colors[i], alpha=1., ls=styles[j],
                    label=labels[j][i][:4])
            else:
                line, = p_line.plot(
                    grids[j] * (r_maj[j] / 1000),
                    (imp / datas[j][:, 9]),
                    c=colors[i], alpha=1., ls=styles[j],
                    label=labels[j][i][:8])

            if j == 0:
                L.append(line)

    p_line.legend(
        L, [l.get_label() for l in L],
        loc='upper center', bbox_to_anchor=(.5, 1.35),
        ncol=4, fancybox=True, shadow=False,
        handletextpad=0.1, labelspacing=0.2, handlelength=1.)
    # total and line radiation
    p_line.set_xlabel('$r_{eff}$ [m]')
    p_line.set_ylabel('P$_{i}$ [100%]')

    # total radiation and anomal diffusion
    if mode == 'full':
        p_line.set_xlim(0.0, np.max(grids[j] * (r_maj[j] / 1000)))
    elif mode == 'edge':
        p_line.set_xlim(0.8 * (r_maj[j] / 1000),
                        np.max(grids[j] * (r_maj[j] / 1000)))
    # autoscale_y(ax)
    p_line.axvline(r_maj[j] / 1000, ls='--', lw=2.0, c='grey', alpha=0.5)

    text = '\n'.join((  # textstring
        names[0] + ': $-$',
        names[1] + ': $-\\cdot-\\cdot$'))
    # build box
    prop = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
    # place a text box in upper left in axes coords
    p_line.text(
        0.05, 0.6, text, transform=p_line.transAxes,
        verticalalignment='top', bbox=prop)

    # SAVING
    # total and line radiation
    p_fig.set_size_inches(figx, figy)
    fig_current_save('compare_strahl_rad_ratios', p_fig)
    p_fig.savefig(file + mode + '.png', bbox_inches='tight', dpi=169.0)
    p.close('all')
    return


def compare_strahl_nete(
        program='20181010.032',
        run_ids=[],
        results=[],
        r_maj=[],
        grids=[],
        names=['f$_{rad}$=90%', 'f$_{rad}$=100%'],
        figx=5.,
        figy=5.,
        mode='edge'):
    """ comparing strahl ne and Te profiles
    Keyword Arguments:
        program {str} -- program modelled after (def '20181010.032')
        run_ids {list} -- strahl ids to read from (def {[]}})
        results {list} -- result objects data (def {[]})
        r_maj {list} -- major radius list (def {[]})
        grids {list} -- grids from runs (def {[]})
        mode {str} -- plot mode, x window (def {'edge'})
    Returns:
        None.
    """
    file = '../results/STRAHL/nete/compare_ne_Te_'
    colors = ['r', 'b']  # cm.brg(np.linspace(0, 1, 6))
    styles = ['-', '-.', '--', ':']
    markers = ['+', '^', 'v', 's', 'o']

    # anomal drift and diffusion
    ne_Te = p.figure()
    n = ne_Te.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[])
    T = ne_Te.add_axes([0.1, 0.1, 0.8, 0.4])

    time = []
    for result in results:
        time.append(rcc.return_attribute(
            netcdf=result, var_name='time')[0][-1])

    TS_profile, time = TS.return_TS_profile_for_t(
        shotno=program, scaling='gauss', t=time)
    if TS_profile is None:
        print('\t// failed TS profile')
        return

    ne_rad = [
        x for i, x in enumerate(TS_profile[0][
            'r without outliers (ne)']) if
        ((x > 0.0) and(x - TS_profile[0][
            'r without outliers (ne)'][i - 1] > 0.01))]
    Te_rad = [
        x for i, x in enumerate(TS_profile[0][
            'r without outliers (Te)']) if
        ((x > 0.0) and (x - TS_profile[0][
            'r without outliers (Te)'][i - 1] > 0.01))]

    lines = []
    for k, t in enumerate(time):
        ne = [  # 1e19
            TS_profile[k]['ne map without outliers'][i] / 10 for i, x in
            enumerate(TS_profile[k]['r without outliers (ne)']) if
            ((x > 0.0) and (x - TS_profile[k][
                'r without outliers (ne)'][i - 1] > 0.01))]
        Te = [
            TS_profile[k]['Te map without outliers'][i] for i, x in
            enumerate(TS_profile[k]['r without outliers (Te)'])
            if ((x > 0.0) and (x - TS_profile[k][
                'r without outliers (Te)'][i - 1] > 0.01))]

        n_p, = n.plot(
            ne_rad, ne, marker=markers[k], c=colors[k], markersize=4.,
            ls='', alpha=1., label=names[k])
        T_p, = T.plot(
            Te_rad, Te, marker=markers[k], c=colors[k], markersize=4.,
            ls='', alpha=1.)
        lines.append(n_p)

    n_sep_string, T_sep_string = [], []
    for j, run_id in enumerate(run_ids):
        name = str(int(run_id[:5]))
        file += name + '_'

        # check function of data in pp file
        interp_method = rcc.sort_param_file(
            base='nete/', file='pp' + run_id[:5] + '.0',
            attribute='ne function', dtype='str', debug=False)[0]

        # temperature from file and results
        if interp_method in ['interp', 'interpa']:
            # density from file and results
            [file_n, file_grid] = rcc.sort_param_file(
                base='nete/', file='pp' + run_id[:5] + '.0',
                attribute='ne data', time='time-vector', grid='x-grid')[0]

        elif interp_method == 'ratfun':
            y0, p0, p1, p2 = rcc.sort_param_file(
                base='nete/', file='pp' + run_id[:5] + '.0', debug=False,
                attribute='ne ratfun values', time='time-vector')[0][0:4]
            file_grid = grids[j]
            file_n = y0 * ((1 - p0) * (1 - file_grid**p1)**p2 + p0)
            file_n = np.insert(file_n, 0, file_n[0])
            file_n[1:] = file_n[1:] / file_n[0]

        idx = mClass.find_nearest(grids[j], 1.0)[0]
        n_lcfs = rcc.return_attribute(
            netcdf=results[j], var_name='electron_density')[0][idx] * 1e-14
        n_sep_string.append(format(n_lcfs, '.2f') + 'x10$^{19}$m$^{-3}$')

        n_p, = n.plot(
            grids[j] * (r_maj[j] / 1000),
            rcc.return_attribute(
                netcdf=results[j], var_name='electron_density')[0] * 1e-14,
            c=colors[j], alpha=1., ls='-.', label='interp.')
        lines.append(n_p)

        # check function of data in pp file
        interp_method = rcc.sort_param_file(
            base='nete/', file='pp' + run_id[:5] + '.0',
            attribute='Te function', dtype='str', debug=False)[0]
        # temperature from file and results
        if interp_method in ['interp', 'interpa']:
            [file_T, file_grid] = rcc.sort_param_file(
                base='nete/', file='pp' + run_id[:5] + '.0', debug=False,
                attribute='Te data', time='time-vector', grid='x-grid')[0]

        elif interp_method == 'ratfun':
            y0, p0, p1, p2 = rcc.sort_param_file(
                base='nete/', file='pp' + run_id[:5] + '.0', debug=False,
                attribute='Te ratfun values', time='time-vector')[0][0:4]
            file_grid = grids[j]
            file_T = y0 * ((1 - p0) * (1 - file_grid**p1)**p2 + p0)
            file_T = np.insert(file_T, 0, file_T[0])
            file_T[1:] = file_T[1:] / file_T[0]

        idx = mClass.find_nearest(grids[j], 1.0)[0]
        T_lcfs = rcc.return_attribute(
            netcdf=results[j], var_name='electron_temperature')[0][idx]
        T_sep_string.append(format(T_lcfs, '.2f') + 'eV')

        T_p, = T.plot(
            grids[j] * (r_maj[j] / 1000),
            1e-3 * rcc.return_attribute(
                netcdf=results[j], var_name='electron_temperature')[0],
            c=colors[j], alpha=1., ls='-.')  # label='T$_{e, sim}$')

    n.set_ylabel('n$_{e}$ [10$^{20}$m$^{-3}$]')
    T.set_ylabel('T$_{e}$ [keV]')

    n_string = '\n'.join((  # textstring
        'separatrix:',
        names[0] + ' :' + n_sep_string[0],
        names[1] + ' :' + n_sep_string[1]))
    T_string = '\n'.join((  # textstring
        'separatrix:',
        names[0] + ' :' + T_sep_string[0],
        names[1] + ' :' + T_sep_string[1]))

    n.legend(
        lines, [l.get_label() for l in lines],
        loc='upper center', bbox_to_anchor=(.5, 1.25),
        ncol=4, fancybox=True, shadow=False,
        handletextpad=0.1, labelspacing=0.2, handlelength=1.)
    # build box
    prop = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
    for ax, text, pos in zip(
            [n, T], [n_string, T_string], [[.05, .4], [0.5, 0.9]]):
        # place a text box in upper left in axes coords
        ax.text(
            pos[0], pos[1], text, transform=ax.transAxes,
            verticalalignment='top', bbox=prop)

    # electron temperature and density radiation and anomal diffusion
    for i, ax in enumerate([n, T]):
        ax.set_xlabel('$r_{eff}$ [m]')
        if mode == 'full':
            ax.set_xlim(0.0, np.max(grids[0] * (r_maj[0] / 1000)))
        elif mode == 'edge':
            ax.set_xlim(0.8 * (r_maj[0] / 1000),
                        np.max(grids[0] * (r_maj[0] / 1000)))
        autoscale_y(ax)
        ax.axvline(r_maj[0] / 1000, ls='--', lw=2.0, c='grey', alpha=0.5)

    # SAVING
    # eletron temperature and density
    ne_Te.set_size_inches(figx, figy)
    fig_current_save('compare_strahl_ne_Te', ne_Te)
    ne_Te.savefig(file + mode + '.png', bbox_inches='tight', dpi=169.0)
    p.close('all')
    return


def compare_fract_abund(
        run_ids=[],
        results=[],
        r_maj=[],
        grids=[],
        labels=[],
        names=['f$_{rad}$=90%', 'f$_{rad}$=100%'],
        figx=5.,
        figy=5.,
        mode='edge'):
    """ plotting the STRAHL fractional abundances
    Keyword Arguments:
        run_ids {list} -- run ids (default: {[]})
        r_maj {list} -- minor radius (default: {[]})
        volumes {list} -- integration volumes (default: {[]})
        grids {list} -- radial grids (default: {[]})
        labels {list} -- plot labels (default: {[]})
        mode {str} -- plot mode (default: {'edge'})
    """
    colors = cm.brg(np.linspace(0, 1, 7))
    styles = ['-', '-.', '--', ':']

    # anomal drift and diffusion
    f_abund = p.figure()
    fract = f_abund.add_subplot(111)  # 211)
    # tot_dens = f_abund.add_subplot(212)

    file = '../results/STRAHL/fract_abund/compare_fract_abund_'
    for j, run_id in enumerate(run_ids):
        name = str(int(run_id[:5]))
        file += name + '_'

        imp_dens = rcc.return_attribute(
            netcdf=results[j], var_name='impurity_density')[0].transpose()
        tot_imp_dens = np.sum(imp_dens, axis=0)
        # tot_dens.plot(
        #     grids[j] * (r_maj[0] / 1000), tot_imp_dens / 1e8,
        #     c='k', ls=styles[j],
        #     label='n$_{i, tot}$ ' + names[j])

        for i, dens in enumerate(imp_dens[:]):
            if (i == 0) and (j == 0):
                fract.plot(
                    grids[j] * (r_maj[0] / 1000), dens / tot_imp_dens,
                    c=colors[i], alpha=1., ls=styles[j],
                    label=labels[j][i][:4])
            elif j == 0:
                fract.plot(
                    grids[j] * (r_maj[0] / 1000), dens / tot_imp_dens,
                    c=colors[i], alpha=1., ls=styles[j],
                    label=labels[j][i][:8])
            else:
                fract.plot(
                    grids[j] * (r_maj[0] / 1000), dens / tot_imp_dens,
                    c=colors[i], alpha=1., ls=styles[j])

    fract.set_ylabel('f$_{A}$ [%]')
    # tot_dens.set_ylabel('n$_{i, tot}$ [10$^{14}$m$^{-3}$]')
    fract.legend(
        loc='upper center', bbox_to_anchor=(.5, 1.4),
        ncol=4, fancybox=True, shadow=False,
        handletextpad=0.1, labelspacing=0.2, handlelength=1.)

    # fractional abundance
    for i, ax in enumerate([fract]):  # tot_dens]):
        ax.set_xlabel('$r_{eff}$ [m]')
        if mode == 'full':
            ax.set_xlim(0.0, np.max(grids[0] * (r_maj[0] / 1000)))
        elif mode == 'edge':
            ax.set_xlim(0.8 * (r_maj[0] / 1000),
                        np.max(grids[0] * (r_maj[0] / 1000)))
        # autoscale_y(ax)
        ax.axvline(r_maj[0] / 1000, ls='--', lw=2.0, c='grey', alpha=.5)

    text = '\n'.join((  # textstring
        names[0] + ': $-$',
        names[1] + ': $-\\cdot-\\cdot$'))
    # build box
    prop = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
    # place a text box in upper left in axes coords
    fract.text(
        0.05, 0.6, text, transform=fract.transAxes,
        verticalalignment='top', bbox=prop)

    # SAVING
    # fractional abundance
    f_abund.set_size_inches(figx, figy)
    fig_current_save('compare_strahl_fract_abund', f_abund)
    f_abund.savefig(file + mode + '.png', bbox_inches='tight', dpi=169.0)
    p.close('all')
    return


def compare_core_sol_strahl(
        run_ids=[],
        r_maj=[],
        powers=[],
        volumes=[],
        datas=[],
        grids=[],
        labels=[],
        greyed_out=[0.0, 2.0],
        xticklabels=['33%', '66%', '90%', '100%'],
        xlabel='f$_{rad}$',
        figx=5.,
        figy=5.,
        mode='edge'):
    """ plotting the STRAHL core v SOL ratios
    Keyword Arguments:
        run_ids {list} -- run ids (default: {[]})
        r_maj {list} -- minor radius (default: {[]})
        powers {list} -- powers from SOL/core (default: {[]})
        volumes {list} -- integration volumes (default: {[]})
        datas {list} -- radiation data (default: {[]})
        grids {list} -- radial grids (default: {[]})
        labels {list} -- plot labels (default: {[]})
        mode {str} -- plot mode (default: {'edge'})
    """
    styles, markersc = ['-', '--', ':', '-.'], cycle(['+', '^', 's'])
    colors = cm.brg(np.linspace(0, 1, np.shape(powers[0])[1]))

    ratios = p.figure()
    axes = []
    axes.append(ratios.add_axes([0.1, 0.612, 0.8, 0.28], xticklabels=[]))
    axes.append(ratios.add_axes([0.1, 0.331, 0.8, 0.28], xticklabels=[]))
    axes.append(ratios.add_axes([0.1, 0.05, 0.8, 0.28]))

    names, file = [], '../results/STRAHL/core_v_sol/compare_strahl_corevsol'
    lines = Z((2, np.shape(run_ids)[0], np.shape(powers[0])[1]))
    for k in range(np.shape(powers[0])[1]):
        for j, run_id in enumerate(run_ids):
            if k == 0:
                file += str(int(run_id[:5])) + '_'
            lines[0, j, k] = powers[j][2, k]
            lines[1, j, k] = powers[j][3, k]
            names.append(run_ids[j][0:5])

    # ionisation stages
    for k in range(np.shape(powers[0][:, :7])[1]):
        if (k == 0):
            plot, = axes[0].plot(
                lines[0, :, k], marker='o', alpha=1., ls='-.',
                label=labels[0][k][:4], c=colors[k])
        else:
            plot, = axes[0].plot(
                lines[0, :, k], marker='o', alpha=1., ls='-.',
                label=labels[0][k][:8], c=colors[k])
        axes[1].plot(
            lines[1, :, k], ls='-.', marker='o',
            c=plot.get_color(), alpha=1.)

    # total radiation
    axes[2].plot(lines[0, :, 9], c='k', alpha=1.,
                 marker=next(markersc), label='P$_{core}$')
    axes[2].plot(lines[1, :, 9], c='k', alpha=1.,
                 marker=next(markersc), ls='-.', label='P$_{SOL}$')

    # legends
    axes[0].legend(
        loc='upper center', bbox_to_anchor=(.5, 1.4),
        ncol=4, fancybox=True, shadow=False,
        handletextpad=0.1, labelspacing=0.2, handlelength=1.)
    axes[2].legend()

    # labels
    axes[0].set_ylabel('P$_{core}$ [100%]')
    axes[1].set_ylabel('P$_{SOL}$ [100%]')
    axes[2].set_ylabel('P$_{tot}$ [100%]')
    # axes[2].set_xlabel(xlabel)
    p.xticks(np.linspace(
        0, np.shape(run_ids)[0] - 1, np.shape(run_ids)[0]), xticklabels)

    for ax in axes:
        ax.axvspan(
            greyed_out[0] - 0.1, greyed_out[1] + 0.1,
            facecolor='grey', alpha=0.25)

    # saving
    ratios.set_size_inches(figx, figy)
    fig_current_save('compare_strahl_corevsol', ratios)
    ratios.savefig(file + '_' + mode + '.png', bbox_inches='tight', dpi=169.0)
    p.close('all')
    return


def hexos_and_channels(
        program='20181010.032',
        material='C',
        eChannels=[4, 7, 15, 24, 28],
        times=[0.1, 0.2],
        hexos_dat={'none': None},
        channel_data=[None],
        species=['C_II', 'C_IV', 'C_V', 'C_VI'],
        photon_energies=None,
        debug=False):
    """ HEXOS lines and channel comparisons
    Args:
        program (str, optional):
        material (str, optional):
        eChannels (list, optional):
        times (list, optional):
        hexos_dat (dict, optional):
        channel_data (list, optional):
        species (list, optional):
        photon_energies (None, optional):
        debug (bool, optional):
    Returns:
        None
    """
    t1 = archivedb.get_program_t1(program)
    if debug:
        print('>> plotting HEXOS:', material, 'and channels:', eChannels)

    fig = p.figure()
    ax = fig.add_axes([0.1, 0.6, 0.8, 0.4], xticklabels=[])
    ax2 = fig.add_axes([0.1, 0.15, 0.8, 0.4])

    if times is not None:
        ax.set_xlim(times[0], times[1])
        ax2.set_xlim(times[0], times[1])
    else:
        t5 = archivedb.get_program_from_to(program)[1]
        times = [-0.1, (t5 - t1) / 1e9]
        ax.set_xlim(times[0], times[1])
        ax2.set_xlim(times[0], times[1])

    if species is None:
        species = hexos_dat['values'].keys()

    nHexos, nChans = len(species), len(eChannels)
    cHexos = cycle(cm.brg(np.linspace(0, 1, nHexos)))
    cChans = cycle(cm.brg(np.linspace(0, 1, nChans)))

    for i, key in enumerate(species):
        ShP, = ax.plot(
            [x * 1e-3 for x in hexos_dat['values'][key][0]],
            smoothing([y * 1e-4 for y in hexos_dat['values'][key][1]], 10),
            label=key.replace('_', ' '), linestyle='-.',
            c=next(cHexos), linewidth=0.6)

        if photon_energies is not None:
            ax.text(times[0] + 0.2, np.mean([
                    y * 1e-4 for y in hexos_dat['values'][key][1]]),
                    'E$_{P}$~' + str(round(photon_energies[i], 3)) + 'eV',
                    color=ShP.get_color())

    autoscale_y(ax)
    ax.legend()
    ax.set_ylabel('non rel. intensity [a.u.]')
    ax.set_title('#' + program)

    T = [(t - t1) / 1e9 for t in channel_data['dimensions']]
    for ch in eChannels:
        ax2.plot(T, [v * 1e6 for v in channel_data['values'][ch]],
                 label='eCh#' + str(ch), c=next(cChans), linewidth=.6,
                 linestyle='-.')
    ax2.legend()
    autoscale_y(ax2)
    ax2.set_ylabel('P [$\\mu$W]')
    ax2.set_xlabel('t [s]')

    fig_current_save('hexos_and_channels', fig)
    file = "../results/HEXOS/" + program[0:8]
    if not os.path.exists(file):
        os.makedirs(file)

    fig.savefig(
        file + '/hexos_and_channels_' + program.replace('.', '_') + '_' +
        str(times).replace(' ', '').replace('.', '_') + str(species).replace(
            ' ', '').replace('.', '_') + '.png')
    p.close('all')

    return


def plot_radFraction(
        program='20181010.032',
        vmecID='1000_1000_1000_1000_+0000_+0000/01/00jh_l/',
        ECRH={},
        Prad={},
        rad_fraction=Z((10000)),
        avg_TS_datas=None,
        TS_data={'none': None},
        fracs=None,
        times=None,
        debug=False,
        species=None,
        photon_energies=None):
    """ plotting radiational fraction and relating TS profiles
    Args:
        program (str, optional):
        ECRH (dict, optional):
        Prad (dict, optional):
        rad_fraction (TYPE, optional):
        avg_TS_datas (None, optional):
        TS_data (dict, optional):
        fracs (list, optional):
        times (list, optional):
        debug (bool, optional):
        species (list, optional):
        photon_energies (list, optional):
    Returns:
        None
    """
    file = "../results/RAD_FRAC/" + program[0:8]
    if not os.path.exists(file):
        os.makedirs(file)

    m_R = requests.get(
        'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/' +
        vmecID + 'minorradius.json').json()['minorRadius']

    if debug:
        print('>> plotting radiational fraction:', end=' ')
    tfin = (archivedb.get_program_from_to(
        program)[1] - archivedb.get_program_t1(
        program)) / 1e9

    fig = p.figure()

    if (avg_TS_datas is None) and (TS_data is None):
        ax = fig.add_subplot(111)
    elif fracs is None:
        ax = fig.add_subplot(211)
    else:
        ax = fig.add_axes([0.1, 0.75, 0.8, 0.25])

    if debug:
        print('rad. frac.,', end=' ')

    ax2 = ax.twinx()
    pecrh, = ax.plot(ECRH['dimensions'], np.array(ECRH['values']) * 1e-3,
                     c='k', ls='-.', label='P$_{ECRH}$')
    pprad, = ax.plot(Prad['dimensions'], np.array(Prad['values']) * 1e-6,
                     c='r', ls='-.', label='P$_{rad,HBCm}$')
    pfrad, = ax2.plot(Prad['dimensions'], rad_fraction,
                      c='purple', ls='-', label='f$_{rad}$')

    ax.set_ylabel('P [MW]')
    ax.set_xlabel('t [s]')
    ax2.set_ylabel('rad. fraction')

    ax.set_xlim(-0.1, tfin)
    ax.legend([pecrh, pfrad, pprad],
              [l.get_label() for l in [pecrh, pfrad, pprad]])

    if (fracs is not None) and \
            (TS_data is not None) and (avg_TS_datas is not None):
        colors = cm.brg(np.linspace(0, 1, len(times)))
        ax3 = fig.add_axes([0.1, 0.3, 0.8, 0.3], xticklabels=[])
        ax4 = fig.add_axes([0.1, 0.0, 0.8, 0.3])

        for i, t in enumerate(times):
            ax2.axvline(t, c='grey', alpha=0.5, ls='-.', lw=2.)
            ax2.text(t + 0.005, 0.2 * i, 'f$_{rad}$~' + str(fracs[i]))

        for i, g in enumerate(fracs):
            if avg_TS_datas is not None:
                if debug:
                    print('avg. TS', times[i], ',', end=' ')

                M = len(avg_TS_datas[i]['time'])
                neP, = ax3.plot(
                    avg_TS_datas[i]['values']['n_e']['bin r ne'],
                    avg_TS_datas[i]['values']['n_e']['ne map'] / M,
                    marker='x', markersize=4.0, ls='--', lw=0.5,
                    c=next(colors), alpha=1.,
                    label='f$_{rad}$~' + str(fracs[i]))
                teP, = ax4.plot(
                    avg_TS_datas[i]['values']['T_e']['bin r te'],
                    avg_TS_datas[i]['values']['T_e']['Te map'] / M,
                    marker='x', markersize=4.0, ls='--', lw=0.5,
                    c=neP.get_color(), alpha=1.,
                    label='f$_{rad}$~' + str(fracs[i]))

            elif TS_data is not None:
                if debug:
                    print('TS data', times[i], ',', end=' ')

                k = mClass.find_nearest(
                    TS_data['time'], times[i] * 1e9 +
                    archivedb.get_program_t1(program))[0]

                # ne error bars
                ne_high97 = [v - TS_data['values']['n_e'][
                    'ne map without outliers'][k][j] for j, v in enumerate(
                        TS_data['values']['n_e'][
                            'ne high97 without outliers'][k])]
                ne_low97 = [TS_data['values']['n_e'][
                    'ne map without outliers'][k][j] - v for j, v in
                    enumerate(TS_data['values']['n_e'][
                        'ne low97 without outliers'][k])]

                # Te error bars
                te_high97 = [v - TS_data['values']['T_e'][
                    'Te map without outliers'][k][j] for j, v in enumerate(
                        TS_data['values']['T_e'][
                            'Te high97 without outliers'][k])]
                te_low97 = [TS_data['values']['T_e'][
                    'Te map without outliers'][k][j] - v for j, v in
                    enumerate(TS_data['values']['T_e'][
                        'Te low97 without outliers'][k])]

                ax3.errorbar(
                    TS_data['values']['n_e']['r without outliers (ne)'][k],
                    smoothing(TS_data['values']['n_e'][
                        'ne map without outliers'][k], window_length=3),
                    yerr=[ne_low97, ne_high97], fmt='--x')

                neP = ax3.errorbar(
                    TS_data['values']['n_e']['r without outliers (ne)'][k],
                    smoothing(TS_data['values']['n_e'][
                        'ne map without outliers'][k], window_length=3),
                    yerr=[ne_low97, ne_high97], fmt='--x',
                    c=colors[i], alpha=1.,
                    label='f$_{rad}$~' + str(fracs[i]))
                teP = ax4.errorbar(
                    TS_data['values']['T_e']['r without outliers (Te)'][k],
                    smoothing(TS_data['values']['T_e'][
                        'Te map without outliers'][k], window_length=3),
                    yerr=[te_low97, te_high97], fmt='--x',
                    c=colors[i], alpha=1.,
                    label='f$_{rad}$~' + str(fracs[i]))

        ax3.set_ylabel('n$_{e}$ [10$^{19}$ m$^{-3}$]')
        ax4.set_ylabel('T$_{e}$ [keV]')
        ax4.set_xlabel('r$_{eff}$ [m]')

        for ax in [ax3, ax4]:
            try:
                autoscale_y(ax)
            except Exception:
                pass

            if avg_TS_datas is not None:
                ax.axvline(avg_TS_datas[0]['minor radius'], c='grey',
                           ls='-.', alpha=0.5)
            elif TS_data is not None:
                ax.set_xlim(0.6 * TS_data['minor radius'][k],
                            TS_data['minor radius'][k] + 0.05)
                ax.axvline(TS_data['minor radius'][k], c='grey',
                           ls='-.', alpha=0.5)

        if TS_data is not None:
            xL, yL = \
                TS_data['minor radius'][0] + .01, \
                np.max(TS_data['values'][
                    'T_e']['Te map without outliers'][-1]) * 0.2
            ax4.text(xL, yL, 'r$_{LCFS}$')
        elif avg_TS_datas is not None:
            xL, yL = \
                avg_TS_datas[0]['minor radius'] + .01, \
                np.max(avg_TS_datas[-1]['values']['T_e']['Te map']) * 0.2
            ax4.text(xL, yL, 'r$_{LCFS}$')

        ax3.legend()
        fig_current_save('radiational_fraction_edge', fig)
        fig.savefig(file + '/radiational_fraction_' +
                    program.replace('.', '_') + '_edge.png')

        for ax in [ax3, ax4]:
            if avg_TS_datas is not None:
                ax.axvline(avg_TS_datas[0]['minor radius'], c='grey',
                           ls='-.', alpha=0.5)
            elif TS_data is not None:
                ax.set_xlim(
                    -(TS_data['minor radius'][k] + 0.05),
                    TS_data['minor radius'][k] + 0.05)
                ax.axvline(TS_data['minor radius'][k], c='grey',
                           ls='-.', alpha=0.5)

        if (species is not None) and (photon_energies is not None):
            cPhoton = cycle(cm.brg(np.linspace(0, 1, 2 * len(species))))
            for i, S in enumerate(species):
                phL = ax4.axhline(photon_energies[i] / 1e3, c=next(cPhoton),
                                  ls=':', alpha=0.5)
                ax4.text(-0.5 + i * 0.2, photon_energies[i] / 1e3,
                         S.replace('_', ' ') + ' E$_{P}$~' +
                         str(round(photon_energies[i], 3)) + 'eV',
                         color=phL.get_color())

        for ax in [ax3, ax4]:
            autoscale_y(ax)
            if avg_TS_datas is not None:
                ax.axvline(-avg_TS_datas[0]['minor radius'], c='grey',
                           ls='-.', alpha=0.5)
            elif TS_data is not None:
                ax.axvline(-TS_data['minor radius'][k], c='grey',
                           ls='-.', alpha=0.5)

    p.tight_layout()
    if (TS_data is None) and (avg_TS_datas is None):
        fig.set_size_inches(5., 2.5)
    else:
        fig.set_size_inches(5., 2.5)

    fig_current_save('radiational_fraction', fig)
    fig.savefig(file + '/radiational_fraction_' +
                program.replace('.', '_') + '.png')
    p.close('all')
    return


def chordal_profile(
        program='20181010.032',
        chordal_data={'none': None},
        times=None,
        fracs=[0.33, 0.66, 0.9, 1.0],
        camera_info={'none': None},
        reff=np.array(128),
        cam='HBCm',
        best_chans=[1, 5, 15],
        vmecID='1000_1000_1000_1000_+0000_+0000/01/00jh_l/'):
    """ plotting chordial brightness profile
    Args:
        program (str, optional):
        chordal_data (dict, optional):
        reff (dict, optional):
        times (list, optional):
        cam (str, optional):
        best_chans (list, optional):
    Returns:
        None
    """
    if times is None:
        times = [1.0]
    labels = ['f$_{rad}\\sim$' + format(f, '.2f') for f in fracs]

    m_R = requests.get(
        'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/' +
        vmecID + 'minorradius.json').json()['minorRadius']

    fig = p.figure()
    ax = fig.add_subplot(111)
    ar = ax.twiny()
    colors = cm.brg(np.linspace(0, 1., len(fracs)))

    R, CH = [], []
    broken = camera_info['channels']['droplist']
    for ch in camera_info['channels']['eChannels'][cam]:
        if ch not in broken:
            CH.append(ch)
            R.append(reff[ch] / m_R)

    # split middle and flip with low opacity to check symmetry
    j = mClass.find_nearest(np.array(R), 0.0)[0] + 1
    R = np.array(R)

    index_list = np.argsort(np.array(R))
    sort_channels = np.array(CH)[index_list]
    sort_R = np.array(R)[index_list]

    t1 = archivedb.get_program_t1(program)
    for k, t in enumerate(times):
        idx = mClass.find_nearest(
            np.array(chordal_data['dimensions']), t * 1e9 + t1)[0]
        cpp, = ax.plot(
            sort_R,
            # sort_channels,
            [v * 1e-3 for i, v in enumerate(chordal_data['values'][idx])
             if camera_info['channels']['eChannels'][cam][i] not in broken],
            label=labels[k], alpha=1., ls='-.', c=colors[k])

        # left
        ax.plot(
            sort_R[:j],
            # sort_channels[:j],
            1.e-3 * np.flipud([
                v for i, v in enumerate(chordal_data['values'][idx])
                if camera_info['channels']['eChannels'][cam][i]
                not in broken][-j:]),
            alpha=0.25, ls='-.', c=cpp.get_color())
        # right
        ax.plot(
            sort_R[-j:],
            # sort_channels[-j:],
            1.e-3 * np.flipud([
                v for i, v in enumerate(chordal_data['values'][idx])
                if camera_info['channels']['eChannels'][cam][i]
                not in broken][:j]),
            alpha=0.25, ls='-.', c=cpp.get_color())

    # ax.legend()
    ax.legend(
        loc='upper center', bbox_to_anchor=(.475, 1.44),
        ncol=4, fancybox=True, shadow=False,
        handletextpad=0.3, labelspacing=0.5, handlelength=1.)

    textstring = 'mirrored: $-\\cdot-\\cdot-\\cdot-$'
    # build box
    prop = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(
        .3, .8, textstring, transform=ax.transAxes,
        verticalalignment='top', bbox=prop)

    # ax.axvline(-1. * m_R, lw=1.0, c='k', alpha=0.5, ls='-.')
    # ax.text(-1. * m_R, np.max(chordal_data['values'][idx]) * 0.75 * 1.e-6,
    #         'r$_{LCFS}$')

    for axis in [ax, ar]:
        # axis.set_xlim(np.min(sort_channels), np.max(sort_channels))
        axis.set_xlim(np.min(sort_R), np.max(sort_R))

    tick_locs = ax.get_xticks()
    N = int(np.ceil(
        # np.shape(sort_channels)[0] / np.shape(tick_locs)[0])) + 1
        np.shape(sort_R)[0] / np.shape(tick_locs)[0]))

    # ar.set_xticks(sort_channels[::N])
    # ar.set_xticklabels([str(round(r, 2)) for r in sort_R[::N]])
    ar.set_xticklabels([str(n) for n in sort_channels[::N]])

    # chrodal profile edge
    ax.set_xlabel('radius [r$_{a}$]')
    ar.set_xlabel('channel #')
    ax.set_ylabel('brightness [kWm$^{-3}$]')

    fig.set_size_inches(5., 3.5)

    if best_chans is not None:
        for k, ch in enumerate(best_chans):
            if (reff['radius']['reff'][ch] < -.4):
                ax.axvline(reff['radius']['reff'][ch], lw=1.0, c='k', ls=':')
                ax.text(reff['radius']['reff'][ch] + 0.01,
                        (k / len(best_chans) + 0.1) * np.max(
                            chordal_data['values'][idx]),
                        'eCh#' + str(ch), color='k')

    if False:  # cam == 'HBCm':
        # save
        ax.set_xlim(np.min(R), -0.4)
        fig_current_save('chordal_profile_edge', fig)
        fig.savefig(
            '../results/RAD_FRAC/' + program[0:8] + '/chordal_profile_' +
            cam + '_' + program.replace('.', '_') + '_edge.png')

    # ax.set_xlim(np.min(R), np.max(R))
    # ax.axvline(m_R, lw=1.0, c='k', alpha=0.5, ls='-.')
    if best_chans is not None:
        for k, ch in enumerate(best_chans):
            ax.axvline(reff['radius']['reff'][ch], lw=1.0, c='k', ls=':')
            ax.text(reff['radius']['reff'][ch] + 1.0,
                    (k / len(best_chans) + 0.1) * np.max(
                        chordal_data['values'][idx]),
                    'eCh#' + str(ch), color='k')

    if cam == 'HBCm':
        # chrodal profile full normal
        fig_current_save('chordal_profile', fig)
    fig.savefig('../results/RAD_FRAC/' + program[0:8] + '/chordal_profile_' +
                cam + '_' + program.replace('.', '_') + '.png')
    p.close('all')
    return


def thomson_plot(
        shotno='20181018.041',
        TS_data={'none': None},
        evalStart=None,
        evalEnd=None,
        neplot=True,
        teplot=True,
        rewrite=True,
        debug=False,
        indent_level='\t'):
    """ plotting TS profile
    Args:
        shotno (str, optional):
        TS_data (dict, optional):
        evalStart (float, optional):
        evalEnd (float, optional):
        neplot (bool, optional):
        teplot (bool, optional):
        debug (bool, optional):
        indent_level (str, optional):
    Returns:
        None
    """
    t1 = int(archivedb.get_program_t1(shotno))
    t5 = int(archivedb.get_program_from_to(shotno)[1])

    if evalStart is not None:
        evalStart = int(t1 + evalStart * 1e9) \
            if evalStart.__class__ == float else int(evalStart)
    else:
        evalStart = int(t1 - 0.1 * 1e9)

    if evalEnd is not None:
        evalEnd = int(t1 + evalEnd * 1e9) \
            if evalEnd.__class__ == float else int(evalEnd)
    else:
        evalEnd = int(t5 + 0.1 * 1e9)

    # re-assort data to more convenient names
    V_Te = TS_data['values']['T_e']
    V_ne = TS_data['values']['n_e']

    IND = np.where((TS_data['time'] <= evalEnd) &
                   (TS_data['time'] >= evalStart))[0]
    M = len(IND)

    print(indent_level + '>> plotting profiles:')
    for k in tqdm(range(M), desc='time'):  # in region of interest
        j = IND[k]
        i = TS_data['time'][j]

        axes = [] if (neplot or teplot) else None
        if (neplot):
            suffix = '_combined' if (teplot) else '_ne'
        else:
            suffix = '_Te'

        filename = '../results/THOMSON/' + shotno[0:8] + \
            '/' + shotno[-3:] + '/TS_profile_at_' + \
            str((i - t1) / 1e9) + 's' + suffix + '.png'
        if not rewrite:
            if not glob.glob(filename) == []:
                continue  # skip, exists already, do next one

        # plot Te
        fig = p.figure()
        if teplot:
            if neplot:
                te_ax = fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[])
            else:
                te_ax = fig.add_subplot(111)
            axes.append(te_ax)
            te_ax.grid(b=True, which='major', linestyle='-.')
            linesA = []

        try:
            if teplot:
                tegaussA = te_ax.fill_between(  # conf. interval Te fit gauss
                    TS_data['r_fit'][j],
                    V_Te['Te fit gauss'][j] - np.sqrt(
                        V_Te['Te fit gauss conv'][j]),
                    V_Te['Te fit gauss'][j] + np.sqrt(
                        V_Te['Te fit gauss conv'][j]),
                    alpha=0.25, color='lightcoral',
                    label='gauss $2\\sigma$')  # 'T$_{e, gauss 2{\\sigma}}$')
                linesA.append(tegaussA)

        except Exception:
            if debug:
                print('failed T_e gauss 2 x sigma', end=', ')

        try:  # interpolate mirrored TS data in vmec, Guassian process fit
            try:
                if teplot:
                    tegaussP, = te_ax.plot(  # gauss fit plot
                        TS_data['r_fit'][j], V_Te['Te gauss'][j],
                        '-.', color='red', label='gauss fit')
                    # 'T$_{e, gauss}$')
                    linesA.append(tegaussP)

                    temapP, = te_ax.plot(  # Te map without outliers
                        V_Te['r without outliers (Te)'][j],
                        V_Te['Te map without outliers'][j],
                        marker='x', ls='--', lw=0.3, c='b',
                        label='map')  # 'T$_{e, map}$')
                    linesA.append(temapP)

            except Exception:
                if debug:
                    print('failed Te map', end=', ')

            try:  # for r without outliers: Te 97.5
                if teplot:
                    temapA = te_ax.fill_between(
                        V_Te['r without outliers (Te)'][j],
                        V_Te['Te low97 without outliers'][j],
                        V_Te['Te high97 without outliers'][j],
                        facecolor='cornflowerblue', alpha=0.25,
                        label='map $2\\sigma$')  # 'T$_{e, map 2{\\sigma}}$')
                    linesA.append(temapA)

            except Exception:
                if debug:
                    print('failed Te 97.5%', end=', ')

        except Exception:
            if debug:
                print('failed r without outliers', end=', ')

        try:
            if teplot:
                tegauss97 = te_ax.fill_between(
                    TS_data['r_fit'][j],
                    V_Te['Te low97 gauss fit'][j],
                    V_Te['Te high97 gauss fit'][j],
                    facecolor='limegreen', alpha=.25,
                    label='$97.5\%$')  # 'T$_{e, 97.5\%}$')
                linesA.append(tegauss97)

        except Exception:
            if debug:
                print('failed T_e 97.5% plot', end=', ')

        try:
            if teplot:
                # te_ax.legend(linesA, [l.get_label() for l in linesA])
                te_ax.set_xlabel('r$_{eff}$ [m]')
                te_ax.set_ylabel('T$_{e}$ [keV]')
                te_ax.legend(
                    linesA, [l.get_label() for l in linesA],
                    loc='upper center', bbox_to_anchor=(.5, 1.3),
                    ncol=5, fancybox=True, shadow=False,
                    handletextpad=0.3, labelspacing=0.5, handlelength=1.)
                te_ax.set_ylim(-0.5, 7.)
                te_ax.set_xlim(
                    -TS_data['minor radius'][j] - 0.05,
                    TS_data['minor radius'][j] + 0.05)
        except Exception:
            if debug:
                print('failed T_e map plot', end=', ')

        if neplot:
            if teplot:
                ne_ax = fig.add_axes([0.1, 0.1, 0.8, 0.4])
            else:
                fig = p.figure()
                ne_ax = fig.add_subplot(111)
                ne_ax.grid(b=True, which='major', linestyle='-.')
                # ne_ax.set_title(
                #     'scaled W7X %s @ %s s' % (shotno, (i - t1) / 1e9))

            axes.append(ne_ax)
            ne_ax.grid(b=True, which='major', linestyle='-.')
            linesB = []

        try:  # interpolate mirrored TS data in vmec coordinates
            try:
                if neplot:
                    nemapA = ne_ax.fill_between(  # error +/-2 sigma
                        V_ne['r without outliers (ne)'][j],
                        V_ne['ne low97 without outliers'][j],
                        V_ne['ne high97 without outliers'][j],
                        label='n$_{e, map 2 {\\sigma}}$',
                        facecolor='cornflowerblue', alpha=.25)
                    linesB.append(nemapA)

            except Exception:
                if debug:
                    print('failed ne map 2xsigma', end=', ')

            try:  # fit the remaining data
                if neplot:  # gauss plot ne
                    negaussP, = ne_ax.plot(
                        TS_data['r_fit'][j],
                        V_ne['ne fit gauss'][j],
                        ls='-.', c='r', label='n$_{e, gauss}$')
                    linesB.append(negaussP)

                    negaussA = ne_ax.fill_between(
                        TS_data['r_fit'][j],
                        V_ne['ne fit gauss'][j] - np.sqrt(
                            V_ne['ne fit gauss conv'][j]),
                        V_ne['ne fit gauss'][j] + np.sqrt(
                            V_ne['ne fit gauss conv'][j]),
                        alpha=0.25, color='lightcoral',
                        label='n$_{e, gauss2{\\sigma}}$')
                    linesB.append(negaussA)

                    nemapP, = ne_ax.plot(  # mapped scaling
                        V_ne['r without outliers (ne)'][j],
                        V_ne['ne map without outliers'][j],
                        marker='x', markersize=4., c='b', ls=':', lw=0.3,
                        label='n$_{e, map}$')
                    linesB.append(nemapP)

            except Exception:
                if debug:
                    print('failed ne gauss 2xsigma', end=', ')

            if int(shotno[0:4]) == 2018:  # real space scaling
                try:  # factor mapped real space
                    if neplot:
                        nefactorP, = ne_ax.plot(
                            V_ne['r without outliers'][j],
                            V_ne['ne map factor real space'][j],
                            marker='h', markersize=4., ls='', c='m',
                            label='n$_{e}$ real space')
                        linesB.append(nefactorP)

                        # error +/-2 sigma
                        nefactorA = ne_ax.fill_between(
                            V_ne['r without outliers'][j],
                            V_ne['ne low97 factor real space'][j],
                            V_ne['ne high97 factor real space'][j],
                            facecolor='tab:pink', alpha=.25,
                            label='n$_{e, 2{\\sigma}} real space$')
                        linesB.append(nefactorA)

                except Exception:
                    if debug:
                        print('failed factor real space', end=', ')

            try:
                if neplot:
                    negauss97A = ne_ax.fill_between(
                        TS_data['r_fit'][j],
                        V_ne['ne low97 gauss fit'][j],
                        V_ne['ne high97 gauss fit'][j],
                        label='n$_{e, 97.5\%}$',
                        facecolor='limegreen', alpha=.25)
                    linesB.append(negauss97A)

            except Exception:
                if debug:
                    print('failed ne 97.5', end=', ')

        except Exception:
            if debug:
                print('failed ne data assertion', end=' ')

        # limit volumes inside LCFS max ne_map pellet phase
        if neplot:
            # ne_ax.legend(linesB, [l.get_label() for l in linesB])
            ne_ax.set_xlabel('radius [r$_{a}$]')
            ne_ax.set_ylabel('n$_{e}$ [10$^{19}$/m$^{3}$]')
            ne_ax.set_ylim(-0.2, 15.)
            ne_ax.set_xlim(
                -TS_data['minor radius'][j] - 0.05,
                TS_data['minor radius'][j] + 0.05)

        for ax in axes:
            ax.axvline(
                -TS_data['minor radius'][j], linestyle='--',
                color='grey', alpha=0.5, label='LCFS')
            ax.axvline(
                TS_data['minor radius'][j],
                linestyle='--', color='grey', alpha=0.5)

        fig.set_size_inches(5., 4.)
        fig.savefig(filename, box_inches='tight', dpi=169.0)
        fig_current_save('Profile_thomson', fig)
        p.close('all')

    print('... done!', end='\n')
    return


def plot_avg_TS_profiles(
        shotno='20181018.041',
        avg_TS_data={'none': None},
        debug=False,
        indent_level='\t'):
    """ plot average thomson profiles
    Args:
        shotno (str, optional): XP ID
        avg_TS_data (dict, optional): average thomson data from json
        evalStart (float, optional): time point to start evaluation (s f. T1)
        evalEnd (float, optional): time point to stop evaluation
        debug (bool, optional): Debug printing bool
        indent_level (str, optional): Printing indentation
    Returns:
        None.
    Notes:
        None.
    """
    M = len(avg_TS_data['time'])
    fig = p.figure()
    te = fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[])
    # te.set_title('avg. TS prof.,' + shotno + ', T=[' +
    #              str(avg_TS_data['evalStart']) + 's, ' +
    #              str(avg_TS_data['evalEnd']) + 's]')

    te.set_ylabel('T$_{e}$ [keV]')
    te.set_xlabel('r$_{eff}$')

    ne = fig.add_axes([0.1, 0.1, 0.8, 0.4])
    ne.set_ylabel('n$_{e}$ [10$^{19}$ m$^{-3}$]')

    ne.plot(
        avg_TS_data['values']['n_e']['bin r ne'],
        avg_TS_data['values']['n_e']['ne map'] / M,
        marker='x', markersize=4.0, ls='--', lw=0.5,
        c='b', label='n$_{e, map}$')
    ne.plot(
        avg_TS_data['r_fit'],
        avg_TS_data['values']['n_e']['ne gauss'] / M,
        ls='-.', lw=0.5, c='r', label='n$_{e, gauss}$')
    # autoscale_y(ne)

    ne.fill_between(
        avg_TS_data['values']['n_e']['bin r ne'],
        avg_TS_data['values']['n_e']['ne map low97'] / M,
        avg_TS_data['values']['n_e']['ne map high97'] / M,
        facecolor='cornflowerblue', alpha=0.25,
        label='n$_{e, map 2{\\sigma}}$')
    ne.fill_between(
        avg_TS_data['r_fit'],
        avg_TS_data['values']['n_e']['ne gauss'] / M -
        avg_TS_data['values']['n_e']['ne gauss conv'] / M,
        avg_TS_data['values']['n_e']['ne gauss'] / M +
        avg_TS_data['values']['n_e']['ne gauss conv'] / M,
        alpha=0.25, color='lightcoral',
        label='n$_{e, gauss 2{\\sigma}}$')

    te.plot(
        avg_TS_data['values']['T_e']['bin r te'],
        avg_TS_data['values']['T_e']['Te map'] / M,
        marker='x', markersize=4.0, ls='--', lw=0.5,
        c='b', label='map')  # 'T$_{e, map}$')
    te.plot(
        avg_TS_data['r_fit'],
        avg_TS_data['values']['T_e']['Te gauss'] / M,
        ls='-.', lw=0.5, c='r', label='gauss fit')  # 'T$_{e, gauss}$')
    # autoscale_y(te)

    te.fill_between(
        avg_TS_data['values']['T_e']['bin r te'],
        avg_TS_data['values']['T_e']['Te map low97'] / M,
        avg_TS_data['values']['T_e']['Te map high97'] / M,
        facecolor='cornflowerblue', alpha=0.25,
        label='map $2\\sigma$')   # 'T$_{e, map 2{\\sigma}}$')
    te.fill_between(
        avg_TS_data['r_fit'],
        avg_TS_data['values']['T_e']['Te gauss'] / M -
        avg_TS_data['values']['T_e']['Te gauss conv'] / M,
        avg_TS_data['values']['T_e']['Te gauss'] / M +
        avg_TS_data['values']['T_e']['Te gauss conv'] / M,
        alpha=0.25, color='lightcoral',
        label='gauss $2\\sigma$')  # 'T$_{e, gauss 2{\\sigma}}$')

    for ax in [ne, te]:
        ax.axvline(-avg_TS_data['minor radius'], ls='--', c='grey')
        ax.axvline(avg_TS_data['minor radius'], ls='--', c='grey')

    ne.set_ylim(-0.2, 15.)
    te.set_ylim(-0.5, 7.)

    ne.set_xlabel('radius [r$_{a}$]')
    # te.legend()
    # ne.legend()
    te.legend(
        loc='upper center', bbox_to_anchor=(.5, 1.3),
        ncol=4, fancybox=True, shadow=False,
        handletextpad=0.3, labelspacing=0.5, handlelength=1.)

    fig.set_size_inches(5., 4.)
    fig_current_save('avg_Profile_thomson', fig)
    fig.savefig(
        '../results/THOMSON/' + shotno[0:8] + '/' + shotno[-3:] +
        '/avg_Profile_thomson_' + str(avg_TS_data['evalStart']) + 's_' +
        str(avg_TS_data['evalEnd']) + 's.png', bbox_inches='tight', dpi=169.0)
    p.close('all')
    return


def xics_ti_plot(
        date='20181010',
        shotno='032',
        indent_level='\t'):
    """ XICS measured ion temperature plot
    Keyword Arguments:
        date {str} -- XP date (default: {'20181010'})
        shotno {str} -- XP ID (default: {'032'})
        indent_level {str} -- printing indent (default: {'\t'})
    Returns:
    """
    xics_dat = hexos_get.get_xics(
        date=int(date),
        shot=int(shotno),
        path='../results/XICS/' + date + '/',
        saving=False,
        extended=False,
        indent_level='\t')
    tmp = xics_dat['values']['ti']['values']

    if xics_dat is not None:
        stdwrite('\n' + indent_level + '>> XICS Ti plot ...')
        stdflush()
        location = r"../results/XICS/" + date

        if not os.path.exists(location):
            os.makedirs(location)

        M = 20
        colorsc = cycle(cm.brg(np.linspace(
            0, 1, int(np.shape(tmp[0]['ti'])[1] / M))))

        fig = p.figure()
        pXICS_ti = fig.add_subplot(111)
        fig.set_size_inches(7., 5.)

        ti_mean = hexos_get.return_mean_profile(
            input_data=tmp[0]['ti'])
        pXICS_ti.plot(
            tmp[2]['ti_rho'], ti_mean,
            c='k', ls='-.', label='mean T$_{i, reff}$', lw=2.0)

        for i, vec in enumerate(np.transpose(tmp[0]['ti'])):
            if i % M == 0:
                pXICS_ti.plot(
                    tmp[2]['ti_rho'], vec, c=next(colorsc), ls='--',
                    alpha=1., label='T$_{i}$ @ ' + str(round(
                        xics_dat['values']['ti']['dimensions'][i], 2)) + 's')

        pXICS_ti.set_xlim(.0, 1.)
        autoscale_y(pXICS_ti)

        foo = Z((2, np.shape(tmp[0]['ti'])[0]))
        for i, v in enumerate(tmp[0]['ti']):
            vec = np.array(v)
            foo[0, i] = np.max(vec[(vec < 5.) & (vec > 0.01)])
            foo[1, i] = np.min(vec[(vec < 5.) & (vec > 0.01)])

        pXICS_ti.fill_between(
            tmp[2]['ti_rho'], foo[1], foo[0],
            color='gray', alpha=0.25)

        pXICS_ti.legend()
        pXICS_ti.set_xlabel('r$_{eff}$ $\\rho$')
        pXICS_ti.set_ylabel('T$_{i}$ [keV]')

        p.tight_layout()
        fig_current_save('xics_ti', fig)
        fig.savefig('../results/XICS/' + date + '/xics_ti_' + date + '.' +
                    shotno + '.png', bbox_inches='tight', dpi=169.0)
        p.close('all')

    return


def xics_te_plot(
        date='20181010',
        shotno='032',
        threshUp=5.,
        threshLow=.1,
        indent_level='\t'):
    """ XICS electron temperature plot
    Keyword Arguments:
        date {str} -- XP data (default: {'20181010'})
        shotno {str} -- XP ID (default: {'032'})
        threshUp {[type]} -- threshold for mean profile (default: {5.})
        threshLow {float} -- (default: {.1})
        indent_level {str} -- printing indentation (default: {'\t'})
    Returns:
    """
    xics_dat = hexos_get.get_xics(
        date=int(date),
        shot=int(shotno),
        path='../results/XICS/' + date + '/',
        saving=True,
        extended=True,
        indent_level='\t')
    tmp = xics_dat['values']['te']['values']

    if xics_dat is not None:
        stdwrite('\n' + indent_level + '>> XICS Te plot ...')
        stdflush()
        location = r"../results/XICS/" + date

        if not os.path.exists(location):
            os.makedirs(location)

        M = 10
        colorsc = cycle(cm.brg(np.linspace(
            0, 1, int(np.shape(tmp[0]['te'])[1] / M))))

        fig = p.figure()
        pXICS_te = fig.add_subplot(111)
        fig.set_size_inches(7., 5.)

        te_mean = hexos_get.return_mean_profile(
            input_data=tmp[0]['te'], threshUp=threshUp, threshLow=threshLow,
            indent_level=indent_level + '\t')
        pXICS_te.plot(
            tmp[3]['te_rho'], te_mean,
            c='k', ls='-.', label='mean T$_{e, reff}$', lw=2.0)

        for i, vec in enumerate(np.transpose(tmp[0]['te'])):
            if ((np.max(vec) <= threshUp) and (np.min(vec) >= threshLow)):
                pXICS_te.plot(
                    tmp[3]['te_rho'], vec, c=next(colorsc), ls='--',
                    alpha=1., label='T$_{e}$ @ ' + str(round(
                        xics_dat['values']['te']['dimensions'][i], 2)) + 's')

        pXICS_te.set_xlim(.0, 1.)
        autoscale_y(pXICS_te)

        foo = Z((2, np.shape(tmp[0]['te'])[0]))
        for i, v in enumerate(tmp[0]['te']):
            vec = np.array(v)
            foo[0, i] = np.max(vec[(vec < threshUp) & (vec > threshLow)])
            foo[1, i] = np.min(vec[(vec < threshUp) & (vec > threshLow)])

        pXICS_te.fill_between(
            tmp[3]['te_rho'], foo[1], foo[0],
            color='gray', alpha=0.25)

        pXICS_te.legend()
        pXICS_te.set_xlabel('r$_{eff}$ $\\rho$')
        pXICS_te.set_ylabel('T$_{e}$ [keV]')

        p.tight_layout()
        fig.savefig('../results/XICS/' + date + '/xics_te_' + date + '.' +
                    shotno + '.png', bbox_inches='tight', dpi=169.0)
        fig_current_save('xics_te', fig)
        p.close('all')

    return


def hexos_ratio_plot(
        material='C',
        program='20181010.032',
        debug=False,
        keys=['C_II', 'C_III', 'C_IV', 'C_V'],
        indent_level='\t'):
    """ HEXOS ratios plot
    Keyword Arguments:
        material {str} -- species (default: {'C'})
        program {str} -- XP  ID (default: {'20181010.032'})
        debug {bool} -- debugging (default: {False})
        indent_level {str} -- printing indentation (default: {'\t'})
    Returns:
    """
    ratios, labels, hexos_dat = hexos_get.hexos_ratios(
        mat=material, program=program, debug=False, keylist=keys, saving=True)

    if hexos_dat is not None:
        print(indent_level + '>> HEXOS ratios plot ...')

        location = r"../results/HEXOS/" + program[0:8]
        if not os.path.exists(location):
            os.makedirs(location)

        fig = p.figure()
        pHEXOS = fig.add_subplot(111)
        pHEXOS.grid(b=True, which='major', linestyle='-.')

        N = np.shape(ratios)[1]
        colorsc = cycle(cm.brg(np.linspace(0, 1, N)))

        for i, key in enumerate(labels[:]):
            plot, = pHEXOS.plot(
                ratios[0, i, :],
                smoothing(ratios[2, i, :], 10) / np.max(ratios[2, i, :]),
                alpha=0.25, c=next(colorsc))
            pHEXOS.plot(
                ratios[0, i, :],
                smoothing(ratios[2, i, :]) / np.max(ratios[2, i, :]),
                label=labels[i], c=plot.get_color(), alpha=1.)

        pHEXOS.legend()
        pHEXOS.set_xlabel('time [s]')
        pHEXOS.set_ylabel('line ratios norm. [a.u.]')
        pHEXOS.set_title('#' + program)

        fig.savefig(
            '../results/HEXOS/' + program[0:8] + '/' + 'hexos_ratios_' +
            program.replace('.', '_') + '_' + material + '.png',
            bbox_inches='tight', dpi=169.0)
        fig_current_save('hexos_ratios_' + material, fig)
        p.close('all')

    return


def hexos_plot(
        program='20181010.032',
        indent_level='\t',
        material='C'):
    """ plotting HEXOS and XICS lines for given material
    Args:
        date (str, optional): XP date
        shotno (str, optional): XP ID
        indent_level (str, optional): Printing indentation
    Returns:
        None
    Notes:
        None
    """
    # material = 'C'
    hexos_dat, _ = hexos_get.get_hexos_xics(
        date=int(program[0:8]),
        shot=int(program[9:]),
        mat=material, saving=True,
        hexos=True, xics=False,
        indent_level=indent_level + '\t')

    if hexos_dat is not None:
        stdwrite('\n' + indent_level + '>> HEXOS plot ...')
        stdflush()
        location = r"../results/HEXOS/" + program[0:8]

        if not os.path.exists(location):
            os.makedirs(location)

        fig = p.figure()
        pHEXOS = fig.add_subplot(111)
        pHEXOS.grid(b=True, which='major', linestyle='-.')

        N = len(hexos_dat['values'])
        colorsc = cycle(cm.brg(np.linspace(0, 1, N)))

        for i, key in enumerate(hexos_dat['values'].keys()):
            x, y = \
                [x * 1e-3 for x in hexos_dat['values'][key][0]], \
                [y * 1e-4 for y in hexos_dat['values'][key][1]]
            plot, = pHEXOS.plot(
                x, y, c=next(colorsc), alpha=0.25, linewidth=0.6)
            pHEXOS.plot(
                x, smoothing(y, 100), label=key.replace('_', ' '),
                c=plot.get_color())

        pHEXOS.legend()
        pHEXOS.set_xlabel('time [s]')
        pHEXOS.set_ylabel('non rel. intensity [a.u.]')
        pHEXOS.set_title('#' + program)

        fig.savefig(
            '../results/HEXOS/' + program[0:8] + '/' + 'hexos_' +
            program.replace('.', '_') + '_' + material + '.png',
            bbox_inches='tight', dpi=169.0)
        fig_current_save('hexos_' + material, fig)
        p.close('all')

    return


def overview_plot(
        last_panel='divertor load',
        valve_type='DCH',
        prio={'none': {'none': None}},
        dat={'none': {'none': None}},
        power={'none': {'none': None}},
        program_info={'none': None},
        program='20181010.32',
        solo=True,
        indent_level='\t'):
    """ Overviewplot with most important infos all at hand (hopefully)
    Args:
        last_panel (0, str, optional): last panel (divertor_load, channels)
        valve_type (1, str, optional): which valves to look at (QSQ, DCH)
        prio (2, list, optional): List of preloaded stuff
        dat (3, list, optional): List of downloads from archive
        power (4, TYPE, optional): Calculated stuff around Prad
        program_info (7, dict, optional): Dictionary of program info
        program (9, str, optional): Numb and date together
        indent_level (10, str, optional): Printing indentation
    Returns:
        None.
    Notes:
        None.
    """
    print(indent_level + '>> Overview plot ...')
    location = '../results/OVERVIEW/' + program[0:8]
    if not os.path.exists(location):
        os.makedirs(location)

    start = str(program_info['programs'][0]['from'])
    stop = str(program_info['programs'][0]['upto'])

    try:  # FIGURE SETUP
        # Make a square figure and axes
        fig = p.figure()

        prad = fig.add_subplot(511)
        eldens = fig.add_subplot(512)
        eltemp = fig.add_subplot(513)
        gasvalve = fig.add_subplot(514)
        last = fig.add_subplot(515)

        wdia = eldens.twinx()

        for ax in [prad, eltemp, eldens, gasvalve, last]:
            ax.grid(b=True, which='major', linestyle='-.')

        fig.set_size_inches(7., 16.)
        title_str = "Overview of #" + program
        linesdia = []

    except (Exception):
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  overview plot setup failed')

    try:  # TIMES
        time_limits = dat_lists.overview_time_limits(program, prio=prio)

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  overview time limits failed')

    try:  # ECRH
        ECHR = api.download_single(
            api.download_link(name='ECRH'),
            program_info=program_info,
            start_POSIX=start, stop_POSIX=stop)

        prad.plot([(t - prio['t0']) / 1e9 for t in ECHR['dimensions']],
                  [v / 1e3 for v in ECHR['values']], "k", label="ECRH")
        prad.set_xlim(time_limits)
        prad.set_title(title_str)
        prad.set_ylabel("power [MW]")
        prad.legend()
        prad.set_ylim(-0.05, 1.1 * np.max([v / 1e3 for v in ECHR['values']]))

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  overview ecrh plot failed')

    try:  # my calc
        print(indent_level + '\t>> plotting self prad')
        time = power['time']
        prad_val1 = [y / 1e6 for y in power['P_rad_hbc']]  # in MW
        prad_val2 = [y / 1e6 for y in power['P_rad_vbc']]  # in MW

    except Exception:  # archive
        print(indent_level + '\t\\\ failed\n' +
              indent_level + '\t>> plotting archive prad')
        archiveHBC = api.download_single(
            api.download_link(name='Bolo_HBCmPrad'),
            program_info=program_info,
            start_POSIX=start, stop_POSIX=stop)
        archiveVBC = api.download_single(
            api.download_link(name='Bolo_VBCPrad'),
            program_info=program_info,
            start_POSIX=start, stop_POSIX=stop)

        time = [(x - prio['t0']) / 1e9 for x in archiveHBC["dimensions"]]
        prad_val1 = [x for x in archiveHBC['values'][0]]
        prad_val2 = [x for x in archiveVBC['values'][0]]

    try:
        prad.plot(time, prad_val1, c="r", label="P$_{rad}$ (HBC)", lw=2.0)
        prad.plot(time, prad_val2, c="b", label="P$_{rad}$ (VBC)", lw=2.0)

        prad.set_xlim(time_limits)
        prad.set_title(title_str)
        prad.set_ylabel("power [MW]")
        prad.legend()

    except Exception:
        print(indent_level + '\t\\\ failed plotting prad')

    if not solo:
        try:  # plot Vi feedback of real time single channel and P_rad
            fb2 = api.download_single(
                api.download_link(name='BoloSingleChannelFeedback'),
                program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)
            fb1 = api.download_single(
                api.download_link(name='BoloRealTime_P_rad'),
                program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)

            E = rand.randrange(1000, 3500) / 10000
            f1 = np.max(fb1['values'][0]) / (np.max(power['P_rad_hbc']) / 1.e6)
            f2 = np.max(fb2['values'][0]) / (np.max(power['P_rad_hbc']) / 1.e6)

            f1 += E * f1
            f2 += E * f2

            prad.plot(
                [(x - prio['t0']) / 1e9 for x in fb1["dimensions"]],
                [x * f1 for x in fb1["values"][0]], color="orange",
                label="P$_{rad, pred}$ multi ch. $\\times$ f")
            prad.plot(
                [(x - prio['t0']) / 1e9 for x in fb2["dimensions"]],
                [x * f2 for x in fb2["values"][0]], color='cyan',
                label="single ch. $\\times$ f")
            prad.legend()

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\ ', exc_type, fname,
                  exc_tb.tb_lineno, '\n' + indent_level +
                  '\t\\\  Failed at feedback plot')

    try:  # ELECTRON TEMP
        TeCore = api.download_single(
            api.download_link(name='T_e ECE core'),
            program_info=program_info,
            start_POSIX=start, stop_POSIX=stop)
        TeOut = api.download_single(
            api.download_link(name='T_e ECE out'),
            program_info=program_info,
            start_POSIX=start, stop_POSIX=stop)

        eltco_time = [(x - prio['t0']) / (1e9) for x in TeCore["dimensions"]]
        eltout_time = [(x - prio['t0']) / (1e9) for x in TeOut["dimensions"]]
        eltco_val = [y for y in TeCore["values"]]  # in keV
        eltout_val = [y for y in TeOut["values"]]  # in keV

        eltoutp, = eltemp.plot(
            eltout_time, smoothing(eltout_val),
            "lawngreen", label="T$_{e}$ out (ECE24)", lw=2.0)
        eltcop, = eltemp.plot(
            eltco_time, smoothing(eltco_val),
            "green", label="T$_{e}$ core (ECE13)", lw=2.0)

        try:  # Te QTB
            Te_QTB = api.download_single(
                api.download_link(name='T_e QTB vol2'),
                program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)

            elt_thomsp, = eltemp.plot(
                [(x - prio['t0']) / 1e9 for x in Te_QTB['dimensions']],
                [y for y in Te_QTB['values']],  # in keV
                "cyan", marker='^', lw=1.0, markersize=3.0,
                label='T$_{e}$ (QTB vol2V6)')

        except Exception:
            print(indent_level + '\t\\\ failed QTB T_e')

        eltemp.set_xlim(time_limits)
        eltemp.set_ylabel("temperature [keV]")
        eltemp.yaxis.label.set_color(eltcop.get_color())
        eltemp.tick_params(axis='y', colors=eltcop.get_color())
        eltemp.spines["right"].set_edgecolor(eltcop.get_color())
        eltemp.legend()

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level +
              '\t\\\  overview electron temp plot failed')

    try:
        if (valve_type == 'DCH'):
            L = ['H2 flow', 'He flow',
                 'flow [10$^{-3}$ bar$\\,$s$\\,$l$^{-1}$]']
            valve1 = api.download_single(
                api.download_link(name='Main valve BG011'),
                program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)
            valve2 = api.download_single(
                api.download_link(name='Main valve BG031'),
                program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)

        elif (valve_type == 'QSQ'):
            L = ['QSQ AEH51', 'QSQ AEH30',
                 'valve actuation [a.u.]']
            valve1 = api.download_single(
                api.download_link(name='QSQ Feedback AEH51'),
                program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)
            valve2 = api.download_single(
                api.download_link(name='QSQ Feedback AEH30'),
                program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)

        V1, = gasvalve.plot(
            [(t - prio['t0']) / 1e9 for t in valve1['dimensions']],
            [v + 20. for v in valve1['values']],
            'purple', label=L[0], lw=2.0)
        V2, = gasvalve.plot(
            [(t - prio['t0']) / 1e9 for t in valve2['dimensions']],
            [v + 20. for v in valve2['values']],
            'darkgrey', label=L[1], lw=2.0)

        gasvalve.set_ylabel(L[2])
        gasvalve.yaxis.label.set_color(V1.get_color())
        gasvalve.tick_params(axis='y', colors=V1.get_color())
        gasvalve.spines["right"].set_edgecolor(V1.get_color())

        gasvalve.set_xlim(time_limits)
        gasvalve.legend()

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level +
              '\t\\\  overview gasvalve plot failed')

    try:
        try:
            ne = api.download_single(
                api.download_link(name='T_e QTB vol2'),
                program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)
            label, f = "n$_{e}$ (QTB vol2V6)", 1
        except Exception:
            print(indent_level + '\t\\\ failed QTB n_e')
            ne = api.download_single(
                api.download_link(name='n_e lint'),
                program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)
            label, f = "n$_{e}$ (line int.)", 1e19

        eldensp, = eldens.plot(
            [(t - prio['t0']) / 1e9 for t in ne['dimensions']],
            [v / f for v in ne['values']],
            "magenta", label=label, lw=2.0)

        eldens.set_xlim(time_limits)
        eldens.set_ylabel("density [$10^{19}$ m$^{-3}$]")
        eldens.yaxis.label.set_color(eldensp.get_color())
        eldens.tick_params(axis='y', colors=eldensp.get_color())
        eldens.spines["right"].set_edgecolor(eldensp.get_color())

        linesdia.append(eldensp)
        eldens.legend(linesdia, [l.get_label() for l in linesdia])

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level +
              '\t\\\  overview electron density plot failed')

    try:
        dia = api.download_single(
            api.download_link(name='W_dia'),
            program_info=program_info,
            start_POSIX=start, stop_POSIX=stop)

        wdiap, = wdia.plot(
            [(t - prio['t0']) / 1e9 for t in dia['dimensions']],
            dia['values'], "orange", label="W$_{dia}$", lw=2.0)

        wdia.set_ylabel("energy [kJ]")
        wdia.yaxis.label.set_color(wdiap.get_color())
        wdia.tick_params(axis='y', colors=wdiap.get_color())
        wdia.spines["right"].set_edgecolor(wdiap.get_color())

        linesdia.append(wdiap)
        eldens.legend(linesdia, [l.get_label() for l in linesdia])

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level +
              '\t\\\  overview diamag energy plot failed')

    try:
        if (last_panel == 'channels'):
            [L, LY, cs] = [
                ['core rad.', 'outboard', 'inboard'],
                'radiation [a.u.]', ['orange', 'cyan', 'lawngreen']]

            i = 0
            for i, ch in enumerate(power['voltage adjusted'][[48, 66, 56]]):
                last.plot(power['time'], [v * 1e3 for v in ch],
                          c=cs[i], label=L[i], lw=2.0)

        elif (last_panel == 'divertor load'):
            [L0, Ls, LY] = \
                ['AEF', ['20', '21', '30', '31', '40', '41', '50'],
                 'heat load [MW]']
            for i, tag in enumerate(Ls):
                try:
                    div = api.download_single(
                        api.download_link(name='DivAEF' + tag),
                        program_info=program_info,
                        start_POSIX=start, stop_POSIX=stop)
                    if i == 0:
                        power = np.zeros((len(div['dimensions'])))

                    power = [P + (div['values'][i] * 1e-6 * 10 / 6)
                             for i, P in enumerate(power)]

                    last.plot(
                        [(t - prio['t0']) / 1e9 for t in div['dimensions']],
                        [p * 1e-6 * 10 / 6 for p in div['values']],
                        ls='-.', lw=.75, alpha=0.5)

                except Exception:
                    print(indent_level + '\t\\\ ', L0 + Ls[i] + ' not found')

            if (program == '20181010.032'):
                divload = np.loadtxt(
                    '../files/20181010.032_total_divertor_load.txt')
                power = [x * 1e-6 for x in divload[:, 1]]
                time = [(t - prio['t0']) / 1e9 for t in divload[:, 0]]

            else:
                time = [(t - prio['t0']) / 1e9 for t in div['dimensions']]

            error = np.mean(
                [x for i, x in enumerate(power) if time_limits[1] < time[i]])
            last.plot(time, power, lw=1.0, c='pink',
                      label='int. divertor load')
            last.fill_between(
                time, power - error, power + error,
                facecolor='coral', alpha=0.75)

        last.set_ylabel(LY)
        last.set_xlabel("time [s]")
        last.set_xlim(time_limits)
        last.legend()

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level +
              '\t\\\  channels overview plot failed')

    try:
        for ax in [eldens, eltemp, gasvalve, last, wdia]:
            autoscale_y(ax)

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  overview plot failed')

    p.tight_layout()
    fig.savefig(
        '../results/OVERVIEW/' + program[0:8] + '/overview' +
        program + '.png', bbox_inches='tight', dpi=169.0)
    fig_current_save('overview', fig)
    p.close('all')

    return


def trigger_check(
        prio={'none': {'none': None}},
        dat={'none': {'none': None}},
        power={'none': {'none': None}},
        params={'none': {'none': None}},
        program='20181010.032',
        program_info={'none': {'none': None}},
        mode='manual',
        sigma=0.3,
        indent_level='\t'):
    """ Shows early signal onset. Comparison of triggering.
    Args:
        prio
        dat
        power: Calculated power and quantities from signals.
        indent_level
        program
        timing (float, optional): Description
        sigma (float, optional): Description
    Returns:
        None.
    Notes:
        None.
    """
    location = '../results/COMPARISON/TRIGGER/' + program[0:8]
    if not os.path.exists(location):
        os.makedirs(location)

    triggers = dat_lists.triggerC_infos(
        prio=prio, program=program)
    start = str(program_info['programs'][0]['from'])
    stop = str(program_info['programs'][0]['upto'])

    # trigger of t=0
    arg = np.abs([T - prio['t0'] for T in dat['BoloSignal']['dimensions']])
    iD = np.where(np.min(arg) == arg)[0][0]
    color = cm.brg(np.linspace(0, 1, 6))

    # figure sigma left and right
    M = max(params['T1 difference'], params['T4 difference'])
    if M is not 0.0:
        if M < -200.0:
            sigmaL, sigmaR = M * 1.1e-3, sigma
        elif M > 200.0:
            sigmaR, sigmaL = M * 1.1e-3, sigma
        else:
            sigmaL = sigmaR = sigma
    else:
        sigmaL = sigmaR = sigma

    T = iD * (dat['BoloSignal']['dimensions'][1] -
              dat['BoloSignal']['dimensions'][0]) / 1e9
    print(indent_level + '>> trigger : iD:', iD, 'time before T0:', T, 's')

    stdwrite('\r' + indent_level + '\t\\\ triggers: ')
    for i, t in enumerate(triggers):

        try:  # HEXOS loading once
            if (i == 0) and (mode is not None):
                hexos_dat = []
                mats = ['C', 'O']
                if program == '20181011.015':
                    mats.append('Fe')
                    mats.append('Ti')

                stdwrite('HEXOS')
                for j, material in enumerate(mats):
                    stdwrite('.')
                    stdflush()
                    Hdat = hexos_get.get_hexos_xics(
                        date=int(program[0:8]),
                        shot=int(program[-3:]),
                        mat=material, saving=False,
                        hexos=True, xics=False,
                        indent_level=indent_level + '\t')[0]
                    hexos_dat.append(Hdat)
                stdwrite('done, times: ')
                stdflush()

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
                  '\n' + indent_level + '\t\\\ trigger loading HEXOS failed')

        stdwrite(str(i) + '-' + str(t) + ', ')
        stdflush()
        lines = []

        try:
            # slice at start to check triggers in comparison to ecrh,
            fig = p.figure()
            host = fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[])
            voltageax = host.twinx()
            host.grid(b=True, which='major', linestyle='-.')
            host.set_xlim(t - sigmaL, t + sigmaR)
            fig.set_size_inches(7., 10.)
            M = 2.

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
                  '\n' + indent_level + '\t\\\ trigger setup A failed')

        try:
            guest = fig.add_axes([0.1, 0.1, 0.8, 0.4])
            eldensax = guest.twinx()
            hexosax = guest.twinx() if (mode is not None) else None
            guest.grid(b=True, which='major', linestyle='-.')
            guest.set_xlim(t - sigmaL, t + sigmaR)

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
                  '\n' + indent_level + '\t\\\ trigger setup B failed')

        try:
            ECHR = api.download_single(
                api.download_link(name='ECRH'),
                program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)

            ecrh, = host.plot(
                [(x - prio['t0']) * 1e-9 for x in ECHR['dimensions']],
                [x * 1e-3 for x in ECHR['values']],
                c='k', lw=1.0, label="ECRH")
            lines.append(ecrh)

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
                  '\n' + indent_level + '\t\\\ trigger ECRH failed')

        try:
            prad_new, = host.plot(
                power['time'],
                [x * 1e-6 for x in power['P_rad_hbc']],
                c='r', lw=1.0, label="P$_{rad, new}$",
                marker='>', markersize=M)
            prad_old, = host.plot(
                [(x - prio['t0']) / 1e9 for x in params['raw time']],
                [x * 1e-6 for x in power['P_rad_hbc']],
                c='r', lw=.75, label="P$_{rad, old}$", alpha=.5)
            lines.append(prad_new)
            lines.append(prad_old)

            title_str = "#" + program + ', ST:' + \
                str(params['new sample time']) + 's'
            host.set_title(title_str)
            host.set_ylabel('P / MW')
            host.yaxis.label.set_color(prad_new.get_color())
            host.tick_params(axis='y', colors=prad_new.get_color())

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
                  '\n' + indent_level + '\t\\\ trigger Prad failed')

        try:
            # voltages, HBC
            voltage3, = voltageax.plot(
                [(x - prio['t0']) * 1e-9 for x in
                 dat['BoloSignal']['dimensions']],
                [y * 1e3 for y in power['voltage adjusted'][3]],
                c=color[0], lw=1.0, alpha=0.75, label="Volt. Ch#3")
            voltage15, = voltageax.plot(
                [(x - prio['t0']) * 1e-9 for x in
                 dat['BoloSignal']['dimensions']],
                [y * 1e3 for y in power['voltage adjusted'][15]],
                c=color[1], lw=1.0, alpha=0.75, label="Volt. Ch#15")
            voltage27, = voltageax.plot(
                [(x - prio['t0']) * 1e-9 for x in
                 dat['BoloSignal']['dimensions']],
                [y * 1e3 for y in power['voltage adjusted'][27]],
                c=color[2], lw=1.0, alpha=0.75, label="Volt. Ch#27")

            # voltages, VBC
            voltage61, = voltageax.plot(
                [(x - prio['t0']) * 1e-9 for x in
                 dat['BoloSignal']['dimensions']],
                [y * 1e3 for y in power['voltage adjusted'][61]],
                c=color[3], lw=2.0, alpha=0.75, label="Volt. Ch#61")
            voltage78, = voltageax.plot(
                [(x - prio['t0']) * 1e-9 for x in
                 dat['BoloSignal']['dimensions']],
                [y * 1e3 for y in power['voltage adjusted'][61]],
                c=color[4], lw=1.0, alpha=0.75, label="Volt. Ch#78")
            voltage65, = voltageax.plot(
                [(x - prio['t0']) * 1e-9 for x in
                 dat['BoloSignal']['dimensions']],
                [y * 1e3 for y in power['voltage adjusted'][65]],
                c=color[5], lw=1.0, alpha=0.75, label="Volt. Ch#65")

            for ax in [voltage3, voltage15, voltage27,
                       voltage61, voltage78, voltage65]:
                lines.append(ax)

            voltageax.set_ylabel('U / V')
            voltageax.yaxis.label.set_color(voltage15.get_color())
            voltageax.spines["right"].set_edgecolor(voltage15.get_color())
            voltageax.tick_params(axis='y', colors=voltage15.get_color())
            voltageax.legend(lines, [l.get_label() for l in lines])
            autoscale_y(voltageax)
        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
                  '\n' + indent_level + '\t\\\ trigger voltages failed')

        try:
            # lower plot
            lines = []

            dia = api.download_single(
                api.download_link(name='W_dia'),
                program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)
            diamag, =  guest.plot(
                [(x - prio['t0']) * 1e-9 for x in dia['dimensions']],
                [x * 1e-2 for x in dia['values']],
                c='k', lw=1.0, label="W$_{dia}$")

            ne = api.download_single(
                api.download_link(name='n_e lint'),
                program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)
            eldens, = eldensax.plot(
                [(x - prio['t0']) * 1e-9 for x in ne['dimensions']],
                [x * 1e-19 for x in ne['values']],
                c='b', lw=1.0, label="n$_{el}$ lin.")
            lines.append(diamag)
            lines.append(eldens)

            guest.set_xlabel('t / s')
            guest.set_ylabel('W$_{dia}$ / 100 kJ')
            guest.spines["right"].set_edgecolor(diamag.get_color())
            guest.yaxis.label.set_color(diamag.get_color())
            guest.tick_params(axis='y', colors=diamag.get_color())

            eldensax.set_ylabel('n$_{e}$ / 1e19 m$^{-2}$')
            eldensax.yaxis.label.set_color(eldens.get_color())
            eldensax.spines["right"].set_edgecolor(eldens.get_color())
            eldensax.tick_params(axis='y', colors=eldens.get_color())
            eldensax.legend(lines, [l.get_label() for l in lines])
            for ax in [guest, eldensax]:
                autoscale_y(ax)

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
                  '\n' + indent_level + '\t\\\ trigger diamag/eldens failed')

        try:
            if (mode is not None):
                hexosax.spines["right"].set_position(("axes", 1.13))

                for j, material in enumerate(mats):
                    if hexos_dat[j] is not None:
                        amnt = len(hexos_dat[j]['values'].keys())

                        for i, key in enumerate(
                                hexos_dat[j]['values'].keys()):
                            if i == 0:
                                lL = len(hexos_dat[j]['values'][key][1])
                                hexosS = Z((lL))
                            hexosS += hexos_dat[j]['values'][key][1]
                        hexosS = hexosS / amnt  # smoothing(hexosS, 10) / amnt

                    pHEXOS, = hexosax.plot(
                        [x * 1e-3 for x in hexos_dat[j]['values'][key][0]],
                        [y * 1e-4 for y in hexosS],
                        label='HEXOS ${\\Sigma}$ ' + material, linestyle='-.',
                        c=color[j], linewidth=1.0)
                    lines.append(pHEXOS)

                    eldensax.legend(lines, [l.get_label() for l in lines])
                    hexosax.set_ylabel('line intensity / a.u.')
                    hexosax.yaxis.label.set_color(pHEXOS.get_color())
                    hexosax.spines["right"].set_edgecolor(pHEXOS.get_color())
                    hexosax.tick_params(axis='y', colors=pHEXOS.get_color())

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
                  '\n' + indent_level + '\t\\\ trigger HEXOS failed')

        autoscale_y(host)
        p.tight_layout()
        p.ion()
        p.show()

        fig.savefig(
            "../results/COMPARISON/TRIGGER/" + program[0:8] +
            "/trigger_check_" + program.replace('.', '_') + "_t=" +
            str(t).replace('.', '_') + "__" + str(mode) + ".png",
            bbox_inches='tight', dpi=169.0)
        fig_current_save('trigger_check', fig)
        p.close('all')

    stdwrite('\n')
    stdflush()

    return


def comparison_to_shot(
        prio={'none': {'none': None}},
        dat={'none': {'none': None}},
        power={'none': {'none': None}},
        params={'none': {'none': None}},
        program='20181010.032',
        program_info={'none': {'none': None}},
        compare_archive=False,
        compare_shots=[["20171115.039"]],
        comparison=[[[True, True, True]]],
        data_names=[["20171115/vbc_powch_20171115039.dat"]],
        indent_level='\t'):
    """ Plots all the channels individually, power and voltage (for the cams).
    Args:
        prio (0, list): Constants and pre-defined quantities
        dat (1, list): Archive data downloaded prior
        power (2, list): Calculated radiation/bolometer properties
        program (5, str): Date and shotno concatenated
        program_info (6, str): HTTP resp of logbook
        compare_archive (8, bool): Bool whether to plot the archive raw data
        comparison_shots (9, list): List of XP IDs to plot with files
        comparison (10, list): Bool list what to compare at the XP ID
        data_names (11, list): File names/locations where to load
        indent_level (12, str): indentation level
    Returns:
        None
    Notes:
        None
    """
    print(indent_level + ">> Individual channels plot ...")
    loc1 = "../results/COMPARISON/CHANNELS/" + program[0:8]
    loc2 = "../results/COMPARISON/CROSSCHECK/" + program[0:8]

    if not os.path.exists(loc1):
        os.makedirs(loc1)
    if not os.path.exists(loc2):
        os.makedirs(loc2)

    try:  # loading
        start = str(program_info['programs'][0]['from'])
        stop = str(program_info['programs'][0]['upto'])

        geom = prio['geometry']
        vbc_chn = geom['channels']['eChannels']['VBC']
        hbcm_chn = geom['channels']['eChannels']['HBCm']
        time, t1, t2 = power['time'], min(power['time']), max(power['time'])
        voltage = power['voltage adjusted']
        power = power['power']

        tl, vl, hl = len(power[0]), len(vbc_chn), len(hbcm_chn)
        power_hbc_list, voltage_hbc_list = Z((tl, hl)), Z((tl, hl))
        power_vbc_list, voltage_vbc_list = Z((tl, vl)), Z((tl, vl))

        for h, ch in enumerate(hbcm_chn):
            power_hbc_list[:, h] = power[ch]
            voltage_hbc_list[:, h] = voltage[ch]

        for v, ch in enumerate(vbc_chn):
            power_vbc_list[:, v] = power[ch]
            voltage_vbc_list[:, v] = voltage[ch]

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  indiv. chan. loading failed')
        return

    if not compare_archive:
        # if not same shot, dont compare, because nonsense
        if program in compare_shots:
            it = compare_shots.index(program)
            [[compare_vbc, voltage_vbc, power_vbc],
             [compare_hbc, voltage_hbc, power_hbc], _] = comparison[it]

        else:
            compare_vbc = compare_hbc = voltage_vbc = \
                voltage_hbc = power_vbc = power_hbc = False

        if compare_vbc:
            if voltage_vbc:
                print(indent_level + "\tReading voltage:", data_names[it][1])
                voltage_vbc_c = np.loadtxt(data_names[it][1])
                time_c = np.linspace(t1, t2, len(voltage_vbc_c[0]))
            else:
                voltage_vbc_c = None
                time_c = None

            if power_vbc:
                print(indent_level + "\tReading power:", data_names[it][0])
                power_vbc_c = np.loadtxt(data_names[it][0])
                time_c = np.linspace(t1, t2, len(power_vbc_c[0]))
            else:
                power_vbc_c = None
                time_c = None
        else:
            voltage_vbc_c = None
            power_vbc_c = None
            time_c = None

        if compare_hbc:
            if voltage_hbc:
                print(indent_level + "\tReading volt:", data_names[it][3])
                voltage_hbc_c = np.loadtxt(data_names[it][3])
                time_c = np.linspace(t1, t2, len(voltage_hbc_c[0]))
            else:
                voltage_hbc_c = None

            if power_hbc:
                print(indent_level + "\tReading power:", data_names[it][2])
                power_hbc_c = np.loadtxt(data_names[it][2])
                time_c = np.linspace(t1, t2, len(power_hbc_c[0]))
            else:
                power_hbc_c = None
                time_c = None
        else:
            power_hbc_c = None
            voltage_hbc_c = None
            time_c = None

    elif compare_archive:
        print(indent_level + "\t\\\  loading archive channels ...")
        archive_volt = api.download_single(
            api.download_link(name='BoloAdjusted'),
            program_info=program_info,
            start_POSIX=start, stop_POSIX=stop)
        archive_power = api.download_single(
            api.download_link(name='BoloPowerRaw'),
            program_info=program_info,
            start_POSIX=start, stop_POSIX=stop)

        time_c = [(t - prio['t0']) / 1e9 for t in archive_volt['dimensions']]
        power_vbc = voltage_vbc = power_hbc = voltage_hbc = True
        ta1, ta2 = len(archive_volt["values"][0]), \
            len(archive_power["values"][0])
        voltage_vbc_c, voltage_hbc_c = Z((ta1, vl)), Z((ta1, hl))
        power_vbc_c, power_hbc_c = np.zeros((ta2, vl)), np.zeros((ta2, hl))

        try:
            for h, ch in enumerate(hbcm_chn):
                power_hbc_c[:, h] = archive_power["values"][ch]
                voltage_hbc_c[:, h] = archive_volt["values"][ch]
        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\ archive hbc channels failed',
                  exc_type, fname, exc_tb.tb_lineno)

        try:
            for v, ch in enumerate(vbc_chn):
                power_vbc_c[:, v] = archive_power["values"][ch]
                voltage_vbc_c[:, v] = archive_volt["values"][ch]
        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\  archive vbc channels failed',
                  exc_type, fname, exc_tb.tb_lineno)

    channels_individually(
        compare_shot=program,
        power_or_voltage='voltage',
        ylabels="voltage [V]",
        compare_vbc=voltage_vbc,
        vbc_list1=voltage_vbc_list,
        vbc_list2c=voltage_vbc_c,
        compare_hbc=voltage_hbc,
        hbc_list1=voltage_hbc_list,
        hbc_list2c=voltage_hbc_c,
        time=time,
        time_c=time_c,
        channels=[vbc_chn, hbcm_chn],
        date=program[0:8],
        indent_level=indent_level)
    channels_individually(
        compare_shot=program,
        power_or_voltage='power',
        ylabels="power [W]",
        compare_vbc=power_vbc,
        vbc_list1=power_vbc_list,
        vbc_list2c=power_vbc_c,
        compare_hbc=power_hbc,
        hbc_list1=power_hbc_list,
        hbc_list2c=power_hbc_c,
        time=time,
        time_c=time_c,
        channels=[vbc_chn, hbcm_chn],
        date=program[0:8],
        indent_level=indent_level)
    return


def channels_individually(
        compare_shot='20181010.032',
        power_or_voltage='voltage',
        ylabels='voltage [mV]',
        xlabels='time [s]',
        compare_vbc=False,
        vbc_list1=Z((128, 10000)),
        vbc_list2c=Z((128, 10000)),
        compare_hbc=False,
        hbc_list1=Z((128, 10000)),
        hbc_list2c=Z((128, 10000)),
        time=Z((10000)),
        time_c=Z((10000)),
        channels=[[], []],
        date='20181010',
        indent_level='\t'):
    """ Sets up the plot of all the camera array channels with possible
        comparisons.
    Args:
        compare_shot (0, str): shot that is compared
        power_or_voltage (1, str): show power or voltage
        ylabel (2, str) : Global ylabels
        xlabels (3, str): Global xlabels
        compare_vbc (4, bool): Whether to compare the VBC channel
        vbc_list1 (5, ndarray): Array giving the VBC channels
        vbc_list2c (6, ndarray): Array the first VBC is compared to
        compare_hbc (7, bool): Whether to compare the HBC channels
        hbc_list1 (8, ndarray): Array giving the HBC channels
        hbc_list2c (9, ndarray): Array the first HBC is compared to
        time (ndarray, required):
        time_c (ndarray, required):
        channels (10, list): list of channel arrays
        date (11, str): Date specified
        indent_level (12, str): Indentation level.
    Returns:
        None.
    Notes:
        None.
    """
    linesc = cycle(["-", "--", "-.", ":"])
    N = 64 if (compare_hbc or compare_vbc) else 32
    colorsc = cycle(cm.brg(np.linspace(0, 1, N)))

    try:
        if not compare_hbc and not compare_vbc:
            nrows, ncols = 2, 3
        elif (compare_hbc and not compare_vbc) or \
             (compare_vbc and not compare_hbc):
            pass
        elif compare_hbc and compare_vbc:
            nrows, ncols = 2, 6

        fig, axes = p.subplots(nrows=nrows, ncols=ncols)
        fig.set_size_inches(7. * ncols, 5. * nrows)

        for i, vec in enumerate(np.transpose(vbc_list1)[:]):
            axes[0, int(np.ceil(((i + 1) * ncols / 34)) - 1)].plot(
                time,
                vec, next(linesc), c=next(colorsc),
                label='E#' + str(channels[0][i]))

            if compare_vbc:
                axes[0, int(np.ceil(((i + 1) * ncols / 34)) - 1)].plot(
                    time_c,
                    np.transpose(vbc_list2c)[i],
                    next(linesc), c=next(colorsc),
                    label='C#' + str(channels[0][i]))

        for i, vec in enumerate(np.transpose(hbc_list1)[:]):
            axes[1, int(np.ceil(((i + 1) * ncols / 34)) - 1)].plot(
                time,
                vec, next(linesc), c=next(colorsc),
                label='E#' + str(channels[1][i]))

            if compare_hbc:
                axes[1, int(np.ceil(((i + 1) * ncols / 34)) - 1)].plot(
                    time_c,
                    np.transpose(hbc_list2c)[i],
                    next(linesc), c=next(colorsc),
                    label='C#' + str(channels[1][i]))

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\  ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  indiv chan plotting failed')
        return

    try:
        aX = []
        for i, row in enumerate(axes):
            for j, ax in enumerate(row):
                aX.append(ax)

        for i, ax in enumerate(aX):
            ax.grid(b=True, which='major', linestyle='-.')
            ax.legend()
            ax.set_xlabel(xlabels)
            ax.set_ylabel(ylabels)
            if i == 0:
                ax.set_title('#' + compare_shot)

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\  ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  indiv chan etcetera failed ')
        return

    try:
        p.tight_layout()
        if power_or_voltage == "voltage":
            fig.savefig('../results/COMPARISON/CHANNELS/' + date +
                        '/' + compare_shot + '.voltage.png',
                        bbox_inches='tight', dpi=169.0)
            fig_current_save('compare_shot_voltage', fig)
        elif power_or_voltage == 'power':
            fig.savefig('../results/COMPARISON/CHANNELS/' + date +
                        '/' + compare_shot + '.power.png',
                        bbox_inches='tight', dpi=169.0)
            fig_current_save('compare_shot_power', fig)
        p.close('all')

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  indiv chan saving failed')
        return


def plot_dens_diag(
        datas,
        times,
        mx_vals,
        pos_mx,
        date,
        shotno,
        program,
        program_info,
        base,
        indent_level):
    """ Searches and finds for the maximum Prad/ECRH power with density,
        temperature and W_dia at the same time.
    Args:
        datas: List of data vectors normalized in dimension and interpolated.
        times: List of time vectors, also interpolated.
        mx_vals: Maximum values of data vectors.
        pos_mx: Index positions of given maximum values.
        date: Date selected.
        shotno: XP ID selected.
        program: Date and XP ID string cat.
        program_info: HTTP resp from logbook for program.
        base: base directory.
        indent_level: Indentation level.
    Returns:
        ecrh_off: Time where the ECRH is off.
    Notes:
        None.
    """
    print(indent_level + '>> plot crit properties')

    try:
        lines = ["-", "--", "-.", ":"]
        linesc = cycle(lines)
        colors = cm.brg(np.linspace(0, 1, 7))
        colorsc = cycle(colors)

        fig, host = p.subplots()
        fig.subplots_adjust(right=0.75)
        diamagax = host.twinx()
        eldensax = host.twinx()
        eldensax.spines["right"].set_position(("axes", 1.15))
        tempax = host.twinx()
        tempax.spines["right"].set_position(('axes', 1.25))

        powers_ecrh, = host.plot(
            times[2], smoothing(datas[2]),
            next(linesc), c=next(colorsc), linewidth=1.0,
            label="ECRH")
        powers_prad, = host.plot(
            times[0], smoothing(datas[0]),
            next(linesc), c=next(colorsc), linewidth=1.0,
            label="P$_{rad}$ HBC")
        powers_pradV, = host.plot(
            times[1], smoothing(datas[1]),
            next(linesc), c=next(colorsc), linewidth=1.0,
            label="P$_{rad}$ VBC")
        diamag, =  diamagax.plot(
            times[3], smoothing(datas[3]),
            next(linesc), c=next(colorsc), linewidth=1.0,
            label="W$_{dia}$")
        eldens, = eldensax.plot(
            times[4], smoothing(datas[4]),
            next(linesc), c=next(colorsc), linewidth=1.0,
            label="n$_{el}$ lin.")
        tempcore, = tempax.plot(
            times[6], smoothing(datas[6]),
            next(linesc), c=next(colorsc), linewidth=1.0,
            label='T$_{e, core}$')
        tempout, = tempax.plot(
            times[5], smoothing(datas[5]),
            next(linesc), c=next(colorsc), linewidth=1.0,
            label='T$_{e, out}$')

        t = int(pos_mx[2])
        while (datas[2][t] > 1e-1):
            t += 1

        host.set_xlim(-0.5, times[2][t] + 1.0)
        host.set_xlabel('time / s')
        host.set_ylabel('power / MW')
        diamagax.set_ylabel('W$_{dia}$ / MJ')
        eldensax.set_ylabel('density / 1e19 m$^{-2}$')
        tempax.set_ylabel('temperature / keV')
        title_str = "#" + date + "." + shotno + ' crit values diagnostic'
        host.set_title(title_str)

        location = base + "/../results/COMPARISON/POWER_DIAGNOSTIC/" + date
        if not os.path.exists(location):
            os.makedirs(location)

        fig.savefig("../results/COMPARISON/POWER_DIAGNOSTIC/" + date +
                    "/power_diagnostic_" + shotno + ".png")
        fig_current_save('power_diagnostic', fig)

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  dens diag plot setup failed')
        return

    try:
        lines = [powers_prad, powers_pradV, powers_ecrh,
                 diamag, eldens, tempout, tempcore]
        tags = ['mx pradH', 'mx pradV', 'mx ECRH', 'mx Wdia',
                'mx dens', 'mx tempout', 'mx tempcore']
        host.legend(lines, [l.get_label() for l in lines])

        for ax in [diamagax, eldensax]:
            ax.set_frame_on(True)
            ax.patch.set_visible(False)
            ax.spines["right"].set_visible(True)

        host.yaxis.label.set_color(powers_ecrh.get_color())
        host.tick_params(axis='y', colors=powers_ecrh.get_color())

        eldensax.yaxis.label.set_color(eldens.get_color())
        eldensax.spines["right"].set_edgecolor(eldens.get_color())
        eldensax.tick_params(axis='y', colors=eldens.get_color())

        diamagax.spines["right"].set_edgecolor(diamag.get_color())
        diamagax.yaxis.label.set_color(diamag.get_color())
        diamagax.tick_params(axis='y', colors=diamag.get_color())

        tempax.spines["right"].set_edgecolor(tempcore.get_color())
        tempax.yaxis.label.set_color(tempcore.get_color())
        tempax.tick_params(axis='y', colors=tempcore.get_color())

        for i in range(0, len(mx_vals)):
            try:
                host.axvline(
                    x=times[i][pos_mx[i]],
                    color=lines[i].get_color())
                host.text(
                    times[i][pos_mx[i] + 50],
                    datas[i][pos_mx[i]] * 0.75,
                    tags[i] + ' ' + str(round(mx_vals[i], 3)),
                    rotation=90, verticalalignment='center',
                    color=lines[i].get_color())
            except Exception:
                pass

        fig.savefig("../results/COMPARISON/POWER_DIAGNOSTIC/" + date +
                    "/power_diagnostic_" + shotno + ".png")
        fig_current_save('power_diagnostic', fig)
        p.close('all')

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  dens diag plot failed')

    return times[2][t] + 0.25


def balance_plot(
        interpDatas,
        pbalance,
        BSTime,
        ecrh_off,
        date,
        shotno,
        base,
        indent_level='\t'):
    """ Plot power results.
    Args:
        interpDatas: Interpolated/stretched data to calculate power balance.
        pbalance: Power balance array.
        BSTime: Time vector.
        base: Base directory.
        ecrh_off: Time value (s) where ECRH off.
    Returns:
        None.
    Notes:
        None.
    """
    print(indent_level + '>> trigger check plot')

    lines = ["-", "--", "-.", ":"]
    linesc = cycle(lines)
    colors = cm.brg(np.linspace(0, 1, 5))
    colorsc = cycle(colors)

    fig, host = p.subplots()
    fig.subplots_adjust(right=0.75)
    diamagax = host.twinx()
    eldensax = host.twinx()
    eldensax.spines["right"].set_position(("axes", 1.15))

    prad, = host.plot(
        BSTime, interpDatas[1], next(linesc),
        color=next(colorsc), label='P$_{rad}$')
    ecrh, = host.plot(
        BSTime, interpDatas[2], next(linesc),
        color=next(colorsc), label='P$_{ECRH}$')
    wdiadt, = host.plot(BSTime, sc.signal.savgol_filter(np.gradient(
        interpDatas[3], BSTime[1] - BSTime[0]), 51, 2), next(linesc),
        color=next(colorsc), label='dW$_{dia}$/dt')
    balance, = host.plot(
        BSTime, pbalance, next(linesc),
        color=next(colorsc), label='power balance')

    wdia, = diamagax.plot(
        BSTime, interpDatas[3], next(linesc),
        color=next(colorsc), label='W$_{dia}$')
    ne, = eldensax.plot(
        BSTime, interpDatas[4], next(linesc),
        color=next(colorsc), label='n$_{el}$')

    host.legend()
    host.set_xlim(-.05, ecrh_off)
    host.set_xlabel('time / s')
    host.set_ylabel('power / MW')

    diamagax.set_ylabel('W$_{dia}$ / MJ')
    eldensax.set_ylabel('density / 1e19 m$^{-2}$')

    lines = [prad, ecrh, wdiadt, balance, wdia, ne]
    host.legend(lines, [l.get_label() for l in lines])

    for ax in [diamagax, eldensax]:
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        ax.spines["right"].set_visible(True)

    host.yaxis.label.set_color(balance.get_color())
    host.tick_params(axis='y', colors=balance.get_color())
    foo = max(max(np.abs(interpDatas[1])), max(np.abs(interpDatas[2])),
              max(pbalance),
              max(sc.signal.savgol_filter(np.gradient(
                  interpDatas[3], BSTime[1] - BSTime[0]), 51, 2)))
    host.set_ylim(-1.05 * foo, 1.05 * foo)

    eldensax.yaxis.label.set_color(ne.get_color())
    eldensax.spines["right"].set_edgecolor(ne.get_color())
    eldensax.tick_params(axis='y', colors=ne.get_color())
    eldensax.set_ylim(-1.05 * max(np.abs(interpDatas[4])),
                      1.05 * max(np.abs(interpDatas[4])))

    diamagax.spines["right"].set_edgecolor(wdia.get_color())
    diamagax.yaxis.label.set_color(wdia.get_color())
    diamagax.tick_params(axis='y', colors=wdia.get_color())
    diamagax.set_ylim(-1.05 * max(interpDatas[3]), 1.05 * max(interpDatas[3]))
    diamagax.spines["left"].set_edgecolor(balance.get_color())

    location = base + "/../results/COMPARISON/POWER_BALANCE/" + date
    if not os.path.exists(location):
        os.makedirs(location)
    fig_current_save('power_balance', fig)
    fig.savefig("../results/COMPARISON/POWER_BALANCE/" + date +
                "/power_balance_" + shotno + ".png")
    p.close('all')

    return


def op_2018_statistics(
        data={'none': None},
        debug=False):

    n_dates = len(data.keys())
    if debug:
        print(n_dates)

    date_ints = np.zeros((n_dates))
    kappa, rohm, taum, std, abs_offs, slope, raw_max, \
        raw_min, adj_max, adj_median = [], [], [], [], [], [], [], [], [], []

    for i, key in enumerate(data.keys()):
        date_ints[i] = int(key[:8]) * 100 + int(key[9:])

        kappa.append(data[key]['Kappam'])
        rohm.append(data[key]['rohm'])
        kappa.append(data[key][''])
        kappa.append(data[key][''])
        kappa.append(data[key][''])
        kappa.append(data[key][''])
        kappa.append(data[key][''])
        kappa.append(data[key][''])
        kappa.append(data[key][''])

    print([
        i  # date_ints[i]
        for i, arr in enumerate(kappa) if (arr != [0.0] * 128)])

    return


def peak_database_plots(
        data={'none': None},
        width=0.025,
        filter_value='none',
        filter_key='none'):
    N = 4
    fig, axes = p.subplots(2, 2)  # sharex=False)
    fig.set_size_inches(8., 4.)

    # radiation increase over puff increase
    L1, = axes[0, 0].plot(
        [data['PradHBCm']['source']['peak'][i] -
         v for i, v in enumerate(data['PradHBCm']['source']['base'])],
        [(data['PradHBCm']['peak'][i] - v) / 1.e6
         for i, v in enumerate(data['PradHBCm']['base'])],
        ls='', c='r', marker='x', label='HBCm')
    L2, = axes[0, 0].plot(
        [data['PradVBC']['source']['peak'][i] -
         v for i, v in enumerate(data['PradVBC']['source']['base'])],
        [(data['PradVBC']['peak'][i] - v) / 1.e6
         for i, v in enumerate(data['PradVBC']['base'])],
        ls='', c='b', marker='+', label='VBC')
    axes[0, 0].set_xlabel('$\Delta\Gamma$ [a.u.]')
    axes[0, 0].set_ylabel('$\Delta$P$_{rad}$ [MW]')

    # radiation increase over peak width
    axes[1, 0].plot(
        [1e0 * t for t in data['PradHBCm']['source']['width']],
        [(data['PradHBCm']['peak'][i] - v) / 1.e6
         for i, v in enumerate(data['PradHBCm']['base'])],
        ls='', c='r', marker='x')
    axes[1, 0].plot(
        [1e0 * t for t in data['PradVBC']['source']['width']],
        [(data['PradVBC']['peak'][i] - v) / 1.e6
         for i, v in enumerate(data['PradVBC']['base'])],
        ls='', c='b', marker='+')
    axes[1, 0].set_xlabel('$T_{\Gamma}$ [s]')
    axes[1, 0].set_ylabel('$\Delta$P$_{rad}$ [MW]')

    # radiation increase over gas integral
    # (assume gaussian function shape with s as width
    # 0.2 sigma -> int 0.501; 0.1 sigma -> 0.2506)
    axes[0, 1].plot(
        [np.sqrt(2. * np.pi) * v * (
         data['PradHBCm']['source']['peak'][i] -
         data['PradHBCm']['source']['base'][i])
         for i, v in enumerate(data['PradHBCm']['source']['width'])],
        [(data['PradHBCm']['peak'][i] - v) / 1.e6
         for i, v in enumerate(data['PradHBCm']['base'])],
        ls='', c='r', marker='x')
    axes[0, 1].plot(
        [np.sqrt(2. * np.pi) * v * (
         data['PradVBC']['source']['peak'][i] -
         data['PradVBC']['source']['base'][i])
         for i, v in enumerate(data['PradVBC']['source']['width'])],
        [(data['PradVBC']['peak'][i] - v) / 1.e6
         for i, v in enumerate(data['PradVBC']['base'])],
        ls='', c='b', marker='+')
    axes[0, 1].set_xlabel('$\int\Gamma$ [a.u.]')
    axes[0, 1].set_ylabel('$\Delta$P$_{rad}$ [MW]')

    # time delay of peak over puff intensity
    axes[1, 1].plot(
        [v - data['PradHBCm']['source']['base'][i]
         for i, v in enumerate(data['PradHBCm']['source']['peak'])],
        [(data['PradHBCm']['time'][i] - v)
         for i, v in enumerate(data['PradHBCm']['source']['time'])],
        ls='', c='r', marker='x')
    axes[1, 1].plot(
        [v - data['PradVBC']['source']['base'][i]
         for i, v in enumerate(data['PradVBC']['source']['peak'])],
        [(data['PradVBC']['time'][i] - v)
         for i, v in enumerate(data['PradVBC']['source']['time'])],
        ls='', c='b', marker='+')
    axes[1, 1].set_xlabel('$\Delta\Gamma$ [a.u.]')
    axes[1, 1].set_ylabel('$\Delta$T$_{peak}$ [s]')

    # limits dPrad v dGamma
    axes[0, 0].set_xlim(0.0, 35.)
    axes[0, 0].set_ylim(0.0, 1.5)
    # limits dPrad v TGamma
    axes[1, 0].set_xlim(.15, 1.3)
    axes[1, 0].set_ylim(0.0, 1.5)
    # limits dPrad v IntGamma
    axes[0, 1].set_xlim(0.0, 50.)
    axes[0, 1].set_ylim(0.0, 1.5)
    # limits dTPeaks v dGamma
    axes[1, 1].set_xlim(0.0, 35.)
    axes[1, 1].set_ylim(0.0, .5)

    # legend
    axes[0, 0].legend(
        [L1, L2], [l.get_label() for l in [L1, L2]],
        loc='upper center', bbox_to_anchor=(.5, 1.3),
        ncol=2, fancybox=True, shadow=False,
        handletextpad=0.1, labelspacing=0.2, handlelength=1.)

    # saving
    fig_current_save('peaks_filt_db', fig)
    if filter_value is not None and filter_key is not None:
        fig.savefig(
            '../results/PEAKS/image/peaks_key_' +
            filter_key + '_val_' + filter_value +
            '_' + str(width) + 'ms.pdf',
            bbox_inches='tight', dpi=169)

    else:
        fig.savefig(
            '../results/PEAKS/image/peaks_key_all_' +
            str(width) + 'ms.pdf',
            bbox_inches='tight', dpi=169)

    p.close('all')
    return


def peak_scatter_colorbar(
        data={'none': None},
        width=0.025,
        filter_value='none',
        filter_key='none'):
    hbcm, vbc = data['PradHBCm'], data['PradVBC']
    sourceH, sourceV = hbcm['source'], vbc['source']

    neH, TeInH, TeOutH, ecrhH = \
        hbcm['params']['n_e lint'], hbcm['params']['T_e ECE core'], \
        hbcm['params']['T_e ECE out'], hbcm['params']['ECRH']
    neV, TeInV, TeOutV, ecrhV = \
        vbc['params']['n_e lint'], vbc['params']['T_e ECE core'], \
        vbc['params']['T_e ECE out'], vbc['params']['ECRH']

    M, N = 2, 2
    fig, axes = p.subplots(N, M)  # sharex=False)
    fig.set_size_inches(4.* M, 2. * N)

    def scatter_colorbar(
            ax, map, x, y, z, s, m='x',
            label=None, clabel=None):
        sc = ax.scatter(
            x, y, c=z, s=10., cmap=map, marker=m,
            vmin=np.min(z), vmax=np.max(z), label=label)
        if label == 'VBC':
            return (ax)
        bar = fig.colorbar(sc, ax=ax)  # p.colorbar(sc)
        if clabel is not None:
            bar.set_label(clabel)  # , rotation=270)
        return (ax)

    def plot_scatter_cams(
            ax, xs, ys, zs, ss,
            xlabel, ylabel, clabel, xlim=None, ylim=None):
        for cam, marker, cmap, X, Y, Z, S in zip(
                ['HBCm', 'VBC'],
                ['<', '>'],
                ['viridis', 'viridis'],  # ['hot', 'cool'],
                xs, ys, zs, ss):
            scatter_colorbar(
                ax=ax, map=p.cm.get_cmap(cmap), label=cam,
                clabel=clabel, m=marker,
                x=X, y=Y, z=Z, s=S + 10.)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        return

    def cinf(arr, a, b):
        # po = np.where(arr == np.nan)[0]
        # na = np.where(arr == np.posinf)[0]
        # ng = np.where(arr == np.neginf)[0]

        # arr[po] = a
        # arr[ng] = b
        # arr[na] = 0.0
        # return(np.nan_to_num(arr, nan=0.0, posinf=a, neginf=b))
        return(np.clip(arr, a_min=b, a_max=a))

    plot_scatter_cams(  # radiation increase over puff increase
        axes[0, 0],
        xs=[sourceH['peak'] - sourceH['base'],
            sourceV['peak'] - sourceV['base']],
        ys=[(hbcm['peak'] - hbcm['base']) / 1.e6,
            (vbc['peak'] - vbc['base']) / 1.e6],
        zs=[neH['value'] / 1.e19, neV['value'] / 1.e19],
        ss=[1. / abs(hbcm['time'] - neH['time']),
            1. / abs(vbc['time'] - neV['time'])],
        xlim=[0.0, 35.], ylim=[0.0, 1.5],
        xlabel='$\Delta\Gamma$ [a.u.]',
        ylabel='$\Delta$P$_{rad}$ [MW]',
        clabel='n$_{e}$ [10$^{19}$m$^{-3}$]')

    plot_scatter_cams(  # radiation increase over peak width
        axes[1, 0],
        xs=[sourceH['width'], sourceV['width']],
        ys=[(hbcm['peak'] - hbcm['base']) / 1.e6,
            (vbc['peak'] - vbc['base']) / 1.e6],
        zs=[ecrhH['value'] / 1.e3, ecrhV['value'] / 1.e3],
        ss=[1. / abs(hbcm['time'] - ecrhH['time']),
            1. / abs(vbc['time'] - ecrhV['time'])],
        xlim=[0.15, 1.3], ylim=[0.0, 1.5],
        xlabel='T$_{\Gamma}$ [s]',
        ylabel='$\Delta$P$_{rad}$ [MW]',
        clabel='ECRH [MW]')

    # plot_scatter_cams(  # radiation increase over peak width
    #     axes[1, 0],
    #     xs=[sourceH['width'], sourceV['width']],
    #     ys=[(hbcm['peak'] - hbcm['base']) / 1.e6,
    #         (vbc['peak'] - vbc['base']) / 1.e6],
    #     zs=[neH['value'] / 1.e19, neV['value'] / 1.e19],
    #     ss=[1. / abs(hbcm['time'] - neH['time']),
    #         1. / abs(vbc['time'] - neV['time'])],
    #     xlim=[0.0, 2.5], ylim=[0.0, 1.5],
    #     xlabel='T$_{\Gamma}$ [s]',
    #     ylabel='$\Delta$P$_{rad}$ [MW]',
    #     clabel='n$_{e,lin}$ [10$^{19}$m$^{-3}$]')

    # radiation increase over gas integral
    # (assume gaussian function shape with s as width
    # 0.2 sigma -> int 0.501; 0.1 sigma -> 0.2506)
    # plot_scatter_cams(
    #     axes[2, 0],
    #     xs=[np.sqrt(2. * np.pi) * sourceH['width'] * (
    #             sourceH['peak'] - sourceH['base']),
    #         np.sqrt(2. * np.pi) * sourceV['width'] * (
    #             sourceV['peak'] - sourceV['base'])],
    #     ys=[(hbcm['peak'] - hbcm['base']) / 1.e6,
    #         (vbc['peak'] - vbc['base']) / 1.e6],
    #     zs=[neH['value'] / 1.e19, neV['value'] / 1.e19],
    #     ss=[1. / abs(hbcm['time'] - neH['time']),
    #         1. / abs(vbc['time'] - neV['time'])],
    #     xlim=[0.0, 50.], ylim=[0.0, 1.5],
    #     xlabel='$\int\Gamma$ [a.u.]',
    #     ylabel='$\Delta$P$_{rad}$ [MW]',
    #     clabel='n$_{e,lin}$ [10$^{19}$m$^{-3}$]')

    # plot_scatter_cams(  # time delay of peak over puff intensity
    #     axes[3, 0],
    #     xs=[sourceH['peak'] - sourceH['base'],
    #         sourceV['peak'] - sourceV['base']],
    #     ys=[hbcm['time'] - sourceH['time'], vbc['time'] - sourceV['time']],
    #     zs=[neH['value'] / 1.e19, neV['value'] / 1.e19],
    #     ss=[1. / abs(hbcm['time'] - neH['time']),
    #         1. / abs(vbc['time'] - neV['time'])],
    #     xlim=[0.0, 50.], ylim=[0.0, .8],
    #     xlabel='$\Delta\Gamma$ [a.u.]',
    #     ylabel='$\Delta$T$_{peak}$ [s]',
    #     clabel='n$_{e,lin}$ [10$^{19}$m$^{-3}$]')

    plot_scatter_cams(  # frad over ecrh with ne scale
        axes[0, 1],
        xs=[ecrhH['value'] / 1.e3, ecrhV['value'] / 1.e3],
        ys=[cinf((hbcm['peak'] / ecrhH['value']) / 1.e3, 2., .0),
            cinf((vbc['peak'] / ecrhV['value']) / 1.e3, 2., .0)],
        zs=[neH['value'] / 1.e19, neV['value'] / 1.e19],
        ss=[1. / abs(hbcm['time'] - neH['time']),
            1. / abs(vbc['time'] - neV['time'])],
        xlim=[0.5, 7.], ylim=[0.0, 1.25],
        xlabel='ECRH [MW]',
        ylabel='f$_{rad}$ [a.u.]',
        clabel='n$_{e}$ [10$^{19}$m$^{-3}$]')

    # plot_scatter_cams(  # radiation increase over gas integral with ecrh
    #     axes[1, 1],
    #     xs=[np.sqrt(2. * np.pi) * sourceH['width'] * (
    #             sourceH['peak'] - sourceH['base']),
    #         np.sqrt(2. * np.pi) * sourceV['width'] * (
    #             sourceV['peak'] - sourceV['base'])],
    #     ys=[(hbcm['peak'] - hbcm['base']) / 1.e6,
    #         (vbc['peak'] - vbc['base']) / 1.e6],
    #     zs=[ecrhH['value'] / 1.e3, ecrhV['value'] / 1.e3],
    #     ss=[1. / abs(hbcm['time'] - ecrhH['time'] + 1.),
    #         1. / abs(vbc['time'] - ecrhV['time'] + .1)],
    #     xlim=[0.0, 50.], ylim=[0.0, 3.],
    #     xlabel='$\int\Gamma$ [a.u.]',
    #     ylabel='$\Delta$P$_{rad}$ [MW]',
    #     clabel='P$_{ECRH}$ [MW]')

    # plot_scatter_cams(  # radiation increase over ecrh with ne
    #     axes[2, 1],
    #     xs=[ecrhH['value'] / 1.e3, ecrhV['value'] / 1.e3],
    #     ys=[(hbcm['peak'] - hbcm['base']) / 1.e6,
    #         (vbc['peak'] - vbc['base']) / 1.e6],
    #     zs=[neH['value'] / 1.e19, neV['value'] / 1.e19],
    #     ss=[1. / abs(hbcm['time'] - neH['time']),
    #         1. / abs(vbc['time'] - neV['time'])],
    #     xlim=[0.5, 7.], ylim=[0.0, 3.],
    #     xlabel='P$_{ECRH}$ [MW]',
    #     ylabel='$\Delta$P$_{rad}$ [MW]',
    #     clabel='n$_{e,lin}$ [10$^{19}$m$^{-3}$]')

    plot_scatter_cams(  # ne over ecrh with frad scale
        axes[1, 1],
        xs=[ecrhH['value'] / 1.e3, ecrhV['value'] / 1.e3],
        ys=[neH['value'] / 1.e19, neV['value'] / 1.e19],
        zs=[cinf((hbcm['peak'] / ecrhH['value']) / 1.e3, 2., .0),
            cinf((vbc['peak'] / ecrhV['value']) / 1.e3, 2., .0)],
        ss=[1. / abs(hbcm['time'] - neH['time']),
            1. / abs(vbc['time'] - neV['time'])],
        xlim=[0.0, 7.], ylim=[0.0, 14.],
        xlabel='ECRH [MW]',
        clabel='f$_{rad}$ [a.u.]',
        ylabel='n$_{e}$ [10$^{19}$m$^{-3}$]')

    axes[0, 0].legend(
        loc='upper center', bbox_to_anchor=(.5, 1.3),
        ncol=2, fancybox=True, shadow=False,
        handletextpad=0.1, labelspacing=0.2, handlelength=1.)

    fig_current_save('peaks_filt_param_db', fig)
    if filter_value is not None and filter_key is not None:
        fig.savefig(
            '../results/PEAKS/image/peaks_param_key_' +
            filter_key + '_val_' + filter_value +
            '_' + str(width) + 'ms.pdf', bbox_inches='tight',
            dpi=169.)
    else:
        fig.savefig(
            '../results/PEAKS/image/peaks_param_key_all_' +
            str(width) + 'ms.pdf', bbox_inches='tight', dpi=169.)
    p.close('all')
    return


def peak_plot_wrapper(
        date,
        id,
        data={},
        peaks={},
        time=0.075,
        debug=False):
    N = len(peaks.keys())
    M = len(np.where([
        'QTB' in s for s in data.keys()])[0])
    fig, axes = p.subplots(N - M + 2, 1, sharex=True)
    fig.set_size_inches(4., 2. * (N - M + 2))

    i = 0
    for j, key in enumerate(peaks.keys()):
        try:
            if (key in peaks.keys()) and ('QTB' not in key):
                plot_peaks(
                    ax=axes[i], label=key,
                    x=data[key]['dimensions'],
                    y=data[key]['values'],
                    peak=peaks[key]['index'],
                    start=peaks[key]['start_index'],
                    stop=peaks[key]['stop_index'])
                axes[i].legend()
                i += 1

            elif ('QTB' in key):
                if ('n_e' in key):
                    k = -2
                elif ('T_e' in key):
                    k = -1

                plot_peaks(
                    ax=axes[k], label=key,
                    x=data[key]['dimensions'],
                    y=data[key]['values'],
                    peak=peaks[key]['index'],
                    start=peaks[key]['start_index'],
                    stop=peaks[key]['stop_index'])

        except Exception:
            print('\\\ failed ' + key + ' plot')

    # fig_current_save('peaks_find', fig)
    fig.savefig('../results/PEAKS/peaks_' + date +
                '_' + str(id).zfill(3) + '_' +
                str(time * 1.e3) + 'ms.png')
    p.close()  # p.show()
    return


def plot_peaks(
        x,
        y,
        peak,
        start,
        stop,
        ax,
        label,
        debug=False):

    if ax is None:
        fig, ax = p.subplots(1, 1)

    if np.max(y) > 0:
        order = math.floor(math.log(np.max(y), 10))
        f = 1. / 10**order if order > 0 else 1.
        y = f * y  # rescale

    # plotting the data raw
    ax.plot(x, y, label=label,
            c='k', marker='.', markersize=1.,
            lw=0.75, alpha=0.5, ls='-')

    # index, time, start index, stop index, point width, time width
    for i, position in enumerate(peak):
        ax.plot(
            [x[position], x[position]],
            [y[position], y[position]],
            c='r', marker='x', ls='None')
        ax.plot(
            [x[start[i]], x[stop[i]]],
            [y[start[i]], y[stop[i]]],
            c='r', ls='-.', alpha=0.5, lw=0.5)
    return  # (ax)


def plot_plasma_peak_params(
        D,
        label,
        debug=False):

    N, M = 2, 4
    fig, axes = p.subplots(N, M)
    fig.set_size_inches(M * 5., 3. * N)
    c1, c2, c3 = 'k', 'orange', 'cyan'

    # dPrad vs ne
    axes[0, 0].plot(
        [v / 1.e19 for v in
         D['PradHBCm']['params']['n_e lint']['start_val']],
        [(v - D['PradHBCm']['start_val'][i]) / 1.e6
         for i, v in enumerate(D['PradHBCm']['peak_val'])],
        label='HBCm', c='r', marker='x', markersize=2., ls='None')
    axes[0, 0].plot(
        [v / 1.e19 for v in
         D['PradVBC']['params']['n_e lint']['start_val']],
        [(v - D['PradVBC']['start_val'][i]) / 1.e6
         for i, v in enumerate(D['PradVBC']['peak_val'])],
        label='VBC', c='b', marker='x', markersize=2., ls='None')
    axes[0, 0].set_xlim(2.5, 15.0)
    axes[0, 0].set_ylim(0.0, 2.0)
    axes[0, 0].set_xlabel('n$_{e,lin}$ [10$^{19}$ m$^{-3}$]')
    axes[0, 0].set_ylabel('$\Delta$P$_{rad}$ [MW]')

    # dPrad vs dne
    axes[1, 0].plot(
        [(v - D['PradHBCm']['params']['n_e lint']['start_val'][i]) / 1.e19
        for i, v in enumerate(D['PradHBCm']['params']['n_e lint']['peak_val'])],
        [(v - D['PradHBCm']['start_val'][i]) / 1.e6
         for i, v in enumerate(D['PradHBCm']['peak_val'])],
        label='HBCm', c='r', marker='x', markersize=2., ls='None')
    axes[1, 0].plot(
        [(v - D['PradVBC']['params']['n_e lint']['start_val'][i]) / 1.e19
        for i, v in enumerate(D['PradVBC']['params']['n_e lint']['peak_val'])],
        [(v - D['PradVBC']['start_val'][i]) / 1.e6
         for i, v in enumerate(D['PradVBC']['peak_val'])],
        label='VBC', c='b', marker='x', markersize=2., ls='None')
    axes[1, 0].set_xlim(-5., 10.)
    axes[1, 0].set_ylim(0.0, 2.0)
    axes[1, 0].set_xlabel('$\Delta$n$_{e,lin}$ [10$^{19}$ m$^{-3}$]')
    axes[1, 0].set_ylabel('$\Delta$P$_{rad}$ [MW]')

    # dTe vs ne
    axes[0, 1].plot(
        [v / 1.e19 for v in
         D['T_e ECE core']['params']['n_e lint']['start_val']],
        [(D['T_e ECE core']['peak_val'][i] - v) / v
         for i, v in enumerate(D['T_e ECE core']['start_val'])],
        label='core', c=c1, marker='x', markersize=2., ls='None')
    axes[0, 1].plot(
        [v / 1.e19 for v in
         D['T_e ECE out']['params']['n_e lint']['start_val']],
        [(D['T_e ECE out']['peak_val'][i] - v) / v
         for i, v in enumerate(D['T_e ECE out']['start_val'])],
        label='out', c=c2, marker='x', markersize=2., ls='None')
    # axes[0, 1].set_xlim(1.5, 3.0)
    axes[0, 1].set_ylim(-5., 5.)
    axes[0, 1].set_xlabel('n$_{e,lin}$ [10$^{19}$ m$^{-3}$]')
    axes[0, 1].set_ylabel('$\Delta$T$_{e, norm}$ [a.u.]')

    # dTe vs dne
    axes[1, 1].plot(
        [(v - D['T_e ECE core']['params']['n_e lint']['start_val'][i]) / 1.e19
        for i, v in enumerate(D['T_e ECE core']['params'][
            'n_e lint']['peak_val'])],
        [(D['T_e ECE core']['peak_val'][i] - v) / v
         for i, v in enumerate(D['T_e ECE core']['start_val'])],
        label='core', c=c1, marker='x', markersize=2., ls='None')
    axes[1, 1].plot(
        [(v - D['T_e ECE out']['params']['n_e lint']['start_val'][i]) / 1.e19
        for i, v in enumerate(D['T_e ECE out']['params'][
            'n_e lint']['peak_val'])],
        [(D['T_e ECE out']['peak_val'][i] - v) / v
         for i, v in enumerate(D['T_e ECE out']['start_val'])],
        label='out', c=c2, marker='x', markersize=2., ls='None')
    # axes[1, 1].set_xlim(-2.5, 2.5)
    axes[1, 1].set_ylim(-5., 5.)
    axes[1, 1].set_xlabel('$\Delta$n$_{e,lin}$ [10$^{19}$ m$^{-3}$]')
    axes[1, 1].set_ylabel('$\Delta$T$_{e, norm}$ [a.u.]')

    # dTe vs dBG011 Main
    axes[0, 2].plot(
        [v - D['T_e ECE core']['params']['Main valve BG011']['start_val'][i]
        for i, v in enumerate(D['T_e ECE core']['params'][
            'Main valve BG011']['peak_val'])],
        [(D['T_e ECE core']['peak_val'][i] - v) / v
         for i, v in enumerate(D['T_e ECE core']['start_val'])],
        label='core', c=c1, marker='x', markersize=2., ls='None')
    axes[0, 2].plot(
        [v - D['T_e ECE out']['params']['Main valve BG011']['start_val'][i]
        for i, v in enumerate(D['T_e ECE out']['params'][
            'Main valve BG011']['peak_val'])],
        [(D['T_e ECE out']['peak_val'][i] - v) / v
         for i, v in enumerate(D['T_e ECE out']['start_val'])],
        label='out', c=c2, marker='x', markersize=2., ls='None')

    # dTe vs dBG031 Main
    axes[0, 2].plot(
        [v - D['T_e ECE core']['params']['Main valve BG031']['start_val'][i]
        for i, v in enumerate(D['T_e ECE core']['params'][
            'Main valve BG031']['peak_val'])],
        [(D['T_e ECE core']['peak_val'][i] - v) / v
         for i, v in enumerate(D['T_e ECE core']['start_val'])],
        c=c1, marker='s', markersize=2., ls='None')
    axes[0, 2].plot(
        [v - D['T_e ECE out']['params']['Main valve BG031']['start_val'][i]
        for i, v in enumerate(D['T_e ECE out']['params'][
            'Main valve BG031']['peak_val'])],
        [(D['T_e ECE out']['peak_val'][i] - v) / v
         for i, v in enumerate(D['T_e ECE out']['start_val'])],
        c=c2, marker='s', markersize=2., ls='None')
    axes[0, 2].set_xlim(-5., 5.)
    axes[0, 2].set_ylim(-5., 5.)
    axes[0, 2].set_xlabel('$\Delta$f$_{main valve}$ [mbar$\cdot$l/s]')
    axes[0, 2].set_ylabel('$\Delta$T$_{e, norm}$ [a.u.]')

    # dne vs dBG Main
    axes[1, 2].plot(
        [v - D['n_e lint']['params']['Main valve BG011']['start_val'][i]
        for i, v in enumerate(D['n_e lint']['params'][
            'Main valve BG011']['peak_val'])],
        [(v - D['n_e lint']['start_val'][i]) / 1.e19
         for i, v in enumerate(D['n_e lint']['peak_val'])],
        label='BG011', c=c1, marker='x', markersize=2., ls='None')
    axes[1, 2].plot(
        [v - D['n_e lint']['params']['Main valve BG031']['start_val'][i]
        for i, v in enumerate(D['n_e lint']['params'][
            'Main valve BG031']['peak_val'])],
        [(v - D['n_e lint']['start_val'][i]) / 1.e19
         for i, v in enumerate(D['n_e lint']['peak_val'])],
        label='BG031', c=c2, marker='s', markersize=2., ls='None')
    axes[1, 2].set_xlim(-25., 25.)
    axes[1, 2].set_ylim(0.0, 6.)
    axes[1, 2].set_xlabel('$\Delta$f$_{main valve}$ [mbar$\cdot$l/s]')
    axes[1, 2].set_ylabel('$\Delta$n$_{e, lin}$ [10$^{19}$m$^{-3}$]')

    # dne vs dQSQ
    # axes[0, 3].plot(
    #     [v - D['n_e lint']['params']['QSQ Feedback AEH51']['start_val'][i]
    #     for i, v in enumerate(D['n_e lint']['params'][
    #         'QSQ Feedback AEH51']['peak_val'])],
    #     [(v - D['n_e lint']['start_val'][i]) / 1.e19
    #      for i, v in enumerate(D['n_e lint']['peak_val'])],
    #     label='AEH51', c=c1, marker='x', markersize=2., ls='None')
    # axes[0, 3].plot(
    #     [v - D['n_e lint']['params']['QSQ Feedback AEH30']['start_val'][i]
    #     for i, v in enumerate(D['n_e lint']['params'][
    #         'QSQ Feedback AEH30']['peak_val'])],
    #     [(v - D['n_e lint']['start_val'][i]) / 1.e19
    #      for i, v in enumerate(D['n_e lint']['peak_val'])],
    #     label='AEH30', c=c2, marker='x', markersize=2., ls='None')
    # axes[0, 3].set_xlim(-100., 100.)
    # axes[0, 3].set_ylim(0.0, 6.)
    # axes[0, 3].set_xlabel('$\Delta$f$_{QSQ}$ [a.u.]')
    # axes[0, 3].set_ylabel('$\Delta$n$_{e, lin}$ [10$^{19}$m$^{-3}$]')

    # dPrad vs dWdia
    axes[0, 3].plot(
        [(v - D['PradHBCm']['params']['W_dia']['start_val'][i]) / 1.e3
         for i, v in enumerate(D['PradHBCm']['params']['W_dia']['peak_val'])],
        [(v - D['PradHBCm']['start_val'][i]) / 1.e6
        for i, v in enumerate(D['PradHBCm']['peak_val'])],
        label='HBC', c='r', marker='x', markersize=2., ls='None')
    axes[0, 3].plot(
        [(v - D['PradVBC']['params']['W_dia']['start_val'][i]) / 1.e3
         for i, v in enumerate(D['PradVBC']['params']['W_dia']['peak_val'])],
        [(v - D['PradVBC']['start_val'][i]) / 1.e6
        for i, v in enumerate(D['PradVBC']['peak_val'])],
        label='VBC', c='b', marker='x', markersize=2., ls='None')
    # axes[0, 4].set_xlim(., .)
    axes[0, 3].set_ylim(0.0, 2.)
    axes[0, 3].set_xlabel('$\Delta$W$_{dia}$ [kJ]')
    axes[0, 3].set_ylabel('$\Delta$P$_{rad}$ [MW]')

    # dPrad vs dpressure
    axes[1, 3].plot(
        [(D['PradHBCm']['params']['T_e ECE out']['peak_val'][i] - v) /
         (D['PradHBCm']['params']['n_e lint']['start_val'][i] / 1.e19)
         for i, v in enumerate(D['PradHBCm']['params'][
            'T_e ECE out']['start_val'])],
        [(v - D['PradHBCm']['start_val'][i]) / 1.e6
         for i, v in enumerate(D['PradHBCm']['peak_val'])],
        label='HBC', c='r', marker='x', markersize=2., ls='None')
    axes[1, 3].plot(
        [(D['PradVBC']['params']['T_e ECE out']['peak_val'][i] - v) /
         (D['PradVBC']['params']['n_e lint']['start_val'][i] / 1.e19)
         for i, v in enumerate(D['PradVBC']['params'][
            'T_e ECE out']['start_val'])],
        [(v - D['PradVBC']['start_val'][i]) / 1.e6
         for i, v in enumerate(D['PradVBC']['peak_val'])],
        label='VBC', c='r', marker='x', markersize=2., ls='None')

    # axes[1, 3].plot(
    #     [D['PradVBC']['params']['ECRH']['start_val'][i] *
    #      D['PradVBC']['params']['n_e lint']['start_val'][i] / 1.e19
    #      for i, v in enumerate(D['PradVBC']['start_val'])],
    #     [(v - D['PradVBC']['start_val'][i]) / 1.e6
    #      for i, v in enumerate(D['PradVBC']['peak_val'])],
    #     label='ECRH', c='k', marker='x', markersize=2., ls='None')

    axes[1, 3].set_xlim(-1., 1.)
    axes[1, 3].set_ylim(0.0, 2.)
    axes[1, 3].set_xlabel(
        '$\Delta$p$_{proxy}$ ' +
        '[$\Delta$T$_{e}\cdot$n$_{e}$, keV$\cdot$m$^{-3}$]')
    axes[1, 3].set_ylabel('$\Delta$P$_{rad}$ [MW]')

    for axis in axes:
        for ax in axis:
            ax.legend()

    fig_current_save('peaks_plasma_params', fig)
    fig.savefig(
        '../results/PEAKS/peaks_plasma_params_' + label + '.pdf',
        bbox_inches='tight', dpi=169.)
    p.close('all')
    return


def plot_plasma_peak_regress(
        res={'none': None},
        debug=False):
    N, M = np.shape(res['x'])[0], 1
    fig, axes = p.subplots(N, M)
    fig.set_size_inches(M * 5., 3. * N)

    for i, xlabel in enumerate(res['xlabel']):
        data, params = (res['x'][i], res['y'][i], res['z'][i]), \
            (res['a'][i], res['b'][i], res['c'][i])
        a, b, c = str(frac(params[0]).limit_denominator(10)), \
            str(frac(params[1]).limit_denominator(10)), \
            str(frac(params[2]).limit_denominator(10))
        sort = np.array([  # sorted(zip([
            params[0] * data[0]**(params[1]) * data[1]**(params[2]),
            round(params[0], 2) * data[0]**(round(params[1], 2)) *
            data[1]**(round(params[2], 2))])
        #     ))).transpose()

        axes[i].plot(
            params[0] * data[0] ** params[1] * data[1] ** params[2], data[2],
            marker='x', ls='None', c='k', markersize=2.)
        axes[i].plot(
            sort[0], sort[1], linestyle='-.', color='r',
            label='f=' + a + '$\cdot$u$^{' + b + '}\cdot$v$^{' + c + '}$')
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(res['ylabel'][i])
        axes[i].legend()

    fig_current_save('peaks_plasma_params_regress', fig)
    p.show()
    return
