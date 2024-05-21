""" **************************************************************************
    so HEADER """

import sys
import os
import numpy as np
import matplotlib.pyplot as p
from matplotlib.pyplot import cm
from itertools import cycle
import requests as req
import random as rand

import plot_funcs as plot_funcs
import webapi_access as api
import warnings
import dat_lists as dat_lists

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")

Z = np.zeros

""" eo header
************************************************************************** """


def core_v_sol_prad(
        time=Z((10000)),
        core_SOL={},
        ecrh_off=10.0,
        program='20181010.032',
        program_info={'none': None},
        indent_level='\t'):
    """ core v SOL prad plot
    Keyword Arguments:
        time {[type]} -- time vector (default: {Z((10000))})
        core_SOL {dict} -- core and SOL prads (default: {{}})
        ecrh_off {float} -- ecrh off time (default: {10.0})
        program {str} -- XP ID (default: {'20181010.032'})
        program_info {dict} -- XP info (default: {{'none': None}})
        indent_level {str} -- printing indentation (default: {'\t'})
    """
    print(indent_level + '>> core v SOL plot ...')
    start = str(program_info['programs'][0]['from'])
    stop = str(program_info['programs'][0]['upto'])

    fig = p.figure()
    cvs = fig.add_subplot(111)

    cvs.grid(b=True, which='major', linestyle='-.')
    cvs.set_ylabel("power [MW]")
    cvs.set_xlabel("time [s]")
    cvs.set_title('#' + program)

    colorsc = cycle(cm.plasma(np.linspace(0, 1, 4)))
    for k, cam in enumerate(['HBCm', 'VBC']):
        try:
            cvs.plot(time, core_SOL[cam]['P_rad_core'] * 1e-6, c=next(colorsc),
                     label="P$_{core} $" + cam, alpha=0.75)
            cvs.plot(time, core_SOL[cam]['P_rad_sol'] * 1e-6, c=next(colorsc),
                     label="P$_{SOL}$ " + cam, alpha=0.75)

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
                  '\n' + indent_level + '\t\\\  failed core v sol plot ' + cam)

    cvs.legend()
    cvs.set_xlim(-.5, ecrh_off + 0.3)
    plot_funcs.autoscale_y(cvs)
    cvs.legend()

    p.tight_layout()
    fig.savefig('../results/PRAD/' + program[0:8] + '/prad_corevsol_' +
                program + '.png', bbox_inches='tight', dpi=169.0)
    plot_funcs.fig_current_save('current_prad_corevsol', fig)
    p.close('all')
    return


def core_sol_ratios(
        time=Z((10000)),
        ratios={},
        ecrh_off=10.0,
        program='20181010.032',
        program_info={'none': None},
        indent_level='\t'):
    """ core v SOL ratios of radiation
    Keyword Arguments:
        time {[type]} -- time vector (default: {Z((10000))})
        core_SOL {dict} -- core and SOL prads (default: {{}})
        ecrh_off {float} -- ecrh off time (default: {10.0})
        program {str} -- XP ID (default: {'20181010.032'})
        program_info {dict} -- XP info (default: {{'none': None}})
        indent_level {str} -- printing indentation (default: {'\t'})
    """
    print(indent_level + '>> core SOL ratios plot ...')
    start = str(program_info['programs'][0]['from'])
    stop = str(program_info['programs'][0]['upto'])

    fig = p.figure()
    csr = fig.add_subplot(111)

    csr.grid(b=True, which='major', linestyle='-.')
    csr.set_ylabel("ratio / %")
    csr.set_xlabel("time / s")
    csr.set_title('#' + program)

    colorsc = cycle(cm.plasma(np.linspace(0, 1, 4)))
    for k, cam in enumerate(['HBCm', 'VBC']):
        try:
            csr.plot(time, ratios[cam]['core'], c=next(colorsc),
                     label="R$_{core} $" + cam, alpha=0.6)
            csr.plot(time, ratios[cam]['sol'], c=next(colorsc),
                     label="R$_{SOL}$ " + cam, alpha=0.6)

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
                  '\n' + indent_level +
                  '\t\\\  failed core sol ratio plot ' + cam)

    csr.legend()
    csr.set_xlim(-.5, ecrh_off + 0.3)
    # plot_funcs.autoscale_y(csr)
    csr.legend()

    p.tight_layout()
    fig.savefig('../results/PRAD/' + program[0:8] + '/prad_coresol_ratio_' +
                program + '.png', bbox_inches='tight', dpi=169.0)
    plot_funcs.fig_current_save('current_prad_coresol_ratio', fig)
    p.close('all')
    return


def prad_output(
        prio={'none': {'none': None}},
        dat={'none': {'none': None}},
        power={'none': {'none': None}},
        program='20181010.032',
        program_info={'none': None},
        indent_level='\t',
        solo=True):
    """ Plots all the important power and reference quantities together.
    Args:
        priority_object (0, list): List of calculated power variables
        data_object (1, list): Archive downloaded data prior
        date (2, str): Date selected
        shotno (3, int): XP ID selected
        program (4, str): Date and XP ID concatenated
        program_info (5, dict): HTTP respone from logbook for program
        indent_level (6, str): Indentation level
    Returns:
        None
    Notes:
        None
    """
    print(indent_level + '>> P_rad plot ...')
    start = str(program_info['programs'][0]['from'])
    stop = str(program_info['programs'][0]['upto'])

    # make figure and size and content and labels and axis
    time = power['time']

    if not solo:
        fig, (vhbc, feedback) = p.subplots(nrows=2)
        params = feedback.twinx()
        fig.set_size_inches(7., 10.)
    elif solo:
        fig = p.figure()
        fig.set_size_inches(7., 5.)
        vhbc = fig.add_subplot(111)

    vhbc.grid(b=True, which='major', linestyle='-.')
    vhbc.set_ylabel("power [MW]")
    vhbc.set_xlabel("time [s]")
    vhbc.set_title('#' + program)

    if not solo:
        feedback.grid(b=True, which='major', linestyle='-.')
        feedback.set_ylabel("power [MW]")
        params.set_ylabel("feedback parameter [a.u.]")
        feedback.set_xlabel("time [s]")

    try:
        vhbc.plot(
            time,
            [x / 1e6 for x in power['P_rad_hbc']], "r",
            label="P$_{rad}$ HBC")
        vhbc.plot(
            time,
            [x / 1e6 for x in power['P_rad_vbc']], "b",
            label="P$_{rad}$ VBC")
        vhbc.legend()

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  Failed at python prad')

        try:
            archiveHBC = api.download_single(
                api.download_link(name='Bolo_HBCmPrad'),
                program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)
            archiveVBC = api.download_single(
                api.download_link(name='Bolo_VBCPrad'),
                program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)

            archive_time = \
                [(x - prio['t0']) / 1e9 for x in archiveHBC["dimensions"]]
            archive_hbcprad = \
                [x for x in archiveHBC['values'][0]]
            archive_vbcprad = \
                [x for x in archiveVBC['values'][0]]

            vhbc.plot(
                archive_time, archive_hbcprad, '--',
                c='orange', label='archive P$_{rad}$ HBC')
            vhbc.plot(
                archive_time, archive_vbcprad, ':',
                c='cyan', label='archive P$_{rad}$ VBC')
            vhbc.legend()

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\ ', exc_type,
                  fname, exc_tb.tb_lineno, '\n' + indent_level +
                  '\t\\\  Failed at archive Vi data')

    if not solo:
        try:  # plot qsq channels
            qsq1 = api.download_single(
                api.download_link(name='EKS1 Bolo1'),
                program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)
            qsq2 = api.download_single(
                api.download_link(name='EKS1 Bolo2'),
                program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)
            qsqparams = api.download_single(
                api.download_link(name='QSQ Params'),
                program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)

            a, = feedback.plot(
                [(x - prio['t0']) / 1e9 for x in qsq1["dimensions"]],
                qsq1["values"], color="orange",
                label="He-valve (QSQ) ch#1")
            b, = feedback.plot(
                [(x - prio['t0']) / 1e9 for x in qsq2["dimensions"]],
                qsq2["values"], color='cyan',
                label="He-valve (QSQ) ch#2")
            c, = params.plot(
                [(x - prio['t0']) / 1e9 for x in qsqparams["dimensions"]],
                [v for v in qsqparams["values"]], color='grey',
                label="QSQ PID Param.")

            feedback.legend([a, b, c], [l.get_label() for l in [a, b, c]])
            feedback.set_title('# feedback signals')

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\ ', exc_type, fname,
                  exc_tb.tb_lineno, '\n' + indent_level +
                  '\t\\\  Failed at feedback plot')

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

            E = (rand.randrange(1000, 2000) / 1.e4)
            f1 = np.max(fb1['values'][0]) / (np.max(power['P_rad_hbc']) / 1.e6)
            f2 = np.max(fb2['values'][0]) / (np.max(power['P_rad_hbc']) / 1.e6) 

            f1 += E * f1
            f2 += E * f2

            vhbc.plot(
                [(x - prio['t0']) / 1e9 for x in fb1["dimensions"]],
                [x * f1 for x in fb1["values"][0]], color="orange",
                label="P$_{rad, pred}$ multi ch. $\\times$ f")
            vhbc.plot(
                [(x - prio['t0']) / 1e9 for x in fb2["dimensions"]],
                [x * f2 for x in fb2["values"][0]], color='cyan',
                label="single ch. $\\times$ f")

            vhbc.legend()
            vhbc.set_title('# feedback signals')

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(indent_level + '\t\\\ ', exc_type, fname,
                  exc_tb.tb_lineno, '\n' + indent_level +
                  '\t\\\  Failed at feedback plot')

    vhbc.set_xlim(-.5, prio['ecrh_off'] + 0.3)
    vhbc.legend()

    if not solo:
        feedback.set_xlim(-.5, prio['ecrh_off'] + 0.3)
        feedback.legend()

    p.tight_layout()
    fig.savefig(
        '../results/PRAD/' + program[0:8] + '/prad_handvbc_' +
        program + '.png', bbox_inches='tight', dpi=169.)
    plot_funcs.fig_current_save('current_prad_handvbc', fig)
    p.close('all')
    return


def reff_plot(
        prio={'none': {'none': None}},
        power={'none': {'none': None}},
        program='20181010.032',
        program_info={'none': None},
        lvl=8,
        vmecID='1000_1000_1000_1000_+0000_+0000/01/00jh_l/',
        indent_level='\t'):
    """ Plot channel power over time and effective plasma radius [0, ~60cm]
        as a surface plot without outlines.
    Args:
        prio: Precalculated and constanc variables needed for calc.
        power: List of calculated power variables.
        program: Date and XP ID concatenated.
        program_info: HTTP respone from logbook for program.
        lvl (int, optional): levels of surface plot
        indent_level: Indentation level.
    Returns:
        None.
    Notes:
        None.
    """
    print(indent_level + '>> r_eff plots ...')

    m_R = req.get(
        'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/' +
        vmecID + 'minorradius.json').json()['minorRadius']

    try:
        location = r'../results/REFFS/' + program[0:8]
        if not os.path.exists(location):
            os.makedirs(location)

        time = power['time']
        geom = prio['geometry']
        reff = dat_lists.geom_dat_to_json()
        broken = reff['channels']['droplist']

        fig, (surf_vbc, surf_hbc) = p.subplots(nrows=2)

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  reff contourf setup failed')
        return

    try:
        reff_vbc = Z((len(geom['channels']['eChannels']['VBC'])))
        power_vbc = Z((len(geom['channels']['eChannels']['VBC']), len(time)))
        for i, ch in enumerate(geom['channels']['eChannels']['VBC']):
            if ch not in broken:
                power_vbc[i] = power['power'][ch] * 1e3
                reff_vbc[i] = geom['radius']['reff'][ch]

        reff_hbc = Z((len(geom['channels']['eChannels']['HBCm'])))
        power_hbc = Z((len(geom['channels']['eChannels']['HBCm']), len(time)))
        for i, ch in enumerate(geom['channels']['eChannels']['HBCm']):
            if ch not in broken:
                power_hbc[i] = power['power'][ch] * 1e3
                reff_hbc[i] = geom['radius']['reff'][ch]

    except Exception:
        print(indent_level + '\t\\\  reff data failed')
        return

    try:
        vX, vY = np.meshgrid(reff_vbc, time)
        cs1 = surf_vbc.contourf(vX, vY, np.transpose(power_vbc),
                                lvl, cmap='viridis')
        cbar1 = fig.colorbar(cs1, ax=surf_vbc)
        surf_vbc.set_title("vertical bolometer, #" + program)

        for f in [-1., 1.]:
            surf_vbc.axvline(
                f * m_R, lw=1.0, c='white', alpha=0.75, ls='-.')

        cbar1.ax.set_ylabel("power [mW]")
        surf_vbc.set_xlabel('r$_{eff, min}$ [m]')
        surf_vbc.set_ylabel("time [s]")

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  reff VBC failed')

    try:
        hX, hY = np.meshgrid(time, reff_hbc)
        cs2 = surf_hbc.contourf(hX, hY, power_hbc, lvl, cmap='viridis')
        cbar2 = fig.colorbar(cs2, ax=surf_hbc)
        surf_hbc.set_title("horizontal bolometer")

        for f in [-1., 1.]:
            surf_hbc.axhline(
                f * m_R, lw=1.0, c='white', alpha=0.75, ls='-.')

        cbar2.ax.set_ylabel("power [mW]")
        surf_hbc.set_ylabel('r$_{eff, min}$ [m]')
        surf_hbc.set_xlabel("time [s]")

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  reff HBC failed')

    try:
        surf_vbc.set_ylim(-0.25, prio['ecrh_off'])
        surf_hbc.set_xlim(-0.25, prio['ecrh_off'])

        fig.set_size_inches(10., 9.)
        p.tight_layout()

        fig.savefig('../results/REFFS/' + program[0:8] + '/reffs' +
                    program + '.png', bbox_inches='tight', dpi=169.0)
        plot_funcs.fig_current_save('reffs', fig)
        p.close('all')

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  reff save failed')

    return


def stddiv_output(
        prio={'none': {'none': None}},
        dat={'none': {'none': None}},
        power={'none': {'none': None}},
        program='20181010.032',
        program_info={'none': None},
        indent_level='\t'):
    """ Plots the standard devitation of each channel with color coded\
        backgrounds for cams.
    Args:
        priority_object: Precalculated and constanc variables needed for calc.
        radpower_object: List of calculated power variables.
        data_object: Archive downloaded data prior.
        date: Date selected.
        shotno: XP ID selected.
        program: Date and XP ID concatenated.
        program_info: HTTP respone from logbook for program.
        indent_level: Indentation level.
    Returns:
        None.
    Notes:
        None.
    """
    print(indent_level + '>> power stddiv plot ...')
    fig = p.figure()
    vhbc = fig.add_subplot(111)
    vhbc.grid(b=True, which='major', linestyle='-.')

    try:
        dt = prio['dt']
        # if not, mkdir and cd there
        location = r"../results/TMP/STDDIVS/" + program[0:8] + '/'
        if not os.path.exists(location):
            os.makedirs(location)

        # stddiv data
        vhbc.plot(power['stddiv'] * 1e6, "k", ls='-.', label="std. div.")
        vhbc.legend()
        vhbc.set_ylabel("voltage [µV]")
        vhbc.set_xlabel("channel #")
        vhbc.set_title(program + ' standard deviation')

        fig.savefig(
            '../results/TMP/STDDIVS/' + program[0:8] + '/' + program +
            '.stddiv.png', bbox_inches='tight', dpi=169.0)
        plot_funcs.fig_current_save('stddivs', fig)

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  stddiv plot failed')
        return

    try:
        eChans = prio['geometry']['channels']['eChannels']
        L = len(eChans.keys())
        C = cm.jet(np.linspace(0, 1, L))
        max_stddivs = max(power['stddiv']) * 1.e6

        for i, tag in enumerate(eChans.keys()):
            if isinstance(eChans[tag][0], str):
                continue

            vhbc.axvspan(eChans[tag][0] - 0.5, eChans[tag][-1] + 0.5,
                         facecolor=C[i], alpha=0.44,
                         label=tag + ' [' + str(eChans[tag][0]) + ':' +
                         str(eChans[tag][-1]) + ']')

        vhbc.text(2., max_stddivs * 0.8,
                  'sampling rate :\n' +
                  str(round(dt * 1e3, 3)) + ' ms', color='k',
                  size=12, rotation=0)
        vhbc.set_xlim(0.0, 128.)
        vhbc.legend()

        fig.savefig(
            '../results/TMP/STDDIVS/' + program[0:8] + '/' + program +
            '.stddiv.png', bbox_inches='tight', dpi=169.0)
        plot_funcs.fig_current_save('stddivs', fig)
        p.close('all')

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  stddiv colors plot failed')

    return


def linoffs_output(
        prio={'none': {'none': None}},
        dat={'none': {'none': None}},
        power={'none': {'none': None}},
        program='20181010.032',
        program_info={'none': None},
        indent_level='\t'):
    """ Plot linear offset drift fix coefficients, constant and slope.
    Args:
        priority_object: Precalculated and constanc variables needed for calc.
        radpower_object: List of calculated power variables.
        data_object: Archive downloaded data prior.
        date: Date selected.
        shotno: XP ID selected.
        program: Date and XP ID concatenated.
        program_info: HTTP respone from logbook for program.
        indent_level: Indentation level.
    Returns:
        None.
    Notes:
        None.
    """
    print(indent_level + '>> linear_offset plots ...')
    lines = []

    try:
        # if measurement directory for local saving exists, dont do anything
        # if not, mkdir and cd there
        location = r"../results/TMP/LINOFFS/" + program[0:8] + '/'
        if not os.path.exists(location):
            os.makedirs(location)

        # make figure and size and content and labels and axis
        fig, host = p.subplots()
        ch_corrections = host.twinx()
        slopes, = host.plot(
            [x * 1e6 for x in power['a']],
            "k", marker='*', linestyle='-.', label="slope")
        corrections, = ch_corrections.plot(
            [y * 1e6 for y in power['b']],
            "green", marker='^', linestyle='-.', label="corr.")
        lines = [slopes, corrections]

        host.set_xlim(.0, 128.)
        host.set_ylabel("slope [µV/s]")
        host.set_xlabel("channel #")
        host.set_title(program + ' linear corrections coefficients')

        ch_corrections.set_ylabel("voltage [µV]")
        ch_corrections.set_frame_on(True)
        ch_corrections.patch.set_visible(False)
        ch_corrections.spines['right'].set_visible(True)
        ch_corrections.spines['right'].set_edgecolor(corrections.get_color())
        ch_corrections.yaxis.label.set_color(corrections.get_color())
        ch_corrections.tick_params(axis='y', colors=corrections.get_color())
        host.legend(lines, [l.get_label() for l in lines],
                    frameon=True, loc="upper right")

        host.grid(b=True, which='major', linestyle='-.')
        fig.savefig(
            '../results/TMP/LINOFFS/' + program[0:8] + '/' + program + '.png',
            bbox_inches='tight', dpi=169.0)
        plot_funcs.fig_current_save('linoffs', fig)

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  linoffs plot failed')
        return

    try:
        eChans = prio['geometry']['channels']['eChannels']
        L = len(eChans.keys())
        C = cm.jet(np.linspace(0, 1, L))

        for i, tag in enumerate(eChans.keys()):
            if isinstance(eChans[tag][0], str):
                continue

            area = host.axvspan(eChans[tag][0] - 0.5, eChans[tag][-1] + 0.5,
                                facecolor=C[i], alpha=0.44,
                                label=tag + ' [' + str(eChans[tag][0]) + ':' +
                                str(eChans[tag][-1]) + ']')
            lines.append(area)
        host.set_xlim(0.0, 128.)
        host.legend()

        host.legend(lines, [l.get_label() for l in lines],
                    frameon=True, loc="upper right")

        fig.savefig('../results/TMP/LINOFFS/' + program[0:8] + '/' +
                    program + '.png', bbox_inches='tight', dpi=169.0)
        plot_funcs.fig_current_save('linoffs', fig)
        p.close('all')

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  linoffs colors plot failed')

    return
