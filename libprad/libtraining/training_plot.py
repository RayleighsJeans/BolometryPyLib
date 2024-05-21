""" ********************************************************************** """

import sys
import numpy as np
import matplotlib.pyplot as p
import json
import requests

import plot_funcs as pf
import mClass

Z = np.zeros
ones = np.ones

M = 10000
stdwrite = sys.stdout.write

""" ********************************************************************** """


def correlation_comb_plot(
        method='weighted_deviation',
        pref='weighted_deviation',
        lab='weighted deviation',
        ylab='weighted deviation',
        loc='../results/COMBINATIONS/',
        cell=[0, 15, 31],
        arr1=Z((M)),
        arr2=Z((M)),
        trace=Z((M)),
        phi=Z((M)),
        freqs=Z((M)),
        beta=1.0,
        alpha=1.0,
        theta=0.9,
        time=np.linspace(0, 10, M),
        id_1=1000,
        id_2=9000):
    """ Combination correlation result
    Args:
        method (str, optional): Method used
        cell (list, optional): channels
        prad (TYPE, optional): prad data vector
        calc (TYPE, optional): calculated prad from cell
        trace (TYPE, optional): evaluation vector
        phi (TYPE, optional): norm vector of best possible
        freqs (TYPE, optional): if fft or so, give frequencies as well
        beta (float, optional): multiply factor
        alpha (float, optional): mean norm value
        theta (float, optional): mean value of sensitivity
        time_plot (TYPE, optional): time vector
        id_1 (int, optional): left index
        id_2 (int, optional): right index
        pref (str, optional): prefix for file name
        lab (str, optional): label
        ylab (str, optional): y ax label
        loc (str, optional): path
    """
    calc_color, prad_color, trace_color, phi_color = \
        'blue', 'red', 'purple', 'k'
    if method in [
            'weighted_deviation', 'mean_deviation',
            'self_correlation']:
        fig, host = p.subplots()
        trace_ax = host.twinx()
    elif 'fft' in method:
        fig = p.figure()
        host = fig.add_subplot(111, label="1")
        trace_ax = fig.add_subplot(
            111, label="2", frame_on=False)
    elif method == 'correlation':
        fig, host = p.subplots()
    lines = []

    prad_p, = host.plot(
        time, 1e-6 * arr1, label='P$_{rad, HBC}$',
        color=prad_color, alpha=.5)
    lines.append(prad_p)

    if not isinstance(cell, int):
        char = str([str(cell) if (len(cell) < 10)
                    else str(cell[0:5]) + '...'
                    for i in range(1)][0])
    else:
        char = str(cell)
    calc_p, = host.plot(
        time, 1e-6 * arr2, label='S',  # = ' + char,
        color=calc_color, alpha=0.5)
    lines.append(calc_p)

    if method != 'correlation':
        value_range = \
            freqs if 'fft' in method else time[id_1:id_2]

        if method in ['mean_deviation', 'fft_solo',
                      'fft_and_convolve', 'fftconvolve_integrated']:
            trace = trace / (10 ** int(np.log10(np.max(trace))))
            if method == 'mean_deviation':
                phi[:] = .0

        trace_p, = trace_ax.plot(
            value_range, trace / beta,
            label=lab, c=trace_color,
            alpha=.75, lw=1.)
        lines.append(trace_p)

        if method not in ['fft_solo', 'fft_and_convolve']:
            phi_p, = trace_ax.plot(
                value_range, phi / beta,
                c=phi_color, label='norm',
                alpha=1., ls='-.', lw=0.75)
            lines.append(phi_p)

    T, A = .6 * time[id_2], 0.12e-6 * np.max([
        np.max(arr1[id_1:id_2]), np.max(arr2[id_1:id_2])])
    if method in ['mean_deviation', 'fft_solo', 'correlation',
                  'fft_and_convolve', 'fftconvolve_integrated']:
        scnum = "{:.2e}".format(theta)
        textstring = str(scnum[:4]) + 'x' + '10$^{' + str(
            int(scnum[-3:])) + '}$'
    else:
        textstring = str(round(theta, 3))

    # build box
    prop = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
    # place a text box in upper left in axes coords
    host.text(T, A, '$\\vartheta_{HBC,S}$ = ' + textstring, bbox=prop)

    if method in ['weighted_deviation', 'mean_deviation',
                  'self_correlation']:
        trace_ax.tick_params(axis='y')
        trace_ax.set_ylabel('quality [a.u.]')

    elif 'fft' in method:
        ylab, xlab = 'spectral density [a.u.]', 'freq. [Hz]'
        trace_ax.xaxis.tick_top()
        trace_ax.yaxis.tick_right()
        trace_ax.xaxis.set_label_position('top')
        trace_ax.yaxis.set_label_position('right')
        if method is not 'coherence_fft':
            trace_ax.set_xlim(-20, 20)
        trace_ax.set_xlabel(xlab)
        trace_ax.set_ylabel(ylab)

    if method in ['fft_solo', 'fft_and_convolve',
                  'fftconvolve_integrated', 'coherence_fft']:
        pos = [.5, 1.45]
    else:
        pos = [.5, 1.3]
    host.legend(
        lines, [l.get_label() for l in lines],
        loc='upper center', bbox_to_anchor=(pos[0], pos[1]),
        ncol=4, fancybox=True, shadow=False,
        handletextpad=0.3, labelspacing=0.5, handlelength=1.)

    # colors on main plot
    host.set_xlabel('time [s]')
    host.set_ylabel('radiation a.u.')
    host.grid(b=True, which='major', linestyle='-.')

    host.set_xlim(time[id_1], time[id_2])
    host.set_ylim(-0.2, np.max([np.max(arr1[id_1:id_2]),
                                np.max(arr2[id_1:id_2])]) * 1.e-6 + 0.2)

    if method not in ['correlation', 'coherence_fft']:
        pf.align_yaxis(
            host, .0, trace_ax, .0,
            np.max(trace / beta))

    if method in ['fft_solo', 'fft_and_convolve',
                  'fftconvolve_integrated', 'coherence_fft']:
        size = [5., 4.2]
    else:
        size = [5., 2.5]
    fig.set_size_inches(size[0], size[1])

    pf.fig_current_save('correlation_comb', fig)
    fig.savefig(loc + pref + '_C' + str(cell).replace(' ', '_').replace(
        ',', '').replace('.', '').replace('__', '_') + '.pdf', dpi=169.0)
    p.close('all')
    return


def correlation_results_plot(
        ylab='weighted deviation',
        lab='weighted_deviation',
        str_sel='3',
        results=Z((1000)),
        loc='../results/',
        pref='COMBINATIONS/'):
    """ correlation spectrum for combinations plot
    Args:
        ylab (str, optional): y axis label
        lab (str, optional): label
        str_sel (str, optional): string of selection length
        results (TYPE, optional): results vector
        loc (str, optional): path
        pref (str, optional): name prefix
    Returns:
        None.
    """
    # full results plot
    fig, host = p.subplots()
    host.plot(results,
              label=lab + ' $\\widetilde{S}_{' + str(str_sel) + '}$',
              ls='-.', alpha=1., c='k')
    host.set_xlabel('combination number')
    host.set_ylabel('quality [100%]')
    host.set_xlim(0.0, np.shape(results)[0])

    host.legend()
    host.grid(b=True, which='major', linestyle='-.')

    fig.set_size_inches(7., 4.)
    pf.fig_current_save('correlation_results', fig)
    fig.savefig(loc + pref + '_spectrum_C' + str_sel + '.png',
                bbox_inches='tight', dpi=169.0)
    p.close('all')
    return


def sensitivity_plot(
        method='weighted_deviation',
        prio={'none': {'none': None}},
        results={'none': None},
        camera='HBCm',
        program='20181010.032',
        combination='3',
        vmecID='1000_1000_1000_1000_+0000_+0000/01/00jh_l/'):
    """ sensitivity plot
    Args:
        method (str, optional): used method
        results (dict, optional): snesitivity analysis results
        xlab (str, optional): x label
        ylab (str, optional): y label
        camera (str, optional): camera
        program (str, optional): XP ID
        combination (str, optional): Combination length
        base (str, optional): Path
    """
    base = '../results/COMBINATIONS/' + program + '/' + method + '/'
    geom = prio['geometry']

    fig, host = p.subplots(1, 1)
    guest = host.twiny()

    host.plot(
        geom['radius']['reff'],
        results['av_sense_channels'], c='red', ls='-.',
        marker='x', label='average sensitivity of channels', alpha=1.)

    host.grid(True, 'major', 'both', ls='--', lw=.5, c='k', alpha=.3)
    host.legend()
    host.set_ylabel('quality [a.u.]')
    host.set_xlabel('r$_{eff, min}$ [m] along LOS')
    host.set_title(program + ' ' + camera +
                   ' combinations:' + combination, y=1.15)

    R = []
    hit_c = [
        ch for ch, v in enumerate(results['av_sense_channels'])
        if ((v != 0.0) & (v is not None) & ~np.isnan(v))]
    for c, ch in enumerate(hit_c):
        R.append(geom['radius']['reff'][ch])

    for ax in [host, guest]:
        ax.set_xlim(np.min(R) - 0.02, np.max(R) + 0.02)

    def channel_reff(ch):
        return (geom['radius']['reff'][ch])

    tick_locs = host.get_xticks()
    N = int(np.ceil(np.shape(hit_c)[0] / np.shape(tick_locs)[0]) + 1)

    guest.set_xticks([channel_reff(ch) for ch in hit_c[::N]])
    guest.set_xticklabels([str(ch) for ch in hit_c[::N]])
    guest.set_xlabel('channel #')

    m_R = requests.get(
        'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/' +
        vmecID + 'minorradius.json').json()['minorRadius']
    for f in [-1., 1.]:
        guest.axvline(f * m_R, lw=1.0, c='k', alpha=0.5, ls='-.')
    host.text(-1. * m_R + 0.02, 0.65,
              'r$_{LCFS}$', verticalalignment='center', color='k')

    fig.set_size_inches(7., 4.)
    fig.savefig(base + combination + '_' + camera +
                '/spectrum_analysis_' + method + '.png',
                bbox_inches='tight', dpi=169.0)
    pf.fig_current_save('spectrum_analysis', fig)
    p.close('all')

    return


def best_chans_X(
        programs=['201810101.032', '20181207.024'],
        results=[[{'none': None}], [{'none': None}]],
        method='weighted_deviation',
        data=[{'values': None, 'dimensions': None}, {'none': None}],
        fsdata_object={'none': None},
        ez_losight={'none': None},
        camera_geometry={'none': None},
        tag='best_channels',
        C=3):
    """ plotting best channels with lofs in VMEC profile
    Args:
        programs (list, optional): list of XP IDs
        results (list, optional): list of result dicts with fits etc
        method (str, optional): what eval method was/to use
        data (list, optional): list of data jsons
        fsdata_object (dict, optional): fluxsurfaces from VMEC webservice
        ez_losight (dict, optional): easy lines of sight data
        camera_geometry (dict, optional): camera geomtry data
        tag (str, optional): best/worst chans/combination
        C (int, optional): combination length
    Returns:
        None.
    """
    lines = None
    m = 17
    M = len(programs)
    fig, axes = p.subplots(2, M)

    for k, P in enumerate(programs):
        host = axes[0, k]
        guest = host.twinx()
        lp = []

        host.grid(b=True, which='major', linestyle='--')
        if k == 0:
            host.set_ylabel('power [MW]')

        for i, ch in enumerate(results[(k * m) + C][tag]):
            pp, = guest.plot(
                [v * 1e3 for v in data[(k * 2) + 1]['values'][ch]], ':',
                label=str(ch), alpha=1.)
            lp.append(pp)

            lines = \
                np.c_[np.array(data[(k * 2) + 1]['values'][ch])] \
                if i == 0 else \
                np.c_[lines, np.array(data[(k * 2) + 1]['values'][ch])]

        pradp, = host.plot(
            [v * 1e-6 for v in data[(k * 2) + 0]['values']],
            label='P$_{rad, HBCm}$', alpha=1.)
        fit = np.sum(np.array(
            results[(k * m) + C]['fit_results'][tag]) *
            lines, axis=1)
        fitp, = host.plot(fit * 1e-6, label='fit', alpha=1.5)

        lp.append(pradp)
        lp.append(fitp)
        guest.grid(b=True, which='major', linestyle='-.')
        if k == len(programs) - 1:
            guest.set_ylabel('voltage [mV]')

        guest.yaxis.tick_right()
        host.set_xlabel('samples')
        guest.yaxis.set_label_position('right')
        host.legend(lp, [l.get_label() for l in lp])
        host.set_xlim(
            results[(k * m) + C]['startID'], results[(k * m) + C]['stopID'])

        # Lines of sight
        camera = results[(k * m) + C]['camera']
        if camera == 'HBC_VBC':
            cams = ['HBC', 'VBCr', 'VBCl']
        elif camera == 'HBC':
            cams = ['HBC']
        elif camera == 'VBC':
            cams = ['VBCr', 'VBCl']

        HBC_selection = [[], []]
        VBCr_selection = [[], []]
        VBCl_selection = [[], []]

        cell = results[(k * m) + C][tag]
        if camera == 'HBC':
            HBC_selection = [cell, cell]

        elif 'VBC' in camera:
            for i, ch in enumerate(cell):
                if 32 < ch <= 64:
                    VBCl_selection[0].append(ch - 40)
                    VBCl_selection[1].append(ch)

                elif ch > 64:
                    VBCr_selection[0].append(ch - 64)
                    VBCr_selection[1].append(ch)

                elif ch <= 31:
                    HBC_selection[0].append(ch)
                    HBC_selection[1].append(ch)

        if P in ['20181009.046', '20181009.007', '20181009.041']:
            FS_label = 'KJM_beta202'
        else:
            FS_label = 'EIM_beta000'

        # loading fluxsurface data needed for later
        loc = '../results/INVERSION/'
        f = loc + 'FS_LCFS_MAGAX/' + FS_label + '_fs_data.json'
        with open(f, 'r') as file:
            fsdata_object = json.load(file)
        file.close()

        for ax in [host, guest]:
            pf.autoscale_y(ax)

        host.set_title('#' + P + ' ' + method + ' fit: best cell')

    pf.fig_current_save('best_chans_crossXperiment', fig)
    p.close('all')

    return


def av_sense_X(
        programs=['201810101.032', '20181207.024'],
        results=[[{'none': None}], [{'none': None}]],
        method='weighted_deviation'):
    """ average sensibility cross experiment
    Args:
        programs (list, optional): what to be compared
        results (list, optional): given data objects
        method (str, optional): comparing method
    Returns:
        None.
    """
    fig = p.figure()
    host = fig.add_subplot(111, label="1")

    m, j = 17, 3
    for k, P in enumerate(programs):
        foo = results[(k * m) + j]
        cam, comb = foo['camera'], foo['combination']
        c, = host.plot(foo['av_sense_channels'], '-.', marker='^',
                       alpha=1., label=P, markersize=4.)

    host.set_title('comparison ' + cam + ' ' + comb)
    host.grid(b=True, which='major', linestyle='--')
    host.set_xlabel('# of channels')
    host.set_ylabel(method.replace('_', ' ') + ' [a.u.]')
    host.legend()

    pf.fig_current_save('av_sense_crossXperiment', fig)
    p.close('all')

    return


def p_lin_fit(
        target=Z((M)),
        fit=Z((M)),
        cell=[0, 15, 31],
        lines=Z((128, M)),
        parameters=Z((3)),
        path='../results/COMBINATIONS/20180920.042/',
        method='weighted_deviation',
        camera='HBC',
        tag='best_cell',
        combination='3'):
    """ linear fit function plot
    Args:
        target (0, ndayrray, optional): Data base that has been fit to
        fit (1, ndarray, optional): Fit result
        cell (2, list, optional): Channels selected
        lines (3, ndarray, optional): Channel voltages
        parameters (TYPE, optional): Fit result factors
        fsdata_object (4, dict, optional): Fluxsurface data
        ez_losight (5, dict, optional): Lines of sight data
        camera_geometry (6, ict, optional): Camera geometry data
        camera (7, str, optional): What camera to use
        tag (8, str, optional): Which type of cell
        path (9, str, optional): Where to write to
        method (10, str, optional): Method of eval used
        combination (11, str, optional): How many channels selected
    Returns:
        None.
    """
    fig = p.figure()
    host = fig.add_axes([0.1, 1.0, 0.8, 0.4], xticklabels=[])
    guest = fig.add_axes([0.1, 0.6, 0.8, 0.4])

    host.plot(target * 1e-6, '-.', label='target', c='b', alpha=1.)
    host.plot(fit * 1e-6, label='fit', c='r', alpha=1.)

    host.grid(b=True, which='major', linestyle='-.')
    host.set_ylabel('power [MW]')
    host.legend()

    ex = min([np.floor(np.log10(np.abs(p))).astype(int) for p in parameters])
    mx = max([max(lines[ch]) for i, ch in enumerate(cell)]) * 1e3
    cs = ['k', 'gray', 'r', 'b', 'g', 'purple', 'brown', 'orange', 'cyan']
    for i, ch in enumerate(cell):
        guest.plot(
            [v * 1e3 for v in lines[ch]],
            label=str(ch), c=cs[i], alpha=1.)
        guest.text(
            2e3, mx - i * (mx / 9),
            'p$_{' + str(i) + '}$=' + str(
                round(parameters[i] / (10. ** ex), 3)) +
            ' MW/mV', verticalalignment='center', color=cs[i])

    guest.grid(b=True, which='major', linestyle='-.')
    guest.set_ylabel('voltage [mV]')
    guest.legend()

    guest.xaxis.tick_bottom()
    guest.yaxis.tick_left()
    guest.set_xlabel('samples', color='k')
    guest.xaxis.set_label_position('bottom')
    guest.yaxis.set_label_position('left')

    program = path.replace(
        '../results/COMBINATIONS/', '').replace(method, '').replace(
        combination + '_' + camera, '').replace('/', '')
    host.set_title('#' + program + ' ' + method + ' fit:' + tag)
    fig.set_size_inches(5., 8.)

    fig.savefig(path + tag + '_and_fit.png',
                bbox_inches='tight', dpi=169.0)
    pf.fig_current_save(tag + '_and_fit', fig)

    p.close('all')
    return
