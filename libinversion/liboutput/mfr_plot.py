""" **************************************************************************
    so header """

import numpy as np
import requests
import json

import matplotlib as mpl
import matplotlib.pyplot as p
from matplotlib.pyplot import cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import matplotlib.colorbar as cbar

import mClass
import plot_funcs as pf
import factors_geometry_plots as fpp

import dat_lists
import mfr2D_matrix_gridtransform as mfr_transf

# poincare plots
import poincare
import poincare_plots

# database
import database_inv as database
database = database.import_database()
VMEC_URI = 'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/'

Z = np.zeros
ones = np.ones

""" end of header
************************************************************************** """


def base_mfr2D_plot(
        r_grid,
        z_grid,
        profile,
        debug=False):
    fig, ax = p.subplots(1, 1)
    patches = []
    for i in range(np.shape(profile)[0]):
        for j in range(np.shape(profile)[1]):
            poly = np.array([r_grid[i, j], z_grid[i, j]]).transpose()
            polygon = Polygon(poly, aa=False)
            patches.append(polygon)

    plot = PatchCollection(patches, cmap=mpl.cm.viridis)
    colors = profile.reshape(-1)
    plot.set_array(colors)
    ax.add_collection(plot)

    cb = fig.colorbar(plot, ax=ax)
    cb.set_label('profile [a.u.]')
    p.show
    return


def tomogram_plot_wrapper(
        data=np.zeros((30, 100, 1)),
        x=np.zeros((30, 100, 4)),
        y=np.zeros((30, 100, 4)),
        radial_tomo=np.zeros((2, 30)),
        chordal_tomo=np.zeros((2, 128)),
        chordal_xp=np.zeros((2, 128)),
        xp_error=np.zeros((128)),
        radial_error=0.0,
        absolute_error=0.0,
        kani_profile=np.zeros((2, 30)),
        chi=np.zeros((128)),
        peaks_r=None,
        peaks_id=None,
        half_widths=None,
        minor_radius=0.5139,
        total_power=0.0,
        core_power=0.0,
        prad_tomo=0.0,
        prad_xp=0.0,
        chi2=0.0,
        nr=100,
        nt=30,
        nigs_real=1,
        label='pos_mesh_x5.75_y0.0_mx1.0e+06',
        strgrid='sN8_30x20x150_1.3',
        magconf='EIM_000',
        add_camera=False,
        cartesian=False,
        debug=False):
    camera_info = dat_lists.geom_dat_to_json()
    # load islands and divertor/vessel
    name = 'phi108.0_res10_conf_lin'
    islands = poincare.store_read(name=name, arg='islands1_')
    divertors = poincare.define_divWall(phi=108.)

    mfr2D_plot(
        data=data, x=x, y=y, nr=nr, nt=nt,
        nigs_real=nigs_real, label=label,
        islands=islands, divertors=divertors,
        debug=debug, VMID=magconf, strgrid=strgrid)
    if debug:
        mfr2D_grid_plot(
            x=x, y=y, nr=nr, nt=nt, label=label,
            cartesian=cartesian)

    mfr2D_chordal_plot(
        nr=nr, nt=nt, label=label, chordal_tomo=chordal_tomo,
        chordal_xp=chordal_xp, kani_profile=kani_profile,
        add_camera=add_camera, chi=chi, chi2=chi2,
        camera_info=camera_info, strgrid=strgrid,
        prad_tomo=prad_tomo, prad_xp=prad_xp,
        minor_radius=minor_radius, xp_error=xp_error, debug=debug)

    mfr2D_radial_plot(
        radial=radial_tomo, kani=kani_profile,
        peaks_r=peaks_r, peaks_id=peaks_id, half_widths=half_widths,
        total=total_power, core=core_power, minor_radius=minor_radius,
        radial_error=radial_error, absolute_error=absolute_error,
        nr=nr, nt=nt, label=label, strgrid=strgrid)

    # mfr2D_poloidal(
    #     tomogram=data, fs_radius=radial_tomo[0, :],
    #     label=label, minor_radius=minor_radius,
    #     strgrid=strgrid, debug=debug)
    return


def mfr2D_plot(
        data=np.zeros((30, 100)),
        x=np.zeros((30, 100, 4)),
        y=np.zeros((30, 100, 4)),
        islands=np.zeros((4, 15, 100)),
        divertors=np.shape((4, 525, 2)),
        nr=100,
        nt=30,
        nigs_real=1,
        label='pos_mesh_x5.75_y0.0_mx1.0e+06',
        base='../results/INVERSION/FS/',
        strgrid='sN8_30x20x150_1.3',
        VMID='EIM_000',
        edge_pattern='LCFS',
        debug=False):
    patches = []
    fig, ax = p.subplots(1, 1)
    for i in range(nr):
        for j in range(nt):
            poly = np.array([
                x[i, j], y[i, j]]).transpose()
            polygon = Polygon(poly, aa=False)
            patches.append(polygon)

    plot = PatchCollection(patches, cmap=mpl.cm.viridis)
    colors = data.reshape(-1) / 1.e6
    plot.set_array(colors)
    ax.add_collection(plot)

    cb = fig.colorbar(plot, ax=ax, shrink=0.85)
    cb.set_label('brightness [MW/m$^{3}$]')

    if islands is None:  # LCFS plot
        VMEC_ID = database['values']['magnetic_configurations'][VMID]['URI']
        URI = VMEC_URI + VMEC_ID + '/lcfs.json?phi=108.'
        LCFS = requests.get(URI).json()
        ax.plot(
            LCFS['lcfs'][0]['x1'], LCFS['lcfs'][0]['x3'],
            c='lightgrey', lw=1.5, alpha=0.5, label='LCFS ' + VMID)
    else:  # poincare plots
        poincare_plots.plot_div(axis=ax, points=divertors)
        poincare_plots.PlotPoincare(axis=ax, surfaces=islands)

    ax.axis('off')
    fig.set_size_inches(4.5, 2.5)
    pf.fig_current_save('tomo2D', fig)
    f = '../results/INVERSION/MFR/' + strgrid
    fig.savefig(
        f + '/' + label + '.png',
        bbox_inches='tight', dpi=169.0)
    p.close('all')
    return


def mfr2D_grid_plot(
        nr=30,
        nt=100,
        x=np.zeros((300, 100, 4)),
        y=np.zeros((10, 100, 4)),
        label='pos_mesh_x5.75_y0.0_mx1.0e+06',
        vmec_label='EIM_beta000',
        base='../results/INVERSION/FS/',
        file_fs='EIM_beta000_slanted_HBCmfs_data.json',
        cartesian=False):
    fig, ax = p.subplots(1, 1)
    for i in range(nr):
        for j in range(nt):
            ax.plot(
                x[i, j], y[i, j], c='k',  # colors[i * nt + j],
                lw=0.75, alpha=0.75,
                marker='o', markersize=1.)

    with open(base + file_fs, 'r') as infile:
        indict = json.load(infile)
    infile.close()
    fs_data = mClass.dict_transf(
        indict, list_bool=False)
    FS = fs_data['values']['fs'][-1]
    ax.plot(
        FS[-1][3], FS[-1][2],
        lw=1.5, alpha=0.75, color='lightgrey',
        label='LCFS')

    ax.legend()
    ax.set_title(label.replace('_', ' ')[
        :int(len(label) / 2)] + '\n' + label.replace('_', ' ')[
            int(len(label) / 2):])

    pf.fig_current_save('mfr_grid', fig)
    f = '../results/INVERSION/MFR/' + str(nr) + 'x' + str(nt)
    if cartesian:
        f += '_cartesian'

    fig.savefig(
        f + '/' + label + '_grid.png',
        bbox_inches='tight', dpi=169.0)
    p.close('all')
    return


def mfr2D_chordal_plot(
        nr=30,
        nt=100,
        chordal_tomo=np.zeros((2, 128)),
        chordal_xp=np.zeros((2, 128)),
        xp_error=np.zeros((128)),
        kani_profile=np.zeros((2, 30)),
        minor_radius=0.5139,
        chi=np.zeros((128)),
        chi2=0.0,
        prad_tomo=0.0,
        prad_xp=0.0,
        camera_info={'none': None},
        label='pos_mesh_x5.75_y0.0_mx1.0e+06',
        strgrid='sN8_30x20x150_1.3',
        add_camera=False,
        debug=False):
    channels = mfr_transf.mfr_channel_list()
    fig, ax = p.subplots(1, 1)
    colors = cm.brg(np.linspace(.0, 1., 3))

    cams = ['HBCm', 'VBCr', 'VBCl']
    if add_camera:
        cams.append('ARTf')

    lines = []
    R_min, R_max = 0.0, 0.0
    for c, cam in enumerate(cams):
        if 'reduced' in label:
            nCh = channels['reduced']['eChannels'][cam]
        else:
            nCh = channels['full']['eChannels'][cam]

        R = []
        for ch in nCh:
            R.append(camera_info['radius']['reff'][ch] / minor_radius)
        R_min = np.min(R) if np.min(R) < R_min else R_min
        R_max = np.max(R) if np.max(R) > R_max else R_max

        ax.errorbar(
            R,  # chordal_tomo[0, nCh],
            chordal_tomo[1, nCh] * 1000.,
            yerr=xp_error[nCh] * 1000.,
            c=colors[c], ls='-.', lw=0.75, alpha=0.75)
        L0 = ax.errorbar(
            R,  # chordal_xp[0, nCh],
            chordal_xp[1, nCh] * 1000.,
            yerr=xp_error[nCh] * 1000.,
            label=cam, c=colors[c],
            ls='-', lw=0.75, alpha=0.75)
        lines.append(L0)

        if False and (c == 0):
            L1, = ax.plot(
                R,  # chordal_xp[0, nCh],
                chi[nCh],
                c='k', ls='-.', alpha=.5, lw=.5,
                label='$\\chi_{ch}$')
            lines.append(L1)

    L2, = ax.plot([.0, .0], [.0, .0], c='k', ls='-', label='XP')
    L3, = ax.plot([.0, .0], [.0, .0], c='k', ls='-.', label='tomo.')

    lines.append(L2)
    lines.append(L3)

    text_string = '\n'.join((
        '$\\chi^{2}$=' + format(chi2, '.3f'),
        'P$_{rad,XP}$ = ' + format(prad_xp['HBCm'] / 1.e6, '.3f') + 'MW',
        'P$_{rad,tom}$ = ' + format(prad_tomo['HBCm'] / 1.e6, '.3f') + 'MW'))
    # build box
    prop = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(
        .5, .95, text_string, transform=ax.transAxes,
        horizontalalignment='center', verticalalignment='top', bbox=prop)

    ax.legend(
        lines, [l.get_label() for l in lines],
        loc='upper center', bbox_to_anchor=(.5, 1.32),
        ncol=3, fancybox=True, shadow=False,
        handletextpad=0.1, labelspacing=0.2, handlelength=1.5)

    for f in [1., -1.]:
        ax.axvline(f, c='lightgrey', ls='-.')
    ax.set_xlim(R_min, R_max)

    ax.set_xlabel('radius [r$_{a}$]')
    ax.set_ylabel('power [mW]')

    fig.set_size_inches(4.5, 3.)
    pf.fig_current_save('mfr_chordal', fig)
    f = '../results/INVERSION/MFR/' + strgrid
    fig.savefig(
        f + '/' + label + '_XP_chordal_profiles.png',
        bbox_inches='tight', dpi=169.0)
    p.close('all')
    return


def mfr2D_radial_plot(
        label='pos_mesh_x5.75_y0.0_mx1.0e+06',
        kani=np.zeros((2, 30)),
        radial=np.zeros((2, 30)),
        radial_error=0.0,
        absolute_error=0.0,
        peaks_r=None,
        peaks_id=None,
        half_widths=None,
        minor_radius=0.5139,
        nr=30,
        nt=100,
        total=0.0,
        core=0.0,
        strgrid='sN8_30x20x150_1.3',
        solo=False,
        debug=False):
    fig, ax = p.subplots(1, 1)
    if not solo:
        kani_ax = ax.twinx()

    for i, V in enumerate(half_widths[1]):
        L2 = ax.errorbar(
            np.array([half_widths[2][i], half_widths[3][i]]) / minor_radius,
            [V / 1.e6, V / 1.e6],
            yerr=absolute_error / 1.e6, xerr=radial_error,
            color='r', ls='-.', label='peaks & HW')

    L1, = ax.plot(
        radial[0] / minor_radius, radial[1] / 1.e6,
        c='k', alpha=0.75, lw=0.75,
        label='tomogram')
    if not solo:
        L3, = kani_ax.plot(
            kani[0] / minor_radius, kani[1],
            c='b', ls='-.', lw=1., alpha=0.75,
            label='K$_{ani}$')
    ax.set_xlim(
        np.min([np.min(radial[0][np.where(
            ~np.isnan(radial[0]))]), np.min(kani[0]) / 100.]) / minor_radius,
        np.max([np.max(radial[0][np.where(~np.isnan(
            radial[0]))]), np.max(kani[0]) / 100.]) / minor_radius)

    text_string = '\n'.join((
        'integrated power:',  # textstring
        'core = ' + format(core / 1.e6, '.3f') + 'MW',
        'total = ' + format(total / 1.e6, '.3f') + 'MW'))
    # build box
    prop = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(
        .05, .6, text_string, transform=ax.transAxes,
        horizontalalignment='left',
        verticalalignment='top', bbox=prop)

    ax.errorbar(
        peaks_r / minor_radius, radial[1, peaks_id] / 1.e6,
        yerr=absolute_error / 1.e6, xerr=radial_error,
        ls='', c='r', marker='x', markersize=3.)
    ax.axvline(1., c='grey', ls='-.', lw=0.75)

    ax.set_xlabel('r$_{eff}$ [m]')
    ax.set_ylabel('brightness [MW/m$^{3}$]')
    if not solo:
        kani_ax.set_ylabel('weighting [a.u.]')

    if len(half_widths[1]) > 0:
        lines = [L1, L2, L3] if not solo else [L1, L2]
    else:
        lines = [L1, L3] if not solo else [L1]
    ax.legend(
        lines, [l.get_label() for l in lines],
        loc='upper center', bbox_to_anchor=(.5, 1.2),
        ncol=5, fancybox=True, shadow=False,
        handletextpad=0.1, labelspacing=0.2, handlelength=1.)

    pf.fig_current_save('mfr_radial', fig)

    f = '../results/INVERSION/MFR/' + strgrid
    # fig.set_size_inches(4., 3.)
    fig.savefig(
        f + '/' + label + '_XP_radial_profile.png',
        bbox_inches='tight', dpi=169.0)
    p.close('all')
    return


def mfr2D_poloidal(
        tomogram=np.zeros((30, 100)),
        fs_radius=np.zeros((30)),
        minor_radius=0.0,
        label='pos_mesh_x5.75_y0.0_mx1.0e+06',
        strgrid='sN8_50x30x100_1.4',
        base='../results/INVERSION/MFR/',
        debug=False):
    nr, nz = tomogram.shape
    zax = tomogram

    fs_radius *= 1. / minor_radius

    fig, ax = p.subplots()
    norm = mpl.colors.Normalize(
        vmin=fs_radius.min(),
        vmax=fs_radius.max())
    c_m = mpl.colors.LinearSegmentedColormap.from_list(
        'mycolors', ['blue', 'red'])

    s_m = cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])

    for r in range(nr):
        l, = ax.plot(
            np.linspace(0, 360., nz),
            zax[r, :] / 1.e6, alpha=0.5,
            color=s_m.to_rgba(fs_radius[r]))

    ax.set_xlim(.0, 360.)
    ax.set_ylabel('power density [MW/m$^{3}$]')
    ax.set_xlabel('poloidal angle [°]')

    ax.set_title(label.replace('_', ' ')[
        :int(len(label) / 2)] + '\n' + label.replace('_', ' ')[
            int(len(label) / 2):] + '\npoloidal profile')

    cbar_ax = fig.add_axes([1.0, 0.14, 0.05, 0.7])
    fig.colorbar(s_m, cax=cbar_ax)
    cbar_ax.set_ylabel('effective radius [$r / a$]')

    fig.set_size_inches(8., 5.)
    pf.fig_current_save('mfr_poloidal', fig)
    f = 'mfr_poloidal_' + label + '.png'
    fig.savefig(
        '../results/INVERSION/MFR/' + strgrid + '/' + f,
        bbox_inches='tight', dpi=169.0)
    p.close('all')
    return


def check_reverse_plot(
        save_base='../results/INVERSION/MFR/',
        grid=np.zeros((30, 100, 8)),
        emiss=np.zeros((30, 100)),
        debug=False):
    z = np.sum(emiss, axis=1)
    patches = []

    fig, ax = p.subplots(1, 1)
    for n, points in enumerate(grid[:]):
        R = [points[0], points[2], points[4], points[6]]
        Z = [points[1], points[3], points[5], points[7]]
        poly = np.array([R, Z]).transpose()
        polygon = Polygon(poly, aa=False)
        patches.append(polygon)

    plot = PatchCollection(patches, cmap=mpl.cm.viridis)
    colors = z.reshape(-1)
    plot.set_array(colors)
    ax.add_collection(plot)

    cb = fig.colorbar(plot, ax=ax)
    cb.set_label('emissivity [m$^{3}$]')
    ax.legend()

    pf.fig_current_save('emissivity_test', fig)
    fig.savefig(
        save_base + 'emissivity_test.png',
        bbox_inches='tight', dpi=169.0)

    p.close('all')
    return


def phantom_plot(
        nFS=30,
        nL=100,
        phantom=np.zeros((30, 100)),
        mesh=np.zeros((2, 30, 100)),
        label='pos_mesh_x5.75_y0.0_mx1.0e+06',
        strgrid='sN8_50x30x100_1.4',
        VMID='EIM_000',
        save_base='../results/INVERSION/MFR/'):

    patches = []
    fig, ax = p.subplots(1, 1)
    for S in range(nFS):
        for L in range(nL):
            polys = fpp.poly_from_mesh(
                m=mesh, S=S, L=L, nL=nL)

            R = np.array([p[0] for p in polys])
            Z = np.array([p[1] for p in polys])
            poly = np.array([R, Z]).transpose()
            polygon = Polygon(poly, aa=False)
            patches.append(polygon)

    plot = PatchCollection(patches, cmap=mpl.cm.viridis)
    colors = phantom.reshape(-1) / 1.e6
    plot.set_array(colors)
    ax.add_collection(plot)

    cb = fig.colorbar(plot, ax=ax)
    cb.set_label('brightness [MW/m$^{3}$]')

    # load islands and divertor/vessel
    name = 'phi108.0_res10_conf_lin'
    islands = poincare.store_read(name=name, arg='islands1_')
    divertors = poincare.define_divWall(phi=108.)

    if islands is None:
        VMEC_ID = database['values']['magnetic_configurations'][VMID]['URI']
        URI = VMEC_URI + VMEC_ID + '/lcfs.json?phi=108.'
        LCFS = requests.get(URI).json()

        ax.plot(
            LCFS['lcfs'][0]['x1'], LCFS['lcfs'][0]['x3'],
            c='lightgrey', lw=1.5, alpha=0.5, label='LCFS ' + VMID)
    else:  # poincare plots
        poincare_plots.plot_div(axis=ax, points=divertors)
        poincare_plots.PlotPoincare(axis=ax, surfaces=islands)

    ax.axis('off')
    fig.set_size_inches(4.5, 3.)
    pf.fig_current_save('phantom', fig)

    f = '../results/INVERSION/MFR/' + strgrid
    name = 'phantom_' + label.replace('.', '_') + '_' + VMID
    fig.savefig(
        f + '/' + name + '.png',
        bbox_inches='tight', dpi=169.0)
    p.close('all')
    return


def phantom_forward_plot(
        data=Z((128)),
        error=Z((128)),
        cams=['HBCm', 'VBCl', 'VBCr'],
        new_type=None,
        label='pos_mesh_x5.75_y0.0_mx1.0e+06',
        strgrid='sN8_50x30x100_1.4',
        save_base='../results/INVERSION/MFR/',
        debug=False):
    camera_info = dat_lists.geom_dat_to_json()
    fig, ax = p.subplots(1, 1)
    colors = cm.brg(np.linspace(.0, 1., np.shape(cams)[0]))

    rmn, rmx = .0, .0
    for n, cam in enumerate(cams):
        channels, fwd, eps, R = [], [], [], []
        for c, ch in enumerate(camera_info['channels']['eChannels'][cam]):
            channels.append(c)
            fwd.append(data[ch])
            eps.append(error[ch])
            R.append(camera_info['radius']['rho'][ch])

        if cam == 'ARTf':
            cam = new_type
        ax.errorbar(
            x=channels,  # x=R,
            y=np.array(fwd) * 1.e3, yerr=eps,
            ls='-.', marker='x',
            label=cam, markersize=3.,
            c=colors[n], lw=1., alpha=0.75)

        rmn = np.min(channels) if np.min(channels) < rmn else rmn
        rmx = np.max(channels) if np.max(channels) > rmx else rmx
    ax.set_xlim(rmn, rmx)

    fig.set_size_inches(4., 2.5)
    ax.set_ylabel('power [mW]')
    ax.set_xlabel('channel number [#]')
    ax.legend()

    pf.fig_current_save('forward_phantom', fig)
    f = '../results/INVERSION/MFR/' + strgrid
    name = 'phantom_forward_' + label.replace('.', '_')
    fig.savefig(
        f + '/' + name + '.png',
        bbox_inches='tight', dpi=169.0)
    p.close('all')
    return


def phantom_v_tomogram(
        data={'none': None},
        debug=False):
    fig, axes = p.subplots(5, 1, sharex=True)
    label = data['label']

    L1 = []
    # peak location plots
    L1.append(axes[0].plot(  # tomogram
        data['values']['FWHM']['tomogram']['abscissa'],
        data['values']['FWHM']['tomogram']['peaks'],
        label='tomogram', marker='x', ls='', c='r')[0])
    L1.append(axes[0].plot(  # phantom
        data['values']['FWHM']['phantom']['abscissa'],
        data['values']['FWHM']['phantom']['peaks'],
        label='phantom', marker='+', ls='', c='k')[0])

    L2 = []  # half width plots
    hwp = axes[1]
    L2.append(hwp.plot(  # half width phantom
        data['values']['FWHM']['phantom']['abscissa'],
        data['values']['FWHM']['phantom']['half_widths'],
        label='phantom', marker='+', ls='', c='k')[0])
    L2.append(hwp.plot(  # half width tomogram
        data['values']['FWHM']['tomogram']['abscissa'],
        data['values']['FWHM']['tomogram']['half_widths'],
        label='tomogram', marker='x', ls='', c='r')[0])

    L3, totalMSD = [], axes[2].twinx()  # MSD valus for global qualification
    L3.append(axes[2].plot(  # core
        data['values']['abscissa']['values'],
        data['values']['MSD']['core'],
        label='core', marker='<', ls='-.', c='b')[0])
    L3.append(axes[2].plot(  # SOL
        data['values']['abscissa']['values'],
        data['values']['MSD']['SOL'],
        label='SOL', marker='>', ls='-.', c='r')[0])
    L3.append(totalMSD.plot(  # total
        data['values']['abscissa']['values'],
        data['values']['MSD']['total'],
        label='total', marker='x', ls='-.', c='k')[0])

    L4 = []  # total and core power
    L4.append(axes[3].plot(
        data['values']['abscissa']['values'],
        [v / 1.e6 for v in data['values']['power']['core']['tomogram']],
        label='core,tomo.', marker='x', c='b')[0])
    L4.append(axes[3].plot(
        data['values']['abscissa']['values'],
        [v / 1.e6 for v in data['values']['power']['total']['tomogram']],
        label='total,tomo.', marker='x', c='r', ls='-.')[0])
    L4.append(axes[3].plot(
        data['values']['abscissa']['values'],
        [v / 1.e6 for v in data['values']['power']['core']['phantom']],
        label='core,phan.', marker='+', c='g', ls='-.')[0])
    L4.append(axes[3].plot(
        data['values']['abscissa']['values'],
        [v / 1.e6 for v in data['values']['power']['total']['phantom']],
        label='total,phan.', marker='+', c='k', ls='-.')[0])

    # chi squared as measureable means of accuracy
    L5, pearson_ax = [], axes[4].twinx()
    L5.append(axes[4].plot(  # chi2
        data['values']['abscissa']['values'],
        data['values']['chi2'],
        label='$\\chi^{2}$', marker='x', ls='-.', c='k')[0])
    L5.append(pearson_ax.plot(
        data['values']['abscissa']['values'],
        data['values']['MSD']['pearson_coefficient'],
        label='P$_{c}$', marker='^', ls='-.', c='r')[0])

    axes[0].set_ylabel('peak radius [r$_{a}$]')
    axes[1].set_ylabel('FWHM [cm]')
    axes[2].set_ylabel('variance [%]')
    totalMSD.set_ylabel('abs. variance [%]')
    axes[3].set_ylabel('power [MW]')
    axes[4].set_ylabel('$\\chi^{2}$ [a.u.]')
    pearson_ax.set_ylabel('$\\rho_{c}$ [a.u.]')
    axes[4].set_xlabel(data['values']['abscissa']['label'])

    for lines, ax in zip([L1, L2, L3, L4, L5], axes):
        ax.legend(lines, [l.get_label() for l in lines])

    fig.set_size_inches(5., 9.5)
    pf.fig_current_save('phantom_scan_results', fig)

    f = 'phantom_scan_' + label.replace('.', '_') + '.png'
    strgrid = data['strgrid'][0]
    fig.savefig(
        '../results/INVERSION/MFR/' + 'old_comp/' + f,
        # strgrid + '/' + f,
        bbox_inches='tight', dpi=169.0)
    p.close('all')
    return


def plot_XPID_comparison_scan(
        program='20181010.032'):

    f = '../libinversion/libcalc/' + program + '.json'
    with open(f, 'r') as infile:
        data = json.load(infile)
    infile.close()

    abscissa = [33., 66., 90., 100.]  # frad
    xlabel = 'f$_{rad}$ [%]'

    fig, axes = p.subplots(3, 1, sharex=True)
    colors = cm.brg(np.linspace(.0, 1., np.shape(
        data['values']['kani'])[0]))

    axes[0].set_title(program + ' kdiff scan; ' + data['values']['strgrid'])
    for k, kanis in enumerate(data['values']['kani']):
        axes[0].plot(
            abscissa, data['values']['chi2'][k],
            label=str(kanis[1]), marker='x', c=colors[k], ls='-.')

        axes[1].plot(
            abscissa, data['values']['power']['total'][k],
            label=str(kanis[1]), marker='x', c=colors[k], ls='-.')

        axes[2].plot(
            abscissa, data['values']['power']['core'][k],
            label=str(kanis[1]), marker='x', c=colors[k], ls='-.')

    for ax in axes:
        ax.legend()
        ax.set_xlim(abscissa[0], abscissa[-1])

    axes[0].set_ylabel('X$^{2}$ [a.u.]')
    axes[1].set_ylabel('P$_{total}$ [MW]')
    axes[2].set_ylabel('P$_{core}$ [MW]')
    axes[2].set_xlabel(xlabel)

    fig.set_size_inches(7., 9.)
    pf.fig_current_save('program_scan_results', fig)

    strgrid = data['values']['strgrid']
    f = program + '_kdiff_scan_kani_' + str(
        kanis[0]) + '_' + strgrid + '.png'
    fig.savefig(
        '../results/INVERSION/MFR/' + strgrid + '/' + f,
        bbox_inches='tight', dpi=169.0)

    p.close('all')
    return


def tomogram_phantom_wrapper(
        data_object={'none': None},
        strgrid='sN8_30x20x150_1.35',
        add_camera=False,
        debug=False):
    data = data_object['values']
    # load islands and divertor/vessel
    name = 'phi108.0_res10_conf_lin'
    islands = poincare.store_read(name=name, arg='islands1_')
    divertors = poincare.define_divWall(phi=108.)

    comp_tomogram_phantom(
        label=data_object['label'], strgrid=strgrid,
        VMEC_ID=data['VMEC_ID'], tomogram=data['tomogram'],
        phantom=data['phantom'], edge_pattern='poincare',
        msd=data['difference']['mean_square_deviation'],
        r_grid=data['r_grid'], z_grid=data['z_grid'],
        islands=islands, divertors=divertors)

    profiles, peaks, = data['profiles'], data['peaks']
    comp_profiles(
        strgrid=strgrid, add_camera=add_camera,
        label=data_object['label'], minor_radius=data['minor_radius'],
        total_power=data['total_power'], core_power=data['core_power'],
        tomogram_radial=profiles['radial_tomogram'],
        phantom_radial=profiles['radial_phantom'],
        radial_error=profiles['radial_error'],
        absolute_error=profiles['abs_error'],
        phantom_chordal=profiles['chordal_phantom'],
        tomogram_chordal=profiles['chordal_tomogram'],
        xp_error=profiles['xp_error'],
        prad_hbc_tomo=data['P_rad']['tomogram']['HBCm'],
        prad_hbc_phan=data['P_rad']['phantom']['HBCm'],
        peaks_Rt=peaks['radial_pos']['tomogram'],
        peaks_It=peaks['index_pos']['tomogram'],
        HWt=peaks['half_widths']['tomogram'],
        peaks_Rp=peaks['radial_pos']['phantom'],
        peaks_Ip=peaks['index_pos']['phantom'],
        HWp=peaks['half_widths']['phantom'],
        grad_profile=data['dphi_profile'],
        chi2=data['difference']['chi2'])

    comp_poloidal_map(
        label=data_object['label'], strgrid=strgrid,
        tomogram=data['tomogram'], phantom=data['phantom'],
        minor_radius=data['minor_radius'],
        fs_radius=profiles['radial_tomogram'][0, :])
    return


def comp_tomogram_phantom(
        tomogram=np.zeros((30, 100)),
        phantom=np.zeros((30, 100)),
        msd=np.zeros((30, 100)),
        r_grid=np.zeros((30, 100, 4)),
        z_grid=np.zeros((30, 100, 4)),
        islands=np.zeros((4, 15, 100)),
        divertors=np.shape((4, 525, 2)),
        edge_pattern='LCFS',
        VMEC_ID='EIM_000',
        label='pos_mesh_x5.75_y0.0_mx1.0e+06',
        strgrid='sN8_50x30x100_1.4',
        base='../results/INVERSION/FS/',
        debug=False):
    fig, axes = p.subplots(1, 3, sharey=True)
    for ax in axes:
        if islands is None:
            VMEC_ID = database['values'][
                'magnetic_configurations'][VMEC_ID]['URI']
            URI = VMEC_URI + VMEC_ID + '/lcfs.json?phi=108.'
            LCFS = requests.get(URI).json()

            ax.plot(
                LCFS['lcfs'][0]['x1'], LCFS['lcfs'][0]['x3'],
                c='lightgrey', lw=1.5, alpha=0.5)
        else:
            # poincare plots
            poincare_plots.plot_div(axis=ax, points=divertors)
            poincare_plots.PlotPoincare(axis=ax, surfaces=islands)
        ax.axis('off')

    patches = []
    for S in range(np.shape(tomogram)[0]):
        for L in range(np.shape(tomogram)[1]):
            poly = np.array([
                r_grid[S, L, :],
                z_grid[S, L, :]]).transpose()
            polygon = Polygon(poly, aa=False)
            patches.append(polygon)

    pP = PatchCollection(patches, cmap=mpl.cm.viridis)
    tP = PatchCollection(patches, cmap=mpl.cm.viridis)
    mP = PatchCollection(patches, cmap=mpl.cm.hot)

    tC = tomogram.reshape(-1) / 1.e6
    pC = phantom.reshape(-1) / 1.e6

    pP.set_array(pC)
    pP.set_clim([np.min([pC]), np.max([pC])])  # [pC, tC]
    axes[0].add_collection(pP)

    tP.set_array(tC)
    tP.set_clim([np.min([tC]), np.max([tC])])  # [pC, tC]
    axes[1].add_collection(tP)

    mC = msd.reshape(-1)
    mP.set_array(mC)
    axes[2].add_collection(mP)

    for ax, plot in zip([axes[0], axes[1]], [pP, tP]):
        cb = fig.colorbar(plot, ax=ax, shrink=0.9)
        cb.set_label('brightness [MW/m$^{3}$]')
    cb = fig.colorbar(mP, ax=axes[2], shrink=0.9)
    cb.set_label('deviation [%]')

    fig.set_size_inches(10., 2.5)
    pf.fig_current_save('phantom_v_tomo2D', fig)
    f = 'phantom_v_tomo2D_' + label + '.png'
    fig.savefig(
        '../results/INVERSION/MFR/' + strgrid + '/' + f,
        bbox_inches='tight', dpi=169.0)

    p.close('all')
    return


def comp_profiles(
        total_power={'none': None},
        core_power={'none': None},
        tomogram_radial=np.zeros((2, 128)),
        tomogram_chordal=np.zeros((2, 128)),
        phantom_radial=np.zeros((2, 128)),
        phantom_chordal=np.zeros((2, 128)),
        radial_error=0.0,
        absolute_error=0.0,
        xp_error=np.zeros((128)),
        peaks_Rt=np.zeros((10)),
        peaks_It=np.zeros((10)),
        HWt=np.zeros((4, 10)),
        peaks_Rp=np.zeros((10)),
        peaks_Ip=np.zeros((10)),
        HWp=np.zeros((4, 10)),
        grad_profile=np.zeros((2, 30)),
        prad_hbc_tomo=0.0,
        prad_hbc_phan=0.0,
        minor_radius=0.5139,
        chi2=0.0,
        label='pos_mesh_x5.75_y0.0_mx1.0e+06',
        strgrid='sN8_50x30x100_1.4',
        solo=False,
        add_camera=False,
        debug=False):
    fig, axes = p.subplots(1, 2)
    if not solo:
        kani_ax = axes[0].twinx()

    for i, V in enumerate(HWt[1]):
        axes[0].errorbar(
            np.array([HWt[2][i], HWt[3][i]]) / minor_radius,
            [V / 1.e6, V / 1.e6],
            yerr=absolute_error / 1.e6,
            xerr=radial_error,
            color='r', ls='-.')
    for i, V in enumerate(HWp[1]):
        axes[0].errorbar(
            np.array([HWp[2][i], HWp[3][i]]) / minor_radius,
            [V / 1.e6, V / 1.e6],
            yerr=absolute_error / 1.e6,
            xerr=radial_error,
            color='k', ls='-.')

    L1, = axes[0].plot(
        phantom_radial[0] / minor_radius,
        phantom_radial[1] / 1.e6,
        c='k', alpha=0.75, lw=0.75,
        label='phantom')
    L2, = axes[0].plot(
        tomogram_radial[0] / minor_radius,
        tomogram_radial[1] / 1.e6,
        c='r', alpha=0.75, lw=0.75,
        label='tomogram')

    if not solo:
        L3, = kani_ax.plot(
            grad_profile[0] / minor_radius,
            grad_profile[1],
            c='blue', ls='-.', lw=1., alpha=0.75,
            label='d$r$')
        L4, = kani_ax.plot(
            grad_profile[0] / minor_radius,
            grad_profile[2],
            c='green', ls='-.', lw=1., alpha=0.75,
            label='d$\\vartheta$')
        L5, = kani_ax.plot(
            grad_profile[0] / minor_radius,
            grad_profile[3],
            c='purple', ls='-.', lw=1., alpha=0.75,
            label='d$\\vartheta$\'')

    if not (peaks_It == []):
        axes[0].errorbar(
            peaks_Rt / minor_radius,
            tomogram_radial[1, peaks_It] / 1.e6,
            yerr=absolute_error / 1.e6, xerr=radial_error,
            ls='', c='r', marker='x', markersize=5.)
    if not (peaks_Ip == []):
        axes[0].errorbar(
            peaks_Rp / minor_radius,
            phantom_radial[1, peaks_Ip] / 1.e6,
            yerr=absolute_error / 1.e6, xerr=radial_error,
            ls='', c='k', marker='x', markersize=5.)

    axes[0].axvline(1., c='grey', ls='-.', lw=0.75)
    text_string = '\n'.join((
        'integrated power:',  # textstring
        'tomogram = ' + format(total_power['tomogram'] / 1.e6, '.3f') + 'MW',
        'phantom = ' + format(total_power['phantom'] / 1.e6, '.3f') + 'MW'))
    # build box
    prop = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
    # place a text box in upper left in axes coords
    axes[0].text(
        .05, .93, text_string, transform=axes[0].transAxes,
        horizontalalignment='left',
        verticalalignment='top', bbox=prop)

    cams = ['HBCm', 'VBCl', 'VBCr']
    if add_camera:
        cams.append('ARTf')

    colors = cm.brg(np.linspace(.0, 1., np.shape(cams)[0]))
    channels = mfr_transf.mfr_channel_list()
    R_min, R_max = 0.0, 0.0
    for n, cam in enumerate(cams):
        if 'reduced' in label:
            nCh = channels['reduced']['eChannels'][cam]
        else:
            nCh = channels['full']['eChannels'][cam]

        # R_min = np.min(phantom_chordal[0, nCh]) / minor_radius \
        #     if np.min(phantom_chordal[0, nCh]) < R_min \
        #     else R_min / minor_radius
        # R_max = np.max(phantom_chordal[0, nCh]) / minor_radius \
        #     if np.max(phantom_chordal[0, nCh]) > R_max \
        #     else R_max / minor_radius
        R_min = .0
        R_max = len(nCh) - 1 if len(nCh) - 1 > R_max else R_max

        axes[1].errorbar(
            # phantom_chordal[0, nCh] / minor_radius,
            np.linspace(.0, len(nCh) - 1, len(nCh)),
            phantom_chordal[1, nCh] * 1000.,
            yerr=xp_error[nCh] * 1000.,
            c=colors[n], alpha=0.75, lw=0.75)
        axes[1].errorbar(
            # tomogram_chordal[0, nCh] / minor_radius,
            np.linspace(.0, len(nCh) - 1, len(nCh)),
            tomogram_chordal[1, nCh] * 1000.,
            yerr=xp_error[nCh] * 1000.,
            c=colors[n], ls='-.', alpha=0.75, lw=0.75)

        axes[1].text(
            # (tomogram_chordal[0, nCh] / minor_radius)[
            np.linspace(.0, len(nCh) - 1, len(nCh))[
                np.argmax(tomogram_chordal[1, nCh])] + 1.,
            np.max(tomogram_chordal[1, nCh] * 1000.),
            cam, color=colors[n])

    axes[1].plot([.0, .0], [.0, .0], c='k', ls='-', label='tomogram')
    axes[1].plot([.0, .0], [.0, .0], c='k', ls='-.', label='phantom')

    axes[0].set_xlim(
        np.min([np.min(phantom_radial[0][np.where(
            ~np.isnan(phantom_radial[0]))]),
            np.min(grad_profile[0])]) / minor_radius,
        np.max([np.max(phantom_radial[0][np.where(~np.isnan(
            phantom_radial[0]))]),
            np.max(grad_profile[0])]) / minor_radius)
    axes[1].set_xlim(R_min, R_max)

    text_string = '\n'.join((
        '$\\chi^{2}$=' + format(chi2, '.3f'),
        'P$_{rad,pha}$ = ' + format(prad_hbc_phan / 1.e6, '.3f') + 'MW',
        'P$_{rad,tom}$ = ' + format(prad_hbc_tomo / 1.e6, '.3f') + 'MW'))
    # build box
    prop = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
    # place a text box in upper left in axes coords
    axes[1].text(
        .8, .93, text_string, transform=axes[1].transAxes,
        horizontalalignment='center',
        verticalalignment='top', bbox=prop)

    axes[1].set_ylabel('power [mW]')
    for ax in axes:
        ax.set_xlabel('radius [r$_{a}$]')
    axes[1].set_xlabel('channel [#]')
    # axes[1].axvline(-1., c='grey', ls='-.', lw=0.75)
    # axes[1].axvline(1., c='grey', ls='-.', lw=0.75)

    if not solo:
        kani_ax.set_ylabel('weighting [a.u]')
    axes[0].set_ylabel('brightness [MW/m$^{3}$]')

    lines = [L1, L2] if solo else [L1, L2, L3, L4, L5]
    axes[0].legend(
        lines, [l.get_label() for l in lines],
        loc='upper center', bbox_to_anchor=(.5, 1.2),
        ncol=5, fancybox=True, shadow=False,
        handletextpad=0.1, labelspacing=0.2, handlelength=1.)
    axes[1].legend(
        loc='upper center', bbox_to_anchor=(.5, 1.2),
        ncol=2, fancybox=True, shadow=False,
        handletextpad=0.1, labelspacing=0.2, handlelength=2.)

    fig.set_size_inches(10., 3.)
    pf.fig_current_save('phantom_v_tomo_profiles', fig)

    f = 'phantom_v_tomo_profiles_' + label + '.png'
    fig.savefig(
        '../results/INVERSION/MFR/' + strgrid + '/' + f,
        bbox_inches='tight', dpi=169.0)
    p.close('all')
    return


def comp_poloidal_map(
        tomogram=np.zeros((30, 100)),
        phantom=np.zeros((30, 100)),
        fs_radius=np.zeros((30)),
        minor_radius=0.0,
        label='pos_mesh_x5.75_y0.0_mx1.0e+06',
        strgrid='sN8_50x30x100_1.4',
        base='../results/INVERSION/MFR/',
        debug=False):
    fs_radius *= 1. / minor_radius

    nr, nz = tomogram.shape
    fig, axes = p.subplots(1, 2, sharey=True)
    for ax, zax in zip(axes, [tomogram, phantom]):
        norm = mpl.colors.Normalize(
            vmin=fs_radius.min(),
            vmax=fs_radius.max())

        # c_m = cm.RdPu
        c_m = mpl.colors.LinearSegmentedColormap.from_list(
            'mycolors', ['blue', 'red'])

        s_m = cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])

        for r in range(nr):
            l, = ax.plot(
                np.linspace(0, 360., nz),
                zax[r, :] / 1.e6, alpha=0.75,
                color=s_m.to_rgba(fs_radius[r]))

        ax.set_xlim(.0, 360.)
        ax.set_xlabel('poloidal angle [°]')
    axes[0].set_ylabel('brightness [MW/m$^{3}$]')

    cbar_ax = fig.add_axes([1.0, 0.27, 0.05, 0.65])
    fig.colorbar(s_m, cax=cbar_ax)
    cbar_ax.set_ylabel('radius [r$_{a}$]')

    fig.set_size_inches(7., 2.)
    pf.fig_current_save('phantom_v_tomo_poloidal', fig)
    f = 'phantom_v_tomo_poloidal_' + label + '.png'
    fig.savefig(
        '../results/INVERSION/MFR/' + strgrid + '/' + f,
        bbox_inches='tight', dpi=169.0)
    p.close('all')
    return


def pow_forward_plot(
        program='20181010.032',
        cams=['HBCm', 'VBCl', 'VBCr'],
        camera_info={'none': None},
        power=np.zeros((128, 10000)),
        error=np.zeros((128)),
        target=0.62,  # s
        time=np.zeros((10000)),  # s
        debug=False):
    fig, ax = p.subplots(1, 1)
    colors = cm.brg(np.linspace(0, 1., len(cams)))
    ind = (np.abs([t - target for t in time])).argmin()
    P = power[:, ind]
    for c, camera in enumerate(cams):
        nCh = camera_info['channels']['eChannels'][camera]

        ax.errorbar(
            x=np.linspace(0, len(nCh), len(nCh)),
            y=P[nCh], yerr=error[nCh], c=colors[c],
            label=camera, ls='-.')

    ax.legend()
    ax.set_title(program + ' @ ' + format(target, '.2f') + 's')
    ax.set_ylabel('power [W]')
    ax.set_xlabel('geom. channels')
    pf.fig_current_save('mfr_pow_forward', fig)
    p.close('all')
    return


def poincare_scatter_plot(
        poincare={'x': None, 'y': None, 'z': None, 'r': None},
        phi=107.9,
        points=200,
        step=0.2,
        magconf_ID=0,
        N_3D_points=120,
        shift_space=False):
    fig, ax = p.subplots(1, 1)
    colors = cm.brg(np.linspace(.0, 1., np.shape(poincare['r'])[0]))
    for i in range(np.shape(poincare['r'][::2])[0]):
        ax.plot(
            poincare['r'][i], poincare['z'][i],
            c=colors[i], lw=.5, alpha=.5)  # s=2., marker='o')
    p.show()
    return


def emissivity_comparison_plot(
        channels={'none': None},
        pih_grid=np.zeros((14 * 100, 8)),
        HBCm_pih=np.zeros((20, 14 * 100)),
        VBCr_pih=np.zeros((20, 14 * 100)),
        VBCl_pih=np.zeros((20, 14 * 100)),
        daz_grid=np.zeros((14 * 100, 8)),
        HBCm_daz=np.zeros((20, 14 * 100)),
        VBCr_daz=np.zeros((20, 14 * 100)),
        VBCl_daz=np.zeros((20, 14 * 100))):

    for D, L in zip([[HBCm_daz, HBCm_pih], [VBCl_daz, VBCl_pih],
                     [VBCr_daz, VBCr_pih]], ['HBCm', 'VBCl', 'VBCr']):
        daz0 = D[0]  # [c]  # np.sum(D[0], axis=0)
        pih0 = D[1]  # [c]  # np.sum(D[1], axis=0)

        gCh = channels['eChannels'][L]
        for ch, c in enumerate(gCh):
            daz = daz0[ch]
            pih = pih0[ch]

            fig, axes = p.subplots(2, 1, sharex=True)
            daz_max, pih_max = np.max(daz), np.max(pih)
            for n in range(np.shape(daz)[0]):
                axes[0].fill(
                    daz_grid[n, 0::2], daz_grid[n, 1::2],
                    c=cm.viridis(daz[n] / daz_max))

            for n in range(np.shape(pih)[0]):
                axes[1].fill(
                    pih_grid[n, 0::2], pih_grid[n, 1::2],
                    c=cm.viridis(pih[n] / pih_max))

            for ax, D in zip(axes, [daz, pih]):
                cax, _ = cbar.make_axes(ax, anchor=(3.0, 0.0))
                normal = p.Normalize(np.min(D), np.max(D))
                cb = cbar.ColorbarBase(cax, cmap=cm.viridis, norm=normal)
                cb.set_label('brightness [power/volume]')

            f = 'tm_' + L + '_' + str(c) + \
                '_reduced_comparison_14x100_23x25'
            fig.savefig(
                '../results/INVERSION/MFR/14x100/comparison/' + f + '.png',
                bbox_inches='tight', dpi=169.0)
            p.close('all')
    return


def plot_tomogram_statistics(
        data=None,
        group='none'):

    def base_plot(
            list1, list2, f1, f2, l1, l2,
            maxY=1., maxX=1., label=None, c='k', cl='r', ax=None):
        if ax is None:
            fig, ax = p.subplots(1, 1)
        ax.set_xlabel(l1)
        ax.set_ylabel(l2)

        max, min = -np.inf, np.inf
        for a, b in zip(list1, list2):
            if isinstance(a, float) and isinstance(b, float):
                a, b = [a], [b]

            if len(a) > 0 and len(b) > 0:
                L = len(a) if len(a) < len(b) else len(b)
                P, = ax.plot(
                    [x * f1 for x in a[:L]],
                    [x * f2 for x in b[:L]],
                    ls='None', lw=0.0, alpha=1., markersize=3.,
                    label=label, color=c, marker='x')

                max = max if max > np.max(a) else np.max(a)
                min = min if min < np.min(a) else np.min(a)

        if not (('chi' in l1) or ('chi' in l2) or (
                'MSD' in l1) or ('MSD' in l2) or (
                    'P$_{c}$' in l1) or ('P$_{c}$' in l2)):
            # plot slope m=1 line in range, b=0
            x = np.linspace(min, max, 1000)
            ax.plot(x * f1, x * f2, c=cl, lw=0.5, ls='-.')

        ax.set_xlim(.0, maxX)
        if 'P$_{c}$' in l1:
            ax.set_xlim(0.35, maxX)
        ax.set_ylim(.0, maxY)

        if (label is not None):
            return (P)
        return

    def deltaP_overP():
        power, delta, PuP, Pdown = [], [], [], []
        try:
            for i, P in enumerate(data['power']['total']['tomogram']):
                if data['power']['error'][i] != 0.0:
                    power.append(P)
                    delta.append(data['power']['error'][i])
                    PuP.append(P + data['power']['error'][i])
                    Pdown.append(P - data['power']['error'][i])
        except Exception:
            print('\\\ no errors found')
        return (np.array([
            (x, y, z, u) for x, y, z, u in
            sorted(zip(power, delta, PuP, Pdown))]).transpose())

    delta_power = deltaP_overP()

    if 'comp' in group:
        N = 9 if 'comp' in group else 3
        fig, axes = p.subplots(4, 2)
        fig.set_size_inches(9., 2.2 * 4)
        axes = [
            axes[0, 0], axes[1, 0], axes[2, 0], axes[3, 0],
            axes[0, 1], axes[1, 1], axes[2, 1], axes[3, 1]]
    else:
        fig, axes = p.subplots(3, 1)
        fig.set_size_inches(4.5, 6.5)

    if 'comp' in group:
        # base_plot(
        #     list1=data['peaks']['widths']['tomogram'],
        #     list2=data['peaks']['widths']['phantom'],
        #     f1=1.e-2, f2=1.e-2, maxY=0.2, maxX=0.23,
        #     l1='FWHM$_{tom}$ width [m]', l2='FWHM$_{phan}$ width [m]',
        #     c='k', cl='r', ax=axes[0])

        base_plot(
            list1=data['peaks']['position']['tomogram'],
            list2=data['peaks']['position']['phantom'],
            f1=1., f2=1., maxY=0.7, maxX=0.7,
            l1='FWHM$_{tom}$ location [m]',
            l2='FWHM$_{phan}$ location [m]',
            c='k', cl='r', ax=axes[0])

        base_plot(
            list1=data['peaks']['height']['tomogram'],
            list2=data['peaks']['height']['phantom'],
            f1=1.e-6, f2=1.e-6, maxY=1., maxX=1.,
            l1='FWHM$_{tom}$ max [MW/m$^{3}$]',
            l2='FWHM$_{phan}$ max [MW/m$^{3}$]',
            c='k', cl='r', ax=axes[1])

        P1 = base_plot(
            list1=data['P_rad']['tomogram']['HBCm'],
            list2=data['P_rad']['phantom']['HBCm'],
            label='HBCm', f1=1.e-6, f2=1.e-6,
            l1='', l2='', c='r', cl='k', ax=axes[2])
        P2 = base_plot(
            list1=data['P_rad']['tomogram']['VBC'],
            list2=data['P_rad']['phantom']['VBC'],
            label='VBC', f1=1.e-6, f2=1.e-6, maxY=25., maxX=25.,
            l1='P$_{rad, tom}$ [MW]', l2='P$_{rad, phan}$ [MW]',
            c='b', cl='k', ax=axes[2])
        axes[2].legend([P1, P2], [l.get_label() for l in [P1, P2]])

        P1 = base_plot(
            list1=data['power']['core']['tomogram'],
            list2=data['power']['core']['phantom'],
            label='core', f1=1.e-6, f2=1.e-6, l1='', l2='',
            c='b', cl='k', ax=axes[3])
        P2 = base_plot(
            list1=data['power']['total']['tomogram'],
            list2=data['power']['total']['phantom'],
            label='total', f1=1.e-6, f2=1.e-6, maxY=25., maxX=25.,
            l1='power (tom.) [MW]',
            l2='power (phan.) [MW]',
            c='r', cl='k', ax=axes[3])
        axes[3].legend([P1, P2], [l.get_label() for l in [P1, P2]])

        base_plot(
            list1=data['difference']['xi2'],
            list2=data['difference']['msd'],
            f1=1., f2=1., maxY=2., maxX=4.,
            l1='$\\chi^{2}$', l2='MSD [%]',
            c='k', cl='r', ax=axes[4])

        P1 = base_plot(
            list1=data['difference']['pearson'],
            list2=data['difference']['xi2'],
            label='$\\chi^2$', f1=1., f2=1.,
            l2='', l1='', c='r', cl='k', ax=axes[5])
        P2 = base_plot(
            list2=data['difference']['msd'],
            list1=data['difference']['pearson'],
            label='MSD', f1=1., f2=1., maxY=2., maxX=1.05,
            l2='a.u.', l1='P$_{c}$',
            c='b', cl='k', ax=axes[5])
        axes[5].legend([P1, P2], [l.get_label() for l in [P1, P2]])

        P1 = base_plot(
            list1=data['power']['core']['tomogram'],
            list2=data['P_rad']['phantom']['HBCm'],
            label='HBC', f1=1.e-6, f2=1.e-6,
            l1='', l2='', c='r', cl='k', ax=axes[6])
        P2 = base_plot(
            list1=data['power']['core']['tomogram'],
            list2=data['P_rad']['phantom']['VBC'],
            label='VBC', f1=1.e-6, f2=1.e-6, maxY=20., maxX=20.,
            l1='power (core) [MW]',
            l2='P$_{rad, phan}$ [MW]',
            c='b', cl='k', ax=axes[6])
        axes[6].legend([P1, P2], [l.get_label() for l in [P1, P2]])

        axes[6].plot(
            data['power']['core']['lin_HBCm'][2][0] * 1.e-6,
            data['power']['core']['lin_HBCm'][2][1] * 1.e-6,
            c='r', ls=':', lw=0.75, alpha=0.75)
        textString1 = 'P$_{HBC}$=' + \
            format(data['power']['core']['lin_HBCm'][0], '.1f') + \
            'P$_{core}$+' + \
            format(data['power']['core']['lin_HBCm'][1] * 1.e-6, '.1f') + 'MW'

        axes[6].plot(
            data['power']['core']['lin_VBC'][2][0] * 1.e-6,
            data['power']['core']['lin_VBC'][2][1] * 1.e-6,
            c='b', ls=':', lw=0.75, alpha=0.75)
        textString2 = 'P$_{VBC}$=' + \
            format(data['power']['core']['lin_VBC'][0], '.1f') + \
            'P$_{core}$' + \
            format(data['power']['core']['lin_VBC'][1] * 1.e-6, '.1f') + 'MW'
    
        prop = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
        axes[6].text(
            .75, .25, textString1 + '\n' + textString2,
            transform=axes[6].transAxes, horizontalalignment='center',
            verticalalignment='top', bbox=prop)

        P1 = base_plot(
            list1=data['power']['total']['tomogram'],
            list2=data['P_rad']['phantom']['HBCm'],
            label='HBC', f1=1.e-6, f2=1.e-6, l1='',
            l2='', c='r', cl='k', ax=axes[7])
        P2 = base_plot(
            list1=data['power']['total']['tomogram'],
            list2=data['P_rad']['phantom']['VBC'],
            label='VBC', f1=1.e-6, f2=1.e-6, maxY=20., maxX=25.,
            l1='power (total) [MW]', l2='P$_{rad, phan}$ [MW]',
            c='b', cl='k', ax=axes[7])
        axes[7].legend([P1, P2], [l.get_label() for l in [P1, P2]])

        axes[7].plot(
            data['power']['total']['lin_HBCm'][2][0] * 1.e-6,
            data['power']['total']['lin_HBCm'][2][1] * 1.e-6,
            c='r', ls=':', lw=0.75, alpha=0.75)
        textString1 = 'P$_{HBC}$=' + \
            format(data['power']['total']['lin_HBCm'][0], '.1f') + \
            'P$_{tot}$' + \
            format(data['power']['total']['lin_HBCm'][1] * 1.e-6, '.1f') + 'MW'

        axes[7].plot(
            data['power']['total']['lin_VBC'][2][0] * 1.e-6,
            data['power']['total']['lin_VBC'][2][1] * 1.e-6,
            c='b', ls=':', lw=0.75, alpha=0.75)
        textString2 = 'P$_{VBC}$=' + \
            format(data['power']['total']['lin_VBC'][0], '.1f') + \
            'P$_{tot}$' + \
            format(data['power']['total']['lin_VBC'][1] * 1.e-6, '.1f') + 'MW'

        prop = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
        axes[7].text(
            .8, .25, textString1 + '\n' + textString2,
            transform=axes[7].transAxes, horizontalalignment='center',
            verticalalignment='top', bbox=prop)

        if (np.shape(delta_power)[0] > 0):
            if (np.shape(delta_power[0])[0] > 2):
               for ax in [axes[4], axes[7], axes[8]]:
                    ax.fill_between(
                        delta_power[0], delta_power[2], delta_power[3],
                        facecolor='coral', alpha=0.8)

        pf.fig_current_save('phantom_tomogram_statistics', fig)
        fig.savefig('../results/INVERSION/MFR/phantom_statistics_' +
                    group.replace('comp_phantom_tomogram', '') + '.png')

    elif 'xp' in group:
        P1 = base_plot(
            list1=data['P_rad']['tomogram']['HBCm'],
            list2=data['P_rad']['phantom']['HBCm'],
            label='HBC', f1=1.e-6, f2=1.e-6, l1='',
            l2='', c='r', cl='k', ax=axes[0])
        P2 = base_plot(
            list1=data['P_rad']['tomogram']['VBC'],
            list2=data['P_rad']['phantom']['VBC'],
            label='VBC', f1=1.e-6, f2=1.e-6, maxY=4., maxX=4.,
            l1='P$_{rad, tom}$ [MW]',
            l2='P$_{rad, exp}$ [MW]', c='b', cl='k', ax=axes[0])
        axes[0].legend([P1, P2], [l.get_label() for l in [P1, P2]])

        P1 = base_plot(
            list1=data['power']['core']['tomogram'],
            list2=data['P_rad']['phantom']['HBCm'],
            label='HBC', f1=1.e-6, f2=1.e-6, l1='',
            l2='', c='r', cl='k', ax=axes[1])
        P2 = base_plot(
            list1=data['power']['core']['tomogram'],
            list2=data['P_rad']['phantom']['VBC'],
            label='VBC', f1=1.e-6, f2=1.e-6, maxX=4., maxY=4.,
            l1='power (core) [MW]', l2='P$_{rad, xp}$ [MW]',
            c='b', cl='k', ax=axes[1])
        axes[1].legend([P1, P2], [l.get_label() for l in [P1, P2]])

        axes[1].plot(
            data['power']['core']['lin_HBCm'][2][0] * 1.e-6,
            data['power']['core']['lin_HBCm'][2][1] * 1.e-6,
            c='r', ls=':', lw=0.75, alpha=0.75)
        textString1 = 'P$_{HBC}$=' + \
            format(data['power']['core']['lin_HBCm'][0], '.1f') + \
            'P$_{core}$+' + \
            format(data['power']['core']['lin_HBCm'][1] * 1.e-6, '.1f') + 'MW'

        axes[1].plot(
            data['power']['core']['lin_VBC'][2][0] * 1.e-6,
            data['power']['core']['lin_VBC'][2][1] * 1.e-6,
            c='b', ls=':', lw=0.75, alpha=0.75)
        textString2 = 'P$_{VBC}$=' + \
            format(data['power']['core']['lin_VBC'][0], '.1f') + \
            'P$_{core}$+' + \
            format(data['power']['core']['lin_VBC'][1] * 1.e-6, '.1f') + 'MW'

        prop = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
        axes[1].text(
            .75, .25, textString1 + '\n' + textString2,
            transform=axes[1].transAxes, horizontalalignment='center',
            verticalalignment='top', bbox=prop)

        P1 = base_plot(
            list1=data['power']['total']['tomogram'],
            list2=data['P_rad']['phantom']['HBCm'],
            label='HBC', f1=1.e-6, f2=1.e-6, l1='',
            l2='', c='r', cl='k', ax=axes[2])
        P2 = base_plot(
            list1=data['power']['total']['tomogram'],
            list2=data['P_rad']['phantom']['VBC'],
            label='VBC', f1=1.e-6, f2=1.e-6, maxX=4., maxY=4.,
            l1='power (total) [MW]', l2='P$_{rad, xp}$ [MW]',
            c='b', cl='k', ax=axes[2])
        axes[2].legend([P1, P2], [l.get_label() for l in [P1, P2]])

        axes[2].plot(
            data['power']['total']['lin_HBCm'][2][0] * 1.e-6,
            data['power']['total']['lin_HBCm'][2][1] * 1.e-6,
            c='r', ls=':', lw=0.75, alpha=0.75)
        textString1 = 'P$_{HBC}$=' + \
            format(data['power']['total']['lin_HBCm'][0], '.1f') + \
            'P$_{tot}$' + \
            format(data['power']['total']['lin_HBCm'][1] * 1.e-6, '.1f') + 'MW'

        axes[2].plot(
            data['power']['total']['lin_VBC'][2][0] * 1.e-6,
            data['power']['total']['lin_VBC'][2][1] * 1.e-6,
            c='b', ls=':', lw=0.75, alpha=0.75)
        textString2 = 'P$_{VBC}$=' + \
            format(data['power']['total']['lin_VBC'][0], '.1f') + \
            'P$_{tot}$+' + \
            format(data['power']['total']['lin_VBC'][1] * 1.e-6, '.1f') + 'MW'

        axes[2].text(
            .75, .25, textString1 + '\n' + textString2,
            transform=axes[2].transAxes, horizontalalignment='center',
            verticalalignment='top', bbox=prop)

        if (np.shape(delta_power)[0] > 0):
            if (np.shape(delta_power)[1] > 2):
                for ax in axes[1:]:
                    ax.fill_between(
                        delta_power[0] * 1.e-6, delta_power[2] * 1.e-6,
                        delta_power[3] * 1.e-6, facecolor='coral', alpha=0.8)

        pf.fig_current_save('exp_tomogram_statistics', fig)
        fig.savefig('../results/INVERSION/MFR/exp_statistics_' +
                    group.replace('results_xp_tomogram_', '') + '.png')

    p.close('all')
    return


def power_balance_plot(
        program='20181010.032',
        ecrh=np.zeros((2, 1000)),
        hbc=np.zeros((2, 1000)),
        vbc=np.zeros((2, 1000)),
        dwdia_dt=np.zeros((2, 1000)),
        div_load=np.zeros((2, 1000)),
        balance=np.zeros((4, 1000)),
        tomograms=np.zeros((3, 1000))):

    fig, axes = p.subplots(2, 1)
    fig.set_size_inches(5., 6.)

    for xy, c, S, L in zip(
            [ecrh, hbc, vbc, dwdia_dt, div_load],
            ['k', 'r', 'b', 'orange', 'g'],
            ['-', '--', '-.', ':', '-.'],
            ['ECRH', 'P$_{rad,HBC}$', 'P$_{rad,VBC}$',
             'dW$_{dia}$/dt', 'P$_{div}$']):
        axes[0].plot(
            xy[0], xy[1] * 1.e-6, c=c, ls=S, alpha=0.75, label=L)

    # axes[1].plot(
    #     balance[0], balance[1] * 1.e-6, lw=0.75, alpha=0.75, c='r',
    #     label='P$_{bal,HBC}$')
    axes[1].plot(
        balance[0], balance[3] * 1.e-6, c='k', lw=0.75, alpha=0.75,
        ls='-.', label='P$_{bal}$')

    axes[1].plot(
        tomograms[0], tomograms[1] * 1.e-6, marker='x', c='b',
        ls='None', label='P$_{bal,total}^{tom}$')
    axes[1].plot(
        tomograms[0], tomograms[2] * 1.e-6, marker='+', c='r',
        ls='None', label='P$_{bal,core}^{tom}$')
    
    axes[0].set_xlim(-.1, 13.)
    axes[1].set_xlim(np.min(tomograms[0]) - 0.1,
                     np.max(tomograms[0]) + 0.1)
    # axes[2].set_ylim(np.min(tomograms[1:]) * 1.e-6 - 0.05,
    #                  np.max(tomograms[1:]) * 1.e-6 + 0.05)

    axes[0].legend(
        loc='upper center', bbox_to_anchor=(.5, 1.3),
        ncol=3, fancybox=True, shadow=False,
        handletextpad=0.1, labelspacing=0.2, handlelength=2.)
    axes[1].legend(ncol=2)
    
    for ax in axes:
        ax.set_xlabel('time [s]')
        ax.set_ylabel('power [MW]')

    pf.fig_current_save('tomogram_power_balance', fig)
    fig.savefig('../results/INVERSION/MFR/tomogram_power_balance_' +
                program + '.png')
    p.close()  # p.show()
    return
