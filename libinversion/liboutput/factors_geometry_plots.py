""" **************************************************************************
    so header """

import os
import numpy as np
from itertools import cycle

import matplotlib.pyplot as p
from matplotlib.pyplot import cm

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import mClass
import plot_funcs as pf
import poincare_plots

# database
import requests
import database_inv as database
database = database.import_database()
VMEC_URI = 'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/'

Z = np.zeros
ones = np.ones

""" end of header
************************************************************************** """


def LOS_K_factor(
        camera_info={'none': None},
        cameras=['HBCm', 'VBCl', 'VBCr'],
        emiss=Z((128)),
        volum=Z((128)),
        full_volum=Z((128)),
        daz_K=Z((128)),
        daz_V=Z((128)),
        label='EIM_beta000_slanted_HBCm_50_12',
        show=False):
    """ Volume and K factor for channels/LoS
    Keyword Arguments:
        camera_info {dict} -- general cam info (default: {{'none': None}})
        ch {list} -- channels ({[int(ch) for ch in np.linspace(0, 31, 32)]})
        label {str} -- data label
                       (default: {'EIM_beta000_slanted_HBCm_50_12'})
        extrapolate {int} -- extrapolate FS number (default: {12})
        emiss {[type]} -- emissivity of channel (default: {Z((128, 51, 33))})
        volum {[type]} -- volume of LoS (default: {Z((128, 51, 33))})
        daz_K {[type]} -- OG factors (default: {Z((128))})
        daz_V {[type]} -- OG volumes (default: {Z((128))})
    """
    markers = [
        '+', 'v', '^', 'o', 's', 'h', '1', '2',
        '3', '*', 'd', '<', '>', '$c$', '$u$']
    cs = cm.brg(np.linspace(.0, 1., len(cameras)))

    for c, cam in enumerate(cameras):
        fig = p.figure()
        axes = []
        axes.append(fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[]))
        axes.append(fig.add_axes([0.1, 0.1, 0.8, 0.4]))

        axes[0].set_ylabel('etendue [cm$^{3}$]')
        axes[1].set_ylabel('volume [cm$^{3}$]')
        axes[1].set_xlabel('channel number [#]')
        ch = camera_info['channels']['eChannels'][cam]

        axes[0].plot(
            # ch,
            [emiss[c] * (100.**3) for c in ch],
            ls='', marker=markers[0], c='k', alpha=0.75,
            label=cam)  # 'K$_{pih,' + cam + '}$')

        axes[1].plot(
            # ch,
            [volum[c] * (100.**3) for c in ch],
            ls='', marker=markers[0], c='k', alpha=0.75,
            label='center cone')  # cam)  # 'V$_{pih c,' + cam + '}$')
        if True:
            axes[1].plot(
                # ch,
                [full_volum[c] * (100.**3) for c in ch],
                ls='', marker=markers[9], c='red', alpha=0.75,
                label='full cone')  # 'V$_{pih f,' + cam + '}$')
        if False:
            axes[1].plot(
                # ch,
                [(volum[c] + full_volum[c] * 0.1) * (100.**3) for c in ch],
                ls='', marker=markers[c * 3], c=cs[c], alpha=0.75,
                label=cam)  # 'V$_{pih +,' + cam + '}$')
        if True:
            axes[1].legend()

        if False and cam != 'ARTf':
            axes[0].plot(
                # ch,
                [daz_K[c] * (100.**3) for c in ch],
                ls='', marker=markers[c * 2], c=cs[c], alpha=0.75,
                label=cam)  # 'K$_{daz,' + cam + '}$')
            axes[1].plot(
                # ch,
                [daz_V[c] * (100.**3) for c in ch],
                ls='', marker=markers[c * 2], c=cs[c], alpha=0.75,
                label=cam)  # 'V$_{daz,' + cam + '}$')

        if not show:
            fig.set_size_inches(4., 4.)
            pf.fig_current_save('LOS_KV_factor', fig)
            fig.savefig(
                '../results/INVERSION/PROFILE/' + label +
                '/LOS_KV_factor_' + cam + '.png',
                bbox_inches='tight', dpi=169.0)
            p.close('all')
        else:
            p.show()
    return


def angles_area(
        factors={'none': None},
        camera_info={'none': None},
        cameras=['HBCm', 'VBCl', 'VBCr'],
        label='EIM_beta000_slanted_HBCm_50_12'):
    """ plot angles and detector areas
    Keyword Arguments:
        fangl {dict} -- factors and geometry (default: {{'none': None}})
        camera_info {dict} -- general cam info (default: {{'none': None}})
        label {str} -- data label (default: {'EIM_beta000_slanted_HBCm_50_12'})
    """
    cs = cm.brg(np.linspace(.0, 1., len(cameras)))
    markers = [
        '+', '*', 'v', '^', 'o', 's', 'h', '1', '2',
        '3', 'd', '<', '>', '$c$', '$u$']

    for c, cam in enumerate(cameras):
        fig = p.figure()
        axes = []
        axes.append(fig.add_axes([0.1, 0.74, 0.8, 0.32], xticklabels=[]))
        axes.append(fig.add_axes([0.1, 0.42, 0.8, 0.32], xticklabels=[]))
        axes.append(fig.add_axes([0.1, 0.1, 0.8, 0.32]))

        axes[0].set_ylabel('dist. apt. [cm]')
        axes[1].set_ylabel('LOS angle [°]')
        axes[2].set_ylabel('det. area [mm$^{2}$]')
        axes[2].set_xlabel('channel number [#]')

        ch = camera_info['channels']['eChannels'][cam]
        axes[0].plot(
            # ch,
            np.mean(factors['d'], axis=1)[ch] * 100.,
            ls='', marker=markers[0], c='k', alpha=0.75,
            label=cam)  # 'd$_{' + cam + '}$')

        axes[1].plot(
            # ch,
            np.mean(factors['angles']['alpha'], axis=1)[ch],
            ls='', marker=markers[0], c='k', alpha=0.75,
            label='$\\alpha$')  # '$\\alpha_{' + cam + '}$')
        axes[1].plot(
            # ch,
            np.mean(factors['angles']['beta'], axis=1)[ch],
            ls='', marker=markers[1], c='r', alpha=0.75,
            label='$\\beta$')  # '$\\beta_{' + cam + '}$')
        axes[1].legend()

        axes[2].plot(
            # ch,
            np.sum(factors['areas']['A_d'], axis=1)[ch] * (1000.**2),
            ls='', marker=markers[0], c='k', alpha=0.75,
            label=cam)  # 'A$_{d,' + cam + '}$')

        if False:
            axes[2].text(
                np.mean(ch), np.mean(
                    np.sum(factors['areas']['A_d'], axis=1)[ch] * 1000.**2),
                'A$_{apt,' + cam + '}$=' + format(np.sum(
                    factors['areas']['A_s'][cam]) * 1.e6, '.3f') + 'mm²', c='k')

        fig.set_size_inches(5., 5.)
        pf.fig_current_save('angles_areas', fig)
        fig.savefig(
            '../results/INVERSION/PROFILE/' + label +
            '/angles_areas_' + cam + '.png', bbox_inches='tight', dpi=169.0)
        p.close('all')
    return


def LoS_reff(
        reff={'none': None},
        camera_info={'none': None},
        cams=['HBCm', 'VBCl', 'VBCr'],
        label='test',
        minor_radius=0.5139,
        show=True):
    """ radius along LOS plot
    Args:
        reff (dict, optional): Radii. Defaults to {'none': None}.
        camera_info (dict, optional): Cam info. Defaults to {'none': None}.
        cams (list, optional): Cameras. Defaults to ['HBCm', 'VBCl', 'VBCr'].
        label (str, optional): Name. Defaults to 'test'.
        minor_radius (float, optional): Minor plasma radius. Defaults to 0.5139.
        show (bool, optional): Show me? Defaults to True.
    """
    colors = cm.brg(np.linspace(.0, 1., len(cams)))

    for c, cam in enumerate(cams):
        fig, ax = p.subplots(1, 1)
        ax.set_ylabel('radius [r$_{a}$]')
        ax.set_xlabel('channel no [#]')

        nCh = camera_info['channels']['eChannels'][cam]
        ax.plot(
            # nCh,
            reff['minimum'][nCh] / minor_radius, c='k',
            ls='', marker='x', markersize=4., label='min.')
        ax.plot(
            # nCh,
            reff['emiss'][nCh] / minor_radius, c='r',
            ls='', marker='*', markersize=4., label='geom.')

        ax.legend()
        ax.axhline(1., ls='-.', c='grey')
        ax.axhline(-1., ls='-.', c='grey')

        fig.set_size_inches(4., 3.)
        if not show:
            pf.fig_current_save('LoS_reff', fig)
            fig.savefig('../results/INVERSION/PROFILE/' + label +
                        '/LoS_reff_' + cam + '.png', bbox_inches='tight', dpi=169.0)
            p.close('all')
        if show:
            p.show()
    return


def poly_from_mesh(
        m=Z((2, 30, 125)),
        S=0,
        L=0,
        nFS=30,
        nL=125):
    """ gets polygon corners from mesh given the indices
    Keyword Arguments:
        m {[type]} -- mesh (default: {Z((2, 30, 125))})
        S {int} -- first index (default: {0})
        L {int} -- second (default: {0})
        nFS {int} -- second (default: {30})
        nL {int} -- first dimension size (default: {125})
    Returns:
        (p1, p2, p3, p4) {tuple(ndarray)} -- corner or polygon
    """
    if S == nFS - 1:
        S = S - 1
    # OG point
    p1 = m[:, S, L]
    # second point on line one FS further
    p2 = m[:, S + 1, L]
    if L == nL - 1:
        L = -1
    # fourth point on FS next line
    p4 = m[:, S, L + 1]
    # third point on next FS next line
    p3 = m[:, S + 1, L + 1]
    return (p1, p2, p3, p4)


def FS_channel_property(
        property=np.zeros((128, 64, 30, 125)),
        LoS_r=np.zeros((128, 64, 1000)),
        LoS_z=np.zeros((128, 64, 1000)),
        lines=np.zeros((128, 16, 30, 125)),
        mesh=Z((2, 30, 125)),
        islands=np.zeros((4, 15, 100)),
        divertors=np.shape((4, 525, 2)),
        nFS=30,
        nL=125,
        N=8,
        channels=[int(c) for c in np.linspace(0., 31., 32)],
        property_tag='FS_reff_cell',
        cam='HBCm',
        label='test',
        x_label='R [m]',
        y_label='Z [m]',
        c_label='[a.u.]',
        show_LOS=False,
        show=True,
        debug=False):
    """ plotting properties on FS cells in mesh
    Keyword Arguments:
        property_tag {str} -- name of property (default: {'FS_reff_cell'})
        location {str} -- location to  save to (default: {'REFF/'})
        property {[type]} -- data package (default: {np.zeros((128, 33, 51))})
        cam_scale {[type]} -- camera property scaling (default: {None})
        ch_scale {[type]} -- indiv. channel propertz scaling (default: {None})
        cam {str} -- camera (default: {'HBCm'})
        channels {list} -- ({[int(c) for c in np.linspace(0., 31., 32)]})
        label {str} -- info label (default: {'EIM_beta000_slanted_HBCm_50_12'})
        LoS_r {[type]} -- r dim of lines of sight ({np.zeros((128, 1000))})
        LoS_z {[type]} -- y dim of LoS (default: {np.zeros((128, 1000))})
        line_seg {[type]} -- line segments in mesh ({np.zeros((128, 33, 51))})
        nFS {int} -- number of fluxsurfaces (default: {33})
        nL {int} -- number of poloidal segments (default: {50})
        mesh {[type]} -- mesh data for polygons (default: {Z((51 * 33))})
        x_label {str} -- x label (default: {'R [m]'})
        y_label {str} -- y label (default: {'Z [m]'})
        c_label {str} -- colour label (default: {'r{eff}$ [m]'})
        debug {bool} -- debugging bool (default: {False})
    """
    fig_cam, ax_cam = p.subplots(1, 1)
    ax_cam.set_xlabel(x_label)
    ax_cam.set_ylabel(y_label)

    prop_color_cam = np.zeros((nFS, nL))

    if show_LOS:  # LOS over mesh
        for n, c in enumerate(channels):
            if ((LoS_r is not None) and (LoS_z is not None)):
                for T in range(N**2):
                    # LOS for channel ch
                    R = np.array(LoS_r[c, T])
                    Z = np.array(LoS_z[c, T])
                    ax_cam.plot(R, Z, c='lightgrey', alpha=0.5, lw=0.5)
                ax_cam.text(R[-1], Z[-1], str(c), size=9., c='grey')

    if ((LoS_r is None) and (LoS_z is None)):
        ax_cam.set_xlim(
            np.min(mesh[0]) - 0.05, np.max(mesh[0]) + 0.05)
        ax_cam.set_ylim(
            np.min(mesh[1]) - 0.05, np.max(mesh[1]) + 0.05)

    patches = []
    for S in range(nFS):
        for L in range(nL):
            p1, p2, p3, p4 = poly_from_mesh(
                m=mesh, S=S, L=L, nL=nL, nFS=nFS)

            poly = np.array([
                [p[0] for p in [p1, p2, p3, p4]],
                [p[1] for p in [p1, p2, p3, p4]]]).transpose()
            polygon = Polygon(poly, aa=False)
            patches.append(polygon)

            if 'phi' in property_tag:
                c_cam = property[S, L]
                prop_color_cam[S, L] = c_cam
            elif 'reff' in property_tag:
                c_cam = np.mean(property[channels, :, S, L])
                prop_color_cam[S, L] = c_cam
            else:
                c_cam = np.sum(property[channels, :, S, L], axis=(0, 1))
                prop_color_cam[S, L] = c_cam

    plot = PatchCollection(patches, cmap=cm.viridis)
    colors = prop_color_cam.reshape(-1)
    plot.set_array(colors)
    plot.set_clim([np.min(colors), np.max(colors)])
    ax_cam.add_collection(plot)
    cb = fig_cam.colorbar(plot, ax=ax_cam)
    cb.set_label(c_label)

    if islands is None:
        VMID = label[:11].replace('beta', '')
        VMEC_ID = database['values']['magnetic_configurations'][VMID]['URI']
        URI = VMEC_URI + VMEC_ID + '/lcfs.json?phi=108.'
        LCFS = requests.get(URI).json()

        ax_cam.plot(
            LCFS['lcfs'][0]['x1'], LCFS['lcfs'][0]['x3'],
            c='lightgrey', lw=1.5, alpha=0.5, label='LCFS ' + VMID)

    # poincare plots
    poincare_plots.plot_div(axis=ax_cam, points=divertors)
    poincare_plots.PlotPoincare(axis=ax_cam, surfaces=islands)
    ax_cam.axis('off')

    fig_cam.set_size_inches(5., 4.)
    if not show:
        pf.fig_current_save(property_tag, fig_cam)
        fig_cam.savefig(
            '../results/INVERSION/PROFILE/' + label +
            '/' + property_tag + '_all_' +
            cam + '.png', bbox_inches='tight', dpi=169.0)
        p.close('all')
    if show:
        p.show()
    return


def channel_property_sum(
        property=np.zeros((128, 64, 30, 125)),
        lines=np.zeros((128, 64, 30, 125)),
        camera_info={'none': None},
        cameras=['HBCm', 'VBCl', 'VBCr'],
        property_tag='channel_reff_mean',
        cam='HBCm',
        label='test',
        x_label='channel no [#]',
        y_label='[a.u.]',
        show=True,
        debug=False):
    colors = cm.brg(np.linspace(.0, 1., len(cameras)))
    if 'line' in property_tag:
        property = np.sum(property, axis=(1, 2, 3))

    for c, cam in enumerate(cameras):
        fig, ax = p.subplots(1, 1)
        if 'line' in property_tag:
            ax.set_ylim(0.0, property.max() * 1.025)

        nCh = camera_info['channels']['eChannels'][cam]
        if 'line' in property_tag:
            ax.plot(
                nCh, property[nCh],
                c='k', ls='', label=cam,
                marker='x', markersize=4.)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        fig.set_size_inches(4., 2.5)
        if not show:
            pf.fig_current_save(property_tag, fig)
            fig.savefig(
                '../results/INVERSION/PROFILE/' + label +
                '/' + property_tag + '_' + cam + '.png',
                bbox_inches='tight', dpi=169.0)
            p.close('all')

        if show:
            p.show()
    return


def forward_chordal_profile(
        cam='HBCm',
        label='EIM_beta000_sN8_30x20x150_1.3',
        chord=[Z((128))],
        reff=Z((128)),
        reff_label='min.',
        ids=['00091', '00092', '00093', '00094'],
        labels=['f$_{rad}\\sim$0.33', 'f$_{rad}\\sim$0.66',
                'f$_{rad}\\sim$0.9', 'f$_{rad}\\sim$1.0'],
        camera_info={'none': None},
        vmecID='1000_1000_1000_1000_+0000_+0000/01/00jh_l/',
        m_R=0.5387891957782814):
    """ plot forward chord profile
    Args:
        cam (str, optional): Camera. Defaults to 'HBCm'.
        label (str, optional): Label of geometry.
                               Defaults to 'EIM_beta000_50_10'.
        chord (list, optional): Chord profiles. Defaults to [Z((128))].
        reff ([type], optional): Radii along LOS. Defaults to Z((128)).
        reff_label (str, optional): Label of mapping. Defaults to 'min.'.
        ids (list, optional): STRAHL names.
                              Defaults to ['00091', '00092', '00093', '00094'].
        labels (list, optional): STRAHL labels.
                                 Defaults to ['f{rad}\\sim.33',
                                 'f{rad}\\sim.66', 'f{rad}\\sim.9',
                                 'f{rad}\\sim[summary].0'].
        camera_info (dict, optional): Camera information.
                                      Defaults to {'none': None}.
        m_R (float, optional): Minor radius. Defaults to 0.5387891957782814.
    """
    fig = p.figure()
    ax = fig.add_subplot(111)
    ar = ax.twiny()
    colors = cm.brg(np.linspace(0, 1., len(labels)))

    try:
        m_R = requests.get(
            'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/' +
            vmecID + 'minorradius.json').json()['minorRadius']
    except Exception:
        pass

    R, CH = [], []
    broken = []  # camera_info['channels']['droplist']
    for ch in camera_info['channels']['eChannels'][cam]:
        if ch not in broken:
            R.append(reff[ch] / m_R)
            CH.append(ch)

    # split middle and flip with low opacity to check symmetry
    j = mClass.find_nearest(np.array(R), 0.0)[0] + 1
    R = np.array(R)

    index_list = np.argsort(np.array(R))
    sort_channels = np.array(CH)[index_list]
    sort_R = np.array(R)[index_list]

    for i, L in enumerate(labels):
        chord_profile = chord[i]

        C = []
        for ch in CH:
            C.append(chord_profile[ch])

        fp, = ax.plot(  # profile
            sort_R,
            # sort_channels,
            C, label=L,
            marker='x', markersize=4,
            alpha=.75, color=colors[i])

        # left and right
        ax.plot(
            sort_R[:j],
            # sort_channels[:j],
            np.flipud(C[-j:]),
            alpha=.25, ls='-.', c=colors[i])
        ax.plot(
            sort_R[-j:],
            # sort_channels[-j:],
            np.flipud(C[:j]),
            alpha=.25, ls='-.', c=colors[i])

    ax.legend(
        loc='upper center', bbox_to_anchor=(.5, 1.5),
        ncol=4, fancybox=True, shadow=False,
        handletextpad=0.3, labelspacing=0.5, handlelength=1.)

    textstring = 'mirrored: $-\\cdot-\\cdot-\\cdot-$'
    # build box
    prop = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(
        .3, .8, textstring, transform=ax.transAxes,
        verticalalignment='top', bbox=prop)

    # chrodal profile edge
    ar.set_xlabel('channel #')
    ax.set_xlabel('radius [r$_{a}$]')
    ax.set_ylabel('brightness [kWm$^{-3}$]')

    # ax.axvline(-1., lw=1.0, c='darkgrey', alpha=.5, ls='-.')
    # ax.axvline(1., lw=1.0, c='darkgrey', alpha=.5, ls='-.')
    # ax.text(m_R - .05, np.max(chord) * 0.1, 'r$_{LCFS}$')
    # ax.set_xlim(np.min(R), np.max(R))

    for axis in [ax, ar]:
        # axis.set_xlim(np.min(sort_channels), np.max(sort_channels))
        axis.set_xlim(np.min(sort_R), np.max(sort_R))

    tick_locs = ax.get_xticks()
    N = int(np.ceil(
        # np.shape(sort_channels)[0] / np.shape(tick_locs)[0]))
        np.shape(sort_R)[0] / np.shape(tick_locs)[0]))

    # ar.set_xticks(sort_channels[::N])
    # ar.set_xticklabels([str(round(r, 2)) for r in sort_R[::N]])
    ar.set_xticklabels([str(n) for n in sort_channels[::N]])

    # chrodal profile full normal
    fig.set_size_inches(5., 3.5)
    if cam == 'HBCm' and reff_label == 'min.':
        pf.fig_current_save('forward_chord', fig)

    if not os.path.exists('../results/INVERSION/PROFILE/' + label + '/'):
        os.mkdir('../results/INVERSION/PROFILE/' + label + '/')
    # fig.savefig(
    #     '../results/INVERSION/PROFILE/' + label + '/' +
    #     'forward_chord_' + cam + str(ids).replace(',', '_').replace(
    #         ']', '_').replace('[', '_').replace("'", '').replace(
    #         ' ', '') + reff_label.replace('.', '') + '.png',
    #     bbox_inches='tight', dpi=169.0)

    fig.savefig(
        'forward_chord_' + cam + str(ids).replace(',', '_').replace(
            ']', '_').replace('[', '_').replace("'", '').replace(
            ' ', '') + reff_label.replace('.', '') + '_' + label + '.png',
        bbox_inches='tight', dpi=169.0)
    p.close('all')
    return
