""" **************************************************************************
    start of file """

import numpy as np
import os
import matplotlib.pyplot as p
from matplotlib.pyplot import cm

import mClass
import plot_funcs as pf

Z = np.zeros
one = np.ones

""" eo header
************************************************************************** """


def plot_FS(
        fsdata_object={'none': None},
        surface=[108.],
        vmec_label='EIM_000',
        suffix='2d',
        view_alpha=20.,
        view_beta=60.,
        debug=False):
    """ fluxsurface plotting
    Args:
        fsdata_object (0, dict, optional): Fluxsurfaces
        surface (1, TYPE, optional): Which to plot
        angles (2, TYPE, optional): Toroidal angles the surfaces are at
        vmec_label (3, str, optional): Label of VMEC entry
        debug (4, bool, optional): Debugging
    Notes:
        None.
    """
    print('\t\tPlotting fluxsurfaces', end=' ')
    FS = fsdata_object['values']['fs']
    LCFS = fsdata_object['values']['lcfs']
    iota = fsdata_object['values']['iota']
    pressure = fsdata_object['values']['pressure']
    angles = fsdata_object['values']['angles']

    fig = p.figure()
    cs = cm.brg(np.linspace(0, 1., np.shape(surface)[0]))
    if suffix == '3d':
        ax = fig.gca(projection='3d')
        for j, phi in enumerate(surface):
            ix, phi = mClass.find_nearest(angles, phi)

            if j == 0:
                print('in 3d')

            for i, S in enumerate(FS[ix]):
                ax.plot(S[0], S[1], S[2], c=cs[j], alpha=0.4, lw=0.5)
                if i == 0:
                    ax.plot(LCFS[ix][0], LCFS[ix][1], LCFS[ix][2], c=cs[j],
                            lw=1., label='FS / LCFS ' +
                            str(format(phi, '.2f')))

        pf.autoscale_x(ax)
        pf.autoscale_y(ax)

        ax.legend()
        ax.set_xlabel("X / m")
        ax.set_ylabel("Y / m")
        ax.set_zlabel("Z / m")
        ax.set_title("Archive@Geiger, VMEC " + vmec_label)
        ax.view_init(view_alpha, view_beta)

    elif suffix == '2d':
        fig.subplots_adjust(right=0.87)
        iota_press = fig.add_subplot(212)
        fluxsurf = fig.add_subplot(211)
        pressureax = iota_press.twinx()

        pressureax.spines['right'].set_position(('axes', 1.0))
        pressureax.set_ylabel('pressure / Pa')

        iota_press.set_xlabel('radial coord. s')
        iota_press.set_ylabel('Iota')
        iota_press.set_title("Archive@Geiger, iota & pressure " + vmec_label)

        iotap, = iota_press.plot(iota, label='iota', ls='-.', c='r')
        pressurep, = pressureax.plot(pressure, label='pressure', c='b')

        pressureax.yaxis.label.set_color(pressurep.get_color())
        pressureax.spines["right"].set_edgecolor(pressurep.get_color())
        pressureax.tick_params(axis='y', colors=pressurep.get_color())

        iota_press.legend([iotap, pressurep],
                          [l.get_label() for l in [iotap, pressurep]])
        iota_press.yaxis.label.set_color(iotap.get_color())
        iota_press.spines["right"].set_edgecolor(iotap.get_color())
        iota_press.tick_params(axis='y', colors=iotap.get_color())

        for j, phi in enumerate(surface):
            ix, phi = mClass.find_nearest(angles, phi)

            if j == 0:
                print('in 2d')
            for i, S in enumerate(FS[ix]):
                fluxsurf.plot(S[3], S[2], c=cs[j], lw=.5, alpha=0.4)

                if i == 0:
                    fluxsurf.plot(LCFS[ix][3], LCFS[ix][2],
                                  lw=1., c=cs[j],
                                  label='FS / LCFS ' + str(format(phi, '.2f')))

        pf.autoscale_x(fluxsurf)
        pf.autoscale_y(fluxsurf)

        fluxsurf.legend()
        fluxsurf.set_xlabel("R / m")
        fluxsurf.set_ylabel("Z / m")
        fluxsurf.set_title("Archive@Geiger, VMEC " + vmec_label)

    location = "../results/INVERSION/"
    if not os.path.exists(location):
        os.makedirs(location)

    # saving
    pf.fig_current_save('fluxsurfaces', fig)
    fig.savefig('../results/INVERSION/FS/fs_' +
                vmec_label + '_' + suffix + '.png',
                bbox_inches='tight', dpi=169.0)
    p.close("all")

    return


def slanted_fs_plot(
        slanted_FS=Z((23, 4, 80)),
        axis=Z((4)),
        fluxsurfaces={'none': None},
        surface=[108.],
        suffix='2d',
        vmec_label='EIM_000',
        camera='HBCm',
        view_alpha=20.,
        view_beta=60.,
        debug=False):
    if camera in ['VBCl', 'ARTf']:
        print('\t\tPlotting slanted FS', end=' ')

    angles = fluxsurfaces['values']['angles']
    FS = fluxsurfaces['values']['fs']
    center = fluxsurfaces['values']['magnetic_axes']
    ix, phi = mClass.find_nearest(angles, surface[0])

    fig = p.figure()
    cs = cm.brg(np.linspace(0, 1., np.shape(slanted_FS)[0]))
    if suffix == '3d':
        ax = fig.gca(projection='3d')
        if camera in ['VBCl', 'ARTf']:
            print('in 3d')

        for i, S in enumerate(slanted_FS):
            ax.plot(S[0], S[1], S[2], c=cs[i], marker='+',
                    markersize=2., alpha=0.6, lw=0.7)
            ax.plot(
                FS[ix, i, 0, :],
                FS[ix, i, 1, :],
                FS[ix, i, 2, :],
                c='k', lw=.5, alpha=0.3)

            ax.scatter(axis[0], axis[1], axis[2],
                       marker='+', c='k', markersize=1.)
            ax.scatter(center[ix][0], center[ix][1], center[ix][2],
                       marker='*', markersize=1., c='k')

        ax.set_xlabel("X / m")
        ax.set_ylabel("Y / m")
        ax.set_zlabel("Z / m")
        ax.view_init(view_alpha, view_beta)

    elif suffix == '2d':
        fig.subplots_adjust(right=0.87)
        fluxsurf = fig.add_subplot(111)
        if camera in ['VBCl', 'ARTf']:
            print('in 2d')

        for i, S in enumerate(slanted_FS):
            fluxsurf.plot(S[3], S[2], c=cs[i], lw=.7, alpha=0.6,
                          marker='+', markersize=2.)
            fluxsurf.plot(FS[ix, i, 3, :], FS[ix, i, 2, :],
                          lw=.5, alpha=0.3, c='k')

            fluxsurf.plot(axis[3], axis[2],
                          marker='+', c='k', markersize=1.)
            fluxsurf.plot(center[ix][3], center[ix][2],
                          marker='*', c='k', markersize=1.)

        fluxsurf.set_xlabel("R / m")
        fluxsurf.set_ylabel("Z / m")

    location = "../results/INVERSION/"
    if not os.path.exists(location):
        os.makedirs(location)

    # saving
    pf.fig_current_save('slanted_fluxsurfaces_' + camera, fig)
    fig.savefig('../results/INVERSION/FS/slantend_fs_' +
                vmec_label + '_' + suffix + '_' + camera + '.png',
                bbox_inches='tight', dpi=169.0)
    p.close("all")
