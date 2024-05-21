""" **************************************************************************
    start of file """

import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as p

import mClass
import plot_funcs as pf

Z = np.zeros
ones = np.ones
line = np.linspace

""" eo header
************************************************************************** """


def detector_plot(
        camera_info={'channels': {'eChannels': {}}, 'geometry': {}},
        geometry={'none': None},
        corners={'none': None},
        projection='2d',
        view_alpha=20.,
        view_beta=60.,
        add_camera=False,
        label='EIM_beta000',
        loc='../results/INVERSION/CAMGEO/',
        debug=False):
    # plotting the detector(s) corners/rectangle and aperture
    corner_geom = mClass.dict_transf(corners, to_list=False)

    c, a = corner_geom['values'], corner_geom['aperture']
    xc, yc, zc, rc = c['x'], c['y'], c['z'], c['r']

    if not add_camera:
        cams = ['HBCm', 'VBCr', 'VBCl']
    else:
        cams = ['HBCm', 'VBCr', 'VBCl', 'ARTf']

    for k, C in enumerate(cams[:]):
        channels = camera_info['channels']['eChannels'][C]
        print('\tPlotting detector(s) corners of ' + str(channels) + ' ...')

        fig = p.figure()
        if projection == '2d':
            ax = fig.add_subplot(111)
        elif projection == '3d':
            ax = fig.gca(projection='3d')

        cs = cm.brg(np.linspace(
            .0, 1., np.shape(channels)[0]))
        for i, ch in enumerate(channels[:]):
            if debug:
                print(i, '/', np.shape(channels)[0], '\t', ch)

            if projection == '3d':
                ax.plot(xc[ch], yc[ch], zc[ch], ls='-.', c=cs[i], marker='+',
                        markersize=5., lw=0.5, alpha=0.4)
            #     ax.plot(xb[ch], yb[ch], zb[ch], ls='', c=cs[i], marker='x',
            #             markersize=5., lw=0.5, alpha=0.4)

            elif projection == '2d':
                ax.plot(rc[ch], zc[ch], ls='-.', c=cs[i], marker='+',
                        markersize=5., lw=0.5, alpha=0.4)
            #     ax.plot(rb[ch], zb[ch], ls='', c=cs[i], marker='x',
            #             markersize=5., lw=0.5, alpha=0.4)

            if i == 0:
                apt_c = a[C]
                if debug:
                    print(apt_c, np.shape(apt_c['x']))

                if projection == '3d':
                    ax.plot(apt_c['x'], apt_c['y'], apt_c['z'],
                            c=cs[i], marker='x',
                            markersize=5., lw=0.5,
                            alpha=0.6)  # , label='apt.' + C)
                    # ax.plot(apt['x'], apt['y'], apt['z'],
                    #         c=cs[i], marker='x',
                    #         ls='', markersize=5.,
                    #         lw=0.5, alpha=0.6)

                elif projection == '2d':
                    ax.plot(apt_c['r'], apt_c['z'],
                            c=cs[i], marker='x',
                            markersize=5., lw=0.5,
                            alpha=0.6)  # , label='apt.' + C)
                    # ax.plot(apt['r'], apt['z'],
                    #         c=cs[i], marker='x',
                    #         ls='', markersize=5.,
                    #         lw=0.5, alpha=0.6)

        # ax.legend()
        if projection == '3d':
            ax.view_init(view_alpha, view_beta)
            ax.set_zlabel("Z [m]")
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            pf.autoscale_x(ax)

        else:
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")
            pf.autoscale_y(ax)

        pf.fig_current_save('detectors', fig)
        fig.savefig(
            '../results/INVERSION/CAMGEO/detectors_' + projection +
            label + '_' + C + '.png',
            bbox_inches='tight', dpi=169.0)
        # p.show()
        p.close('all')
    return


def plot_final_triangulation(
        base=np.zeros((5, 3)),  # 5 points, 3 coords
        triangles=np.zeros((4, 3, 3)),  # all triangles, 3 points, 3 coords
        centers=np.zeros((4, 3)),  # all centers 3 coords
        N=4,  # number of triags
        L='label'):
    fig, ax = p.subplots()
    ax = fig.gca(projection='3d')

    ax.plot(
        base[:, 0],
        base[:, 1],
        base[:, 2], c='k', lw=0.5, alpha=0.5, marker='o')

    colors = cm.brg(np.linspace(.0, 1., N))
    for n in range(N):
        T = triangles[n, :, :]
        ax.plot(
            T[:, 0],
            T[:, 1],
            T[:, 2],
            c=colors[n], lw=0.5, alpha=0.5, marker='^')

        C = centers[n]
        ax.plot(
            [C[0], C[0]],
            [C[1], C[1]],
            [C[2], C[2]], c=colors[n], marker='x')

    pf.fig_current_save('final_' + L, fig)
    # p.show()
    p.close('all')
    return
