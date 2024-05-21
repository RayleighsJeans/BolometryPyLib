""" **************************************************************************
    so header """

import numpy as np
import matplotlib.pyplot as p
from matplotlib.pyplot import cm

import plot_funcs as pf
import factors_geometry_plots as fgp

Z = np.zeros
ones = np.ones

""" end of header
************************************************************************** """


def mesh_plot(
        mesh={'none': None},
        fluxsurfaces={'none': None},
        fsnumber=1,
        lines=50,
        extrapolate=0,
        cams=['HBCm', 'VBCr', 'VBCl'],
        vmec_label='EIM_000',
        cartesian=False):
    print('\t\tPlotting 2D mesh...')

    for c, cam in enumerate(cams):
        fig, ax = p.subplots(1, 1)

        crosspoints = mesh['values']['crosspoints'][cam]
        line_Z = mesh['values']['lines'][cam]['z']
        line_R = mesh['values']['lines'][cam]['r']

        FS = fluxsurfaces['values']['fs'][cam][fsnumber]
        if not cartesian:
            N = np.shape(FS)[0] - 1 + extrapolate
        elif cartesian:
            N = lines
        c = cm.brg(np.linspace(0, 1., N * lines))

        if not cartesian:
            for i, surf in enumerate(FS):
                ax.plot(surf[3], surf[2], c='k', lw=0.5, alpha=0.3)

        for i, rl in enumerate(line_R):
            ax.plot(rl, line_Z[i], c='k', lw=0.5, alpha=0.5)

        for i, points in enumerate(crosspoints[0, :]):
            ax.plot(crosspoints[0, i], crosspoints[1, i], linestyle='',
                    marker='+', markersize=4., c=c[i], alpha=0.7)

        for S in range(N):
            ax.plot(np.append(crosspoints[
                0, S * lines:(S + 1) * lines], crosspoints[0, S * lines]),
                np.append(crosspoints[
                    1, S * lines:(S + 1) * lines], crosspoints[1, S * lines]),
                linewidth=0.3, c='red', alpha=0.3, ls='-.')

        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')

        pf.fig_current_save('mesh_2D_' + cam, fig)
        if not cartesian:
            fig.savefig('../results/INVERSION/MESH/mesh_' + cam +
                        vmec_label + '_' + str(extrapolate) +
                        '_' + str(lines) + '.png',
                        bbox_inches='tight', dpi=169.0)

        elif cartesian:
            fig.savefig('../results/INVERSION/MESH/mesh_cartesian_' +
                        str(lines) + '_' + str(lines) + '.png',
                        bbox_inches='tight', dpi=169.0)
            p.close("all")
            break

        p.close("all")
    return


def mesh_symmetry_check(
        m=np.zeros((33 * 51 + 1)),
        nFS=33,
        nL=51,
        label='EIM_beta000',
        cartesian=False,
        debug=False):
    """ checking symmetry of mesh dots
    Keyword Arguments:
        m {[type]} -- mesh array (default: {np.zeros((33 * 51 + 1))})
        nFS {int} -- number of fluxsurfaces (default: {33})
        nL {int} -- number of poloidal lines (default: {51})
        label {str} -- vmec/geom label (default: {'EIM_beta000'})
        debug {bool} -- debuggingbool (default: {False})
    """
    def plot_mesh_diff(
            axis=None,
            color=np.zeros((4)),
            top_polys=[None],
            bottom_polys=[None]):
        for p1, p2 in zip(top_polys, bottom_polys):
            axis.plot(  # mesh
                [(p1[0] + p2[0]) / 2.] * 2,
                [(p1[1] + p2[1]) / 2.] * 2,
                c=color, marker='^', alpha=0.5)
        return

    C = cm.brg(np.linspace(.0, 1., nFS + 1))
    fig, ax = p.subplots(1, 1)
    for S in range(nFS):
        for L in range(nL):

            if L >= nL / 2.:
                break

            if L == 0:
                if not cartesian:
                    top_polys = fgp.poly_from_mesh(
                        m=m, S=S, L=nL - 1, nL=nL, nFS=nFS)[:]
                    bottom_polys = np.flipud(fgp.poly_from_mesh(
                        m=m, S=S, L=nL - 2, nL=nL, nFS=nFS))[:]

                elif cartesian:
                    top_polys = fgp.square_from_mesh(
                        m=m, S=S, L=nL - 1 - L, nL=nL, nFS=nFS)
                    bottom_polys = np.flipud(fgp.square_from_mesh(
                        m=m, S=S, L=L, nL=nL, nFS=nFS))

                if False:
                    ax.plot(  # mesh
                        [p[0] for p in top_polys],
                        [p[1] for p in top_polys],
                        c='r', alpha=0.5, lw=0.25, marker='^', markersize=2.)
                    ax.plot(  # mesh
                        [p[0] for p in bottom_polys],
                        [p[1] for p in bottom_polys],
                        c='b', alpha=0.5, lw=0.25, marker='^', markersize=2.)

                plot_mesh_diff(
                    axis=ax, color=C[S],
                    top_polys=top_polys, bottom_polys=bottom_polys)

            if (L < int((nL - 1) / 2.)):
                if not cartesian:
                    top_polys = fgp.poly_from_mesh(
                        m=m, S=S, L=nL - 1, nL=nL, nFS=nFS)[:]
                    bottom_polys = np.flipud(fgp.poly_from_mesh(
                        m=m, S=S, L=nL - 2, nL=nL, nFS=nFS))[:]

                elif cartesian:
                    top_polys = fgp.square_from_mesh(
                        m=m, S=S, L=nL - L - 1, nL=nL - 1, nFS=nFS)[:]
                    bottom_polys = np.flipud(fgp.square_from_mesh(
                        m=m, S=S, L=L, nL=nL - 1, nFS=nFS))[:]

                if False:
                    ax.plot(  # mesh
                        [p[0] for p in top_polys],
                        [p[1] for p in top_polys],
                        c='r', alpha=0.5, lw=0.25, marker='^', markersize=2.)
                    ax.plot(  # mesh
                        [p[0] for p in bottom_polys],
                        [p[1] for p in bottom_polys],
                        c='b', alpha=0.5, lw=0.25, marker='^', markersize=2.)

                plot_mesh_diff(
                    axis=ax, color=C[S],
                    top_polys=top_polys, bottom_polys=bottom_polys)

    ax.set_ylabel('Z [m]')
    ax.set_xlabel('R [m]')
    pf.fig_current_save('mesh_symmetry', fig)
    fig.savefig('../results/INVERSION/PROFILE/' + label +
                '/mesh_symmetry.png', bbox_inches='tight', dpi=169.0)
    p.close('all')
    return
