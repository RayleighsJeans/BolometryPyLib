""" **************************************************************************
    start of file """

import numpy as np

import matplotlib.pyplot as p
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import axes3d

import plot_funcs as pf
import LoS_emissivity3D as LoS3D
import camera_geometries as cgeom
import factors_geometry_plots as fgp

Z = np.zeros
ones = np.ones
line = np.linspace

""" eo header
************************************************************************** """


def simple_lines_of_sight(
        lines_of_sight={'none': None},
        geometry={'none': None},
        corners={'none': None},
        camera_info={'none': None},
        cameras=['HBCm', 'VBCr', 'VBCl'],
        channel_selection=[],
        projection='2d',
        label='EIM_beta000',
        injection_mode=False,
        injAxis=None,
        alpha=20.,
        beta=60.,
        debug=False):
    """ Fluxsurface and lines of sight plot easy
    Args:
        angles (0, list, optional): List of toroidal angles
        fluxsurfaces (1, list, optional): ID for fluxsurfaces to be plotted
        lines_of_sight (2, dict, optional): Easy lines of sight geometry
        geometry (3, dict, optional): Camera full geometry
        corners (4, dict, optional): camera detector corner geometry
        camera_info (5, dict, optional): basic camera information
        cameras (6, list, optional): Cameras
        channel_selection (7, list, optional): detailed channels plotted
        projection (8, str, optional): dimensionality
        injection_mode (9, bool, optional): If axis is given to plot to
        injAxis (10, matplotlib.axes, optional): Axes to plot to
        alpha (11, float, optional): 3d viewing angle 1
        beta (12, float, optional): 3d viewing angle 2
        debug (13, bool, optional): debugging bool
    Returns:
        ax (0, matplotlib.axes, optional): Axes plotted to
    """
    print('\tPlotting lines of sight...')

    reff = camera_info['radius']['reff']
    L = lines_of_sight['values']
    apt = corners['aperture']
    N = np.shape(apt['HBCm']['x'])[1]

    c = cm.brg(np.linspace(0, 1, np.shape(cameras)[0]))
    for i, cam in enumerate(cameras[:]):

        if not injection_mode:
            fig = p.figure()
            if projection == '2d':
                ax = fig.add_subplot(111)
            elif projection == '3d':
                ax = fig.gca(projection='3d')
        else:
            projection = '2d'
            ax = injAxis

        if projection == '3d':
            ax.plot(
                [apt[cam]['x'][0, 0], apt[cam]['x'][0, 0]],
                [apt[cam]['y'][0, 0], apt[cam]['y'][0, 0]],
                [apt[cam]['z'][0, 0], apt[cam]['z'][0, 0]],
                c='k', marker='+')

        elif projection == '2d':
            ax.plot(
                [apt[cam]['r'][0, 0], apt[cam]['r'][0, 0]],
                [apt[cam]['z'][0, 0], apt[cam]['z'][0, 0]],
                c='k', marker='+')

        if projection == '3d':
            for k, ch in enumerate(camera_info[
                    'channels']['eChannels'][cam][:]):
                for a in range(N):
                    for d in range(N):
                        if debug:
                            print(i, cam, k, ch, a, d)
                        if k == 0 and d == 0 and a == 0:
                            ax.plot(
                                L['xy_plane']['range'][ch, a * N + d],
                                L['xy_plane']['line'][ch, a * N + d],
                                L['xz_plane']['line'][ch, a * N + d],
                                color=c[i], linewidth=0.5,
                                label=cam, alpha=0.6)
                        else:
                            ax.plot(
                                L['xy_plane']['range'][ch, a * N + d],
                                L['xy_plane']['line'][ch, a * N + d],
                                L['xz_plane']['line'][ch, a * N + d],
                                color=c[i], linewidth=0.5, alpha=0.6)
                        if (a == 0) and (d == 0):
                            ax.text(
                                L['xy_plane']['range'][ch, a * N + d][-1],
                                L['xy_plane']['line'][ch, a * N + d][-1],
                                L['xz_plane']['line'][ch, a * N + d][-1],
                                str(ch), color=c[i], rotation=.0, size=10.)

        elif projection == '2d':
            for k, ch in enumerate(camera_info[
                    'channels']['eChannels'][cam][:]):
                for a in range(N):
                    for d in range(N):
                        if debug:
                            print(i, cam, k, ch, a, d)

                        if k == 0 and a == 0 and d == 0:
                            ax.plot(
                                L['rz_plane']['range'][ch, a * N + d],
                                L['rz_plane']['line'][ch, a * N + d],
                                color=c[i], linewidth=0.5,
                                label=cam, alpha=0.6)
                        else:
                            ax.plot(
                                L['rz_plane']['range'][ch, a * N + d],
                                L['rz_plane']['line'][ch, a * N + d],
                                color=c[i], linewidth=0.5, alpha=0.6)
                        if (a == 0) and (d == 0):
                            ax.text(
                                L['rz_plane']['range'][ch, a * N + d][-1],
                                L['rz_plane']['line'][ch, a * N + d][-1],
                                str(ch), color=c[i], rotation=.0, size=10.)

            for k, ch in enumerate(channel_selection):
                for a in range(N):
                    for d in range(N):
                        if debug:
                            print(i, cam, k, ch, a, d)
                        if k == 0:
                            ax.plot(
                                L['rz_plane']['range'][ch, a * N + d],
                                L['rz_plane']['line'][ch, a * N + d],
                                alpha=.8, c=c[i],
                                linewidth=2., label='channel selection')
                        else:
                            ax.plot(
                                L['rz_plane']['range'][ch, a * N + d],
                                L['rz_plane']['line'][ch, a * N + d],
                                c=c[i], alpha=0.8, lw=2.0)
                        if (a == 0) and (d == 0):
                            ax.text(
                                L['rz_plane']['range'][ch, 0][0],
                                L['rz_plane']['line'][ch, 0][0],
                                str(ch) + ', r~' + str(format(reff[ch], '.2f')),
                                color=c[i], rotation=.0, size=14.)

        if projection == '3d':
            ax.view_init(alpha, beta)
            ax.set_xlabel('x / m')
            ax.set_ylabel('y / m')
            ax.set_zlabel('z / m')

        elif projection == '2d':
            ax.set_xlabel('r [m]')
            ax.set_ylabel('z [m]')

        ax.legend()
        ax.grid(b=True, which='major', linestyle='-.')

        if injection_mode:
            return ax

        ax.set_title("lines of sight with fluxsurfaces")
        pf.fig_current_save('LoS_FS_simple_' + projection, fig)
        fig.savefig(
            '../results/INVERSION/LOFS/LoS_FS_simple_' + projection +
            '_' + label + '_' + cam + '.png',
            bbox_inches='tight', dpi=169.0)

        # p.show()
        p.close("all")
    return


def LoS_symmetry_check(
        linesofsight={'none': None},
        camera_info={'none': None},
        label='EIM_beta000',
        d2=True,
        debug=False):

    los = linesofsight['values']
    rxy, lxy, lxz, rrz, lrz = \
        np.mean(los['xy_plane']['range'], axis=1), \
        np.mean(los['xy_plane']['line'], axis=1), \
        np.mean(los['xz_plane']['line'], axis=1), \
        np.mean(los['rz_plane']['range'], axis=1), \
        np.mean(los['rz_plane']['line'], axis=1)

    cs = cm.brg(np.linspace(0, 1, int(np.shape(
        camera_info['channels']['eChannels']['HBCm'])[0] / 2.) + 1))

    fig = p.figure()
    if d2:
        ax = fig.add_subplot(111)
    else:
        ax = fig.gca(projection='3d')

    for n, c in enumerate(camera_info[  # move all the detectors likewise
            'channels']['eChannels']['HBCm'][:16]):
        c1, c2 = c, 31 - n
        if False:
            print(n, c1, c2)

        if False:
            for ch in [c1, c2]:
                if d2:
                    ax.plot(
                        rrz[ch], lrz[ch],
                        c='grey', lw=0.5, alpha=0.25, ls='-.')
                else:
                    ax.plot(
                        rxy[ch], lxy[ch], lxz[ch],
                        c='grey', lw=0.5, alpha=0.25, ls='-.')

        if d2:
            ax.plot(
                [(v + rrz[c2][i]) / 2. for i, v in enumerate(rrz[c1])],
                [(v + lrz[c2][i]) / 2. for i, v in enumerate(lrz[c1])],
                c=cs[n], lw=0.75, alpha=0.75, ls='-.')
        else:
            ax.plot(
                [(v + rxy[c2][i]) / 2. for i, v in enumerate(rxy[c1])],
                [(v + lxy[c2][i]) / 2. for i, v in enumerate(lxy[c1])],
                [(v + lxz[c2][i]) / 2. for i, v in enumerate(lxz[c1])],
                c=cs[n], lw=0.75, alpha=0.75, ls='-.')

    fig.set_size_inches(7., 3.5)
    fig.savefig('../results/INVERSION/PROFILE/' + label +
                '/LoS_symmetry.png',
                bbox_inches='tight', dpi=169.0)
    p.close('all')
    return


def compare_viewcone(
        M=5000,
        N=4,
        nCh=[88, 89],
        det_corners=[0],
        LoS={'none': None},
        corners={'none': None},
        cam='VBCl',
        projection='3d',
        debug=False):

    L = LoS['values']
    apt = corners['aperture'][cam]
    cor = np.array([  # old geometry
        corners['values']['x'],
        corners['values']['y'],
        corners['values']['z'],
        corners['values']['r']])

    def get_plane_coeffs(v1, v2, origin):
        """
        Args:
            points (zeros((3, 3))): 3 points with 3 coords
        """
        # These two vectors are in the plane
        v1, v2 = v1 - origin[:3], v2 - origin[:3]
        # the cross product is a vector normal to the plane
        cp = np.cross(v1, v2)
        a, b, c = cp
        # This evaluates a * x3 + b * y3 + c * z3 which equals d
        d = np.dot(cp, v2)
        return (
            np.array([
                a, b, c, d]))

    def plane_intersect(a, b):
        """
        Args:
            a, b: 4-tuples/lists
                  Ax + By +Cz + D = 0
                  A,B,C,D in order
        Returns:
            2 points on line of intersection, np.arrays, shape (3,) """
        a_vec, b_vec = np.array(a[:3]), np.array(b[:3])
        aXb_vec = np.cross(a_vec, b_vec)

        A = np.array([a_vec, b_vec, aXb_vec])
        d = np.array([-a[3], -b[3], 0.]).reshape(3, 1)

        # could add np.linalg.det(A) == 0
        # test to prevent linalg.solve throwing error

        p_inter = np.linalg.solve(A, d).T
        return (
            p_inter[0],
            (p_inter + aXb_vec)[0])

    if cam == 'HBCm':
        target_x = -1.  # in m
    else:
        target_x = -2.0  # in m

    fig = p.figure()
    if projection == '2d':
        ax = fig.add_subplot(111)
    else:
        ax = fig.gca(projection='3d')
    colors = cm.brg(np.linspace(0, 1, np.shape(nCh)[0]))

    p03 = np.array([apt['x'][0], apt['y'][0], apt['z'][0]])
    p02 = np.array([
        np.sqrt(apt['x'][0]**2 + apt['y'][0]**2),
        apt['z'][0]])

    T = 2.0
    if projection == '2d':
        ax.plot(
            apt['r'][:], apt['z'][:],
            c='k', lw=0.5, alpha=0.3)

    for i, ch in enumerate(nCh):

        if projection == '2d':
            ax.plot(
                cor[3, ch, :], cor[2, ch, :],
                c='k', linewidth=0.5, alpha=0.3)

        else:
            ax.plot(
                cor[0, ch, :],
                cor[1, ch, :],
                cor[2, ch, :],
                c=colors[i], linewidth=0.5, alpha=0.3)
            ax.plot(
                apt['x'][:],
                apt['y'][:],
                apt['z'][:],
                c=colors[i], lw=0.5, alpha=0.3)

        for a in range(N):
            for d in range(N):

                # plot defined LoS from previous routines
                if projection == '2d':
                    length = np.sqrt(
                        np.square(L['rz_plane']['range'][ch, a * N + d] -
                              L['rz_plane']['range'][ch, a * N + d, -1]) +
                        np.square(L['rz_plane']['line'][ch, a * N + d] -
                              L['rz_plane']['line'][ch, a * N + d, -1]))
                elif projection == '3d':
                    length = np.sqrt(
                            (L['xy_plane']['range'][ch, a * N + d] -
                              L['xy_plane']['range'][ch, a * N + d, -1])**2 +
                            (L['xy_plane']['line'][ch, a * N + d] -
                             L['xy_plane']['line'][ch, a * N + d, -1])**2 +
                            (L['xz_plane']['line'][ch, a * N + d] -
                             L['xz_plane']['line'][ch, a * N + d, -1])**2)
                nn = np.where(length <= T)

                if projection == '2d':
                    ax.plot(
                        L['rz_plane']['range'][ch, a * N + d][nn], #  - p02[0],
                        L['rz_plane']['line'][ch, a * N + d][nn], # - p02[1],
                        color='k', linewidth=0.5, alpha=0.3)
                    if a == 0 and d == 0:
                        ax.text(
                            L['rz_plane']['range'][ch, a * N + d][nn][0],
                            L['rz_plane']['line'][ch, a * N + d][nn][0],
                            str(ch))
                else:
                    ax.plot(
                        L['xy_plane']['range'][ch, a * N + d][nn],
                        L['xy_plane']['line'][ch, a * N + d][nn],
                        L['xz_plane']['line'][ch, a * N + d][nn],
                        color='k', linewidth=0.5, alpha=0.3)

        # viewcone from center of detector through corners of pinhole
        for j, l in enumerate(det_corners):  # either 0 or all five
            for k in range(5):
                s_xy = (apt['y'][k] - cor[1, ch, l]) / (
                        apt['x'][k] - cor[0, ch, l])
                c_xy = apt['y'][k] - s_xy * apt['x'][k]
                s_xz = (apt['z'][k] - cor[2, ch, l]) / (
                        apt['x'][k] - cor[0, ch, l])
                c_xz = apt['z'][k] - s_xz * apt['x'][k]

                x = np.linspace(target_x, cor[0, ch, l], M)
                y = s_xy * x + c_xy
                z = s_xz * x + c_xz
                length = np.sqrt(
                    (x - x[-1])**2 + (y - y[-1])**2 + (z - z[-1])**2)
                nn = np.where(length <= T)

                los = cgeom.v_norm(np.array([
                    x[0] - x[-1], y[0] - y[-1], z[0] - z[-1]]))[0]

                if projection == '3d':
                    ax.plot(
                        x[nn], y[nn], z[nn],
                        c=colors[i], lw=0.5, alpha=0.3)

                slit = np.array([
                    apt['x'], apt['y'], apt['z']])
                slit_norm = -1. * cgeom.v_norm(LoS3D.get_normal(
                    rectangle=slit))[0]
                x0 = np.array([
                    apt['x'][0], apt['y'][0], apt['z'][0]])

                cam_plane_norm = -1. * cgeom.v_norm(LoS3D.get_normal(
                    rectangle=np.array([
                        [apt['x'][0], cor[0, nCh[-1], 0], cor[0, nCh[0], 0]],
                        [apt['y'][0], cor[1, nCh[-1], 0], cor[1, nCh[0], 0]],
                        [apt['z'][0], cor[2, nCh[-1], 0], cor[2, nCh[0], 0]]
                    ])))[0]

                if ((projection == '3d') and (j == 0) and
                        (i == 0) and (k == 0)):
                    ax.plot(
                        [apt['x'][0], apt['x'][0] + cam_plane_norm[0]],
                        [apt['y'][0], apt['y'][0] + cam_plane_norm[1]],
                        [apt['z'][0], apt['z'][0] + cam_plane_norm[2]],
                        # [0.0, cam_plane_norm[0]],
                        # [0.0, cam_plane_norm[1]],
                        # [0.0, cam_plane_norm[2]],
                        c='r', lw=0.5)

                    ax.plot(
                        [apt['x'][0], 0.5 * (apt['x'][0] + slit_norm[0])],
                        [apt['y'][0], 0.5 * (apt['y'][0] + slit_norm[1])],
                        [apt['z'][0], 0.5 * (apt['z'][0] + slit_norm[2])],
                        c='k', lw=0.5)

                a = get_plane_coeffs(
                    v1=los, v2=slit_norm, origin=x0)

                b = get_plane_coeffs(
                    v1=np.array([0., 1., 0.]),
                    v2=np.array([1., 0., 0.]),
                    origin=np.array([0., 0., 0.]))
                p1, p2 = plane_intersect(a, b)
                adj = cgeom.v_norm(p2 - p1)[0]

                if ((projection == '3d') and (j == 0) and
                        (i == 0) and (k == 0)):
                    ax.plot(
                        [apt['x'][0], apt['x'][0] + adj[0]],
                        [apt['y'][0], apt['y'][0] + adj[1]],
                        [apt['z'][0], apt['z'][0] + adj[2]],
                        # [0.0, adj[0]],
                        # [0.0, adj[1]],
                        # [0.0, adj[2]],
                        c='r', lw=0.5)

                if (projection == '3d'):
                    ax.plot(
                        x[nn],  # - p03[0],
                        y[nn],  # - p03[1],
                        z[nn],  # - p03[2],
                        c=colors[i], lw=0.5, alpha=0.3)

                if projection == '2d':
                    gamma = cgeom.vector_angle(
                        v1=adj, v2=los)

                    if (cam == 'HBCm') and (ch > 15):
                        gamma = np.pi - gamma
                    elif (cam == 'VBCl') and (np.rad2deg(gamma) < 60.):
                        gamma = np.pi - gamma
                    elif (cam == 'VBCr'):
                        gamma = np.pi - gamma
                    elif (cam == 'ARTf'):
                        continue

                    D = np.sqrt(
                        (x - p03[0])**2 +
                        (y - p03[1])**2 +
                        (z - p03[2])**2)
                    D = D[np.where(D <= T)]

                    r_new = D * np.cos(gamma) + p02[0]
                    z_new = D * np.sin(gamma) + p02[1]

                    s_rz = (z_new[0] - z_new[-1]) / (r_new[0] - r_new[-1])
                    c_rz = z_new[0] - s_rz * r_new[0]

                    if cam == 'HBCm':
                        target_r = 4.

                    elif cam == 'ARTf':
                        if apt['r'][0] > cor[3, ch, 0]:
                            target_r = 6.75
                        else:
                            if s_rz > 0:
                                target_r = 4.  # in m
                            elif s_rz < 0:
                                target_r = 4.  # in m

                    else:
                        if s_rz > .0:
                            target_r = 6.75  # in m
                        elif s_rz < .0:
                            target_r = 4.  # in m

                    r_new = np.array(line(target_r, apt['r'][0], M))
                    z_new = s_rz * r_new + c_rz

                    length = np.sqrt(
                        (r_new - r_new[-1])**2 + (z_new - z_new[-1])**2)
                    nn = np.where(length <= T)
                    ax.plot(
                        r_new[nn], z_new[nn],
                        # r_new[nn] - p02[0], z_new[nn] - p02[1],
                        color=colors[i], linewidth=0.5, alpha=0.3)
    p.show()
    return


def LoS_over_mesh(
        mesh=np.zeros((31 * 75, 2)),  # r, z
        LoS_r=np.zeros((128, 8 * 8, 2000)),  # ch, T
        LoS_z=np.zeros((128, 8 * 8, 2000)),  # ch, T
        channels=[int(ch) for ch in np.linspace(0, 31, 32)],
        nFS=31,
        nL=75,
        N=8,
        cam='HBCm',
        label='test',
        cartesian=False,
        debug=False):

    fig, ax = p.subplots(1, 1)
    for S in range(nFS):
        FS = []
        for L in range(nL):
            if cartesian:
                p1, p2, p3, p4 = fgp.square_from_mesh(
                    m=mesh, S=S, L=L, nL=nL, nFS=nFS)
            else:
                p1, p2, p3, p4 = fgp.poly_from_mesh(
                    m=mesh, S=S, L=L, nL=nL, nFS=nFS)

            FS.append(p1)
            FS.append(p4)

        poly = np.array([
            [p[0] for p in FS], [p[1] for p in FS]])
        ax.plot(
            poly[0], poly[1], c='k',
            alpha=0.5, lw=0.5, ls='-.')

    for i, ch in enumerate(channels):
        for T in range(N**2):
            if (cam == 'HBCm'):
                nn = np.where(
                    (LoS_r[ch, T] > np.min(mesh[0]) - 0.05) &
                    (LoS_z[ch, T] > np.min(mesh[1]) - 0.05) &
                    (LoS_z[ch, T] < np.max(mesh[1]) + 0.05) )
            elif (cam in ['VBCr', 'VBCl']):
                nn = np.where(
                    (LoS_z[ch, T] < np.max(mesh[1]) + 0.05) &
                    (LoS_r[ch, T] > np.min(mesh[0]) - 0.05) &
                    (LoS_r[ch, T] < np.max(mesh[0]) + 0.05))
            else:
                length = np.sqrt(
                    (LoS_r[ch, T] - LoS_r[ch, T][-1])**2 +
                    (LoS_z[ch, T] - LoS_z[ch, T][-1])**2)
                nn = np.where(length <= 1.5)

            ax.plot(
                LoS_r[ch, T][nn], LoS_z[ch, T][nn],
                c='grey', lw=0.25, alpha=0.5)

            if (T == 0):
                ax.text(
                    LoS_r[ch, T][nn][0], LoS_z[ch, T][nn][0],
                    str(ch) + '$_{' + str(i) + '}$', c='k')

    pf.fig_current_save('LoS_over_mesh_' + cam, fig)
    fig.savefig('../results/INVERSION/PROFILE/' + label +
                '/' + 'LoS_over_mesh_' + cam + '.png',
                bbox_inches='tight', dpi=169.0)
    p.close('all')
    return
