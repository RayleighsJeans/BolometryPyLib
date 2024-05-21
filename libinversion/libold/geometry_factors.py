""" **************************************************************************
    so header """

import os
import json
import sys

import numpy as np
from scipy import stats
from scipy.spatial import ConvexHull as CxH

import mClass
import factors_geometry_plots as fgp
import profile_to_lint as ptl

Z = np.zeros
ones = np.ones

""" end of header
************************************************************************** """


def geometrical_and_emissivity_factors(
        cam='HBCm',
        N=4,
        lines=45,
        mesh={'none': None},
        LoS={'none': None},
        extrapolate=5,
        corners={'none': None},
        camera_info={'none': None},
        cams=['HBCm', 'VBCl', 'VBCr'],
        vmec_label='EIM_000',
        pre='../results/INVERSION/MESH/',
        cartesian=False,
        saving=False,
        overwrite=False,
        debug=False):
    """ super routine to find geomtrical factors in cell for every channel
    Args:
        mesh (0, dict): mesh grid in cartesian or fluxsurfaces
        linesofsight (1, dict): lines of sight of each channel
        corners (dict, required): dict of camera detector corners
        lines (2, int): number of horizontal/vertical/poloidal lines
        vmec_label (3, str): name of VMEC configuration
        extrapolate (4, int): number of extrapolated sufraces around
        cartesian (5, bool): whether mesh is fluxsurface geometry or metric
    Returns:
        factors_angles (0, dict): Results calculated to geom factors
    """
    print('\tCalculate geometrical factors')
    if not cartesian:
        file = '_' + vmec_label + '_' + \
            str(lines) + '_' + str(extrapolate)
    elif cartesian:
        file = '_' + vmec_label + '_' + \
            str(lines) + '_' + str(lines)

    if saving:
        if os.path.isfile(pre + 'TRIAGS/geom_factors' + file + '.json'):
            return (store_read_geom_factors(name=file))
        else:
            pass

    factors_angles = {
        'label': 'geometrical factors, emissivity and angles of all cameras',
        'values': {
            'd': np.zeros((128, N**2)),
            'angles': {
                'alpha': np.zeros((128, N**2)),
                'beta': np.zeros((128, N**2))},
            'areas': {
                'A_s': {
                    'HBCm': np.zeros((N)),
                    'VBCr': np.zeros((N)),
                    'VBCl': np.zeros((N)),
                    'ARTf': np.zeros((N))},
                'A_d': np.zeros((128, N))},
            'normals': np.zeros((128, N, 3)),
            'normal_slit': {
                'HBCm': np.zeros((N, 3)),
                'VBCr': np.zeros((N, 3)),
                'VBCl': np.zeros((N, 3)),
                'ARTf': np.zeros((N, 3))}}}

    nFS = int(np.shape(mesh['values'][
        'crosspoints']['HBCm'])[1] / lines)
    emiss = np.zeros((128, N**2, nFS, lines))
    geom = np.zeros((128, N**2, nFS, lines))

    # advance the dictionary tree here for appropriate storage
    fangl = factors_angles['values']
    angles = fangl['angles']
    areas = fangl['areas']
    normals = fangl['normals']

    cv = corners['values']
    ca = corners['aperture']

    # dive into the subroutine which calculates the
    # geometric factors based on the given mesh and camera
    line_sec, LoS_geometry, polygons, cell_phi = GetGeometricalFactors(
            cams=cams, N=N, aperture=corners['aperture'], mesh=mesh,
            LoS=LoS, lines=lines, camera_info=camera_info, diagnose=False,
            cartesian=cartesian, label=vmec_label)

    # for the geometrical factors one also needs the angles between
    # lines of sight and the surface normals of each detector
    # and slit, plus their respective areas
    for c, cam in enumerate(cams):
        slit = np.array([  # m, 3 dims, 4 points (center, corners) N triags
            ca[cam]['x'], ca[cam]['y'], ca[cam]['z']])
        for p in range(N):  # all triangles
            areas['A_s'][cam][p] = calculate_area(  # m^2
                c=slit[:, 1:, p])  # ith triag

            if debug:
                v1 = slit[:, 2, p] - slit[:, 1, p]
                v2 = slit[:, 4, p] - slit[:, 1, p]
                
                d1, d2 = np.sqrt(np.dot(v1, v1)), np.sqrt(np.dot(v2, v2))
                aB = d1 * d2
                print(cam, p, 'cxHull', areas['A_s'][cam][p] * 1000.**2,
                      'a,b:', format(d1 * 1000., '.3f'), 'm',
                      format(d2 * 1000., '.3f'), 'm',
                      format(aB * 1000.**2, '.3f'), 'mm^2')

            fangl['normal_slit'][cam][p] = get_normal(
                rectangle=slit[:, :, p])  # in m
            if debug:
                print(fangl['normal_slit'][cam][p])

        for n, ch in enumerate(
                camera_info['channels']['eChannels'][cam][:]):
            det = np.array([cv['x'][ch], cv['y'][ch], cv['z'][ch]])  # in m

            for d in range(N):  # all triangles
                # again areas
                areas['A_d'][ch, d] = calculate_area(
                    c=det[:, d, 1:])  # in m²
                if debug:
                    print(cam, ch, d, 'A', areas['A_d'][ch, d])

                normals[ch, d] = get_normal(rectangle=det[:, d, :])  # in m
                if debug:
                    print('D normal', normals[ch, d])
                    print('D', np.shape(det[:, d, :]), det[:, d, :])

            for p in range(N):  # slit
                for d in range(N):  # detector
                    fangl['d'][ch, p * N + d] = np.sqrt(np.square(  # in m
                        det[0, d, 0] - slit[0, 0, p]) + np.square(
                        det[1, d, 0] - slit[1, 0, p]) + np.square(
                        det[2, d, 0] - slit[2, 0, p]))
                    if debug:
                        print(cam, ch, d, p, 'd', fangl['d'][ch, d * N + p])

                    # get them normals and angles
                    if debug:
                        print('L', np.shape(slit[:, :, p]), slit[:, :, p])
                    angles['alpha'][ch, p * N + d], \
                        angles['beta'][ch, p * N + d] = get_angles(  # deg, in m
                            b=slit[:, :, p], a=det[:, d, :],  # in m
                            a_normal=normals[ch, d],
                            b_normal=fangl['normal_slit'][cam][p])  # in m

                    if debug:
                        print(cam, ch, d, p,
                              'alpha', format(
                                  angles['alpha'][ch, p * N + d], '.3f'),
                              'beta', format(
                                  angles['beta'][ch, p * N + d], '.3f'),
                              'area_d', format(
                                  areas['A_d'][ch, d] * 1.e6, '.3f'),
                              'sum_d', format(np.sum(
                                  areas['A_d'][ch, :]) * 1.e6, '.3f'))

        # total slit area
        tot_As = np.sum(areas['A_s'][cam])

        # calculate the final geometrical factor as follows
        # units are m * cos(rad) * cos(rad) * m^2 * m^2 / m^2 = m^3
        for n, ch in enumerate(
                camera_info['channels']['eChannels'][cam][:]):

            # total detector area
            tot_Ad = np.sum(areas['A_d'][ch])
            if debug:
                print(c, cam, n, ch, total_Ad, total_As)

            for p in range(N):
                for d in range(N):
                    i = p * N + d

                    # area ratios so that it keeps consistent
                    f_Ad = areas['A_d'][ch, d] / tot_Ad
                    f_As = areas['A_s'][cam][p] / tot_As

                    R_Ad = f_Ad * areas['A_s'][cam][p]
                    R_As = f_As * areas['A_d'][ch, d]
                    areaF = R_Ad * R_As

                    # angles between normals
                    alpha = np.abs(np.cos(np.deg2rad(
                        angles['alpha'][ch, i])))
                    beta = np.abs(np.cos(np.deg2rad(
                        angles['beta'][ch, i])))
                    delta = alpha * beta

                    # angle out of plane from LoS
                    gamma = 1. / np.cos(corners['angles'][c, i])
                    line_sec[ch, i] *= gamma

                    emiss[ch, i] = delta * areaF / (
                        4 * np.pi * fangl['d'][ch, i]**2) * line_sec[ch, i]

                    geom[ch, i] = f_Ad * f_As * LoS_geometry[3, ch, i]

    if saving:
        store_read_geom_factors(
            name=file, factors_angles=factors_angles, emissivity=emiss,
            geometry=geom, line_sections=line_sec, polygons=polygons,
            cell_phi=cell_phi, LoS_geometry=LoS_geometry, overwrite=overwrite)

    if True:
        fgp.FS_channel_property(
            property_tag='FS_entry_area', location='LIN_SEG/',
            property=LoS_geometry[0], LoS_r=None, LoS_z=None,
            line_seg=line_sec, nFS=nFS, nL=lines, N=N, show=True,
            cam='VBCl', mesh=mesh['values']['crosspoints']['108.'],
            channels=camera_info['channels']['eChannels']['VBCl'])

        fgp.FS_channel_property(
            property_tag='FS_geomf_cell', location='GEOMF/',
            property=geom * 100.**3, LoS_r=None, LoS_z=None,
            line_seg=line_sec, nFS=nFS, nL=lines, N=N, show=True,
            cam='VBCl', mesh=mesh['values']['crosspoints']['108.'],
            channels=camera_info['channels']['eChannels']['VBCl'])
    
        volume, kbolo = ptl.volume_in_plasma_tor(
            nFS=nFS, nL=lines, N=N, camera_info=camera_info, cameras=['VBCl'],
            emissivity=emiss, volume=geom, line_sections=line_sec)
        fgp.LOS_K_factor(
            camera_info=camera_info, label='test',
            extrapolate=nFS - 21, emiss=kbolo, volum=volume, show=True,
            daz_K=camera_info['geometry']['kbolo'], cameras=['VBCl'],
            daz_V=camera_info['geometry']['vbolo'])

    return (
        factors_angles, emiss, geom, line_sec,
        LoS_geometry, polygons, cell_phi)


def GetGeometricalFactors(
        camera='HBCm',
        N=4,
        aperture={'HCBm': {'x': None, 'y': None, 'z': None, 'r': None}},
        mesh={'none': None},
        cams=['HBCm', 'VBCl', 'VBCr'],
        camera_info={'none': None},
        LoS={'none': None},
        lines=50,
        diagnose=False,
        label='EIM_beta000',
        cartesian=False,
        debug=False):
    """ line partition inside each cell based on LOFS
    Args:
        mesh (0, dict): grid mesh based in cartesian metric or fluxsurfaces
        LoS (1, dict): lines of sight of one camera (e.g. channels)
        lines (2, int): number of horizontal/vertical/poloidal lines in mesh
        diag (3, bool): if routine debug prints
        cartesian (4, bool): whether the mesh is cartesian or not
        cam (6, str): name of camera
    Returns:
        geometrical_factors {ndarray}: Results of geomfactors calculations
        polys {ndarray}: Polygons where LOFS intersects
        cuts {ndarray}: points Polygons where LOFS intersects
    """
    print('\t\tFactors... find line cuts inside each mesh cell: ', end='')

    # being carefull where the last line should be, because the
    # next to last fluxsurface-line is the first one again
    if cartesian:
        nL = end = lines - 1
    elif not cartesian:
        nL = end = lines

    cell_phi = {}
    for c, cam in enumerate(['VBCl']):
        # prepare logic parameters
        [hit, hit2] = [False, False]
        M = mesh['values']['crosspoints'][cam]  # m
        nFS = np.int(len(M[0, :]) / lines)

        fs_phi = mesh['values']['fs_phi'][cam]  # deg
        cell_phi[cam] = fs_phi.reshape(nFS, -1)
        if c == 0:
            cell_phi['108.'] = mesh['values'][
                'fs_phi']['108.'].reshape(nFS, -1)

        if c == 0:
            polys = np.zeros((128, N**2, nFS, lines, 2, 2, 2))

            line_seg = np.zeros((128, N**2, nFS, lines))
            LoS_geom = np.zeros((4, 128, N**2, nFS, lines))

        for L in range(end):
            # percentag bar
            print(format((L + c * end) / (
                end * np.shape(cams)[0]) * 100, '.1f'),
                '%', end=' ')
            for S in range(nFS):
                if cartesian:
                    p1, p2, p3, p4 = fgp.square_from_mesh(
                        m=mesh['values']['crosspoints'][cam],
                        S=S, L=L, nL=nL, nFS=nFS)
                elif not cartesian:
                    p1, p2, p3, p4 = fgp.poly_from_mesh(
                        m=mesh['values']['crosspoints'][cam],
                        S=S, L=L, nL=nL, nFS=nFS)

                # define lines of rectangle
                poly_p1 = np.array([[p1[0], p2[0]], [p1[1], p2[1]]])
                poly_p2 = np.array([[p2[0], p3[0]], [p2[1], p3[1]]])
                poly_p3 = np.array([[p3[0], p4[0]], [p3[1], p4[1]]])
                poly_p4 = np.array([[p4[0], p1[0]], [p4[1], p1[1]]])

                # now geht the LOFS
                for n, ch in enumerate(camera_info[
                        'channels']['eChannels'][cam][:]):

                    if (L == 0) and (S == 0):
                        Xf, Yf, Zf, rR, rZ = \
                            LoS['xy_plane']['range'][ch], \
                            LoS['xy_plane']['line'][ch], \
                            LoS['xz_plane']['line'][ch], \
                            LoS['rz_plane']['range'][ch], \
                            LoS['rz_plane']['line'][ch]

                    for T in range(N**2):  # N x N splits of apt and det
                        length = np.sqrt(  # length of 3D lines
                            (Xf[T] - Xf[T, -1])**2 +
                            (Yf[T] - Yf[T, -1])**2 +
                            (Zf[T] - Zf[T, -1])**2)

                        if diagnose:
                            print('L:', L, 'FS:', S, 'ch:', ch, 'T:', T)
                        Z, R = \
                            np.array(LoS['rz_plane']['line'][ch, T]), \
                            np.array(LoS['rz_plane']['range'][ch, T])
                        line_ofs = np.array([R, Z])

                        # loop polygons see if LoS crosses them
                        polygons = [poly_p1, poly_p2, poly_p3, poly_p4]  # in m
                        for i, poly in enumerate(polygons):
                            hit, CMP, CMV = \
                                have_intersection(  # in m
                                    arr1=poly, arr2=line_ofs,  # in m
                                    cartesian=cartesian,
                                    debug=False, diagnose=False)

                            if hit:  # hit, loop from poly_pX and check exit
                                for j, poly2 in enumerate(polygons[i + 1:]):
                                    hit2, CSP, CSV = \
                                        have_intersection(  # in m
                                            arr1=poly2, arr2=line_ofs,  # in m
                                            cartesian=cartesian,
                                            debug=False, diagnose=False)

                                    if hit2:
                                        polys[ch, T, S, L, 0, :, :] = \
                                            poly[:]  # in m
                                        polys[ch, T, S, L, 1, :, :] = \
                                            poly2[:]  # m

                                        distance = np.sqrt(np.dot(  # in m
                                            [CSP - CMP, CSV - CMV],
                                            [CSP - CMP, CSV - CMV]))
                                        # 1d, m
                                        line_seg[ch, T, S, L] += distance

                                        V, A1, A2, h = get_spread(
                                            L=length, X=Xf, Y=Yf, Z=Zf,
                                            rZ=rZ, rR=rR, T=T,
                                            x01=[CMP, CMV], x02=[CSP, CSV],
                                            debug=False)

                                        # pyramidstump areas, height, volume
                                        LoS_geom[0, ch, T, S, L] += A1  # entry
                                        LoS_geom[1, ch, T, S, L] += A2  # exit
                                        LoS_geom[2, ch, T, S, L] += h  # height
                                        LoS_geom[3, ch, T, S, L] += V  # base

                                        if debug:
                                            print('S', S, 'L', L, 'ch', ch,
                                                  'T', T, 'dist:', distance,
                                                  'h:', h, 'V:', V,
                                                  'A1:', A1, 'A2:', A2)

                                        # second hit found, break the loop
                                        break
                                break
                        [hit, hit2] = [False, False]
    print('... finished!')
    return (line_seg, LoS_geom, polys, cell_phi)


def get_spread(
        T=0,
        L=np.zeros((1000)),  # 3D, length, m
        X=np.zeros((64, 1000)),  # in m
        Y=np.zeros((64, 1000)),  # in m
        Z=np.zeros((64, 1000)),  # in m
        R=np.zeros((64, 1000)),  # in m
        rZ=np.zeros((64, 1000)),  # in m
        rR=np.zeros((64, 1000)),  # in m
        x01=[0.0, 0.0],
        x02=[0.0, 0.0],
        debug=False):
    # n sided pyramid stump has the volume
    # V = 1/3 * height * (surface1 * sqrt(surface1 * surface2) * surface2)

    # find where intersect with cone
    ind1 = int(np.mean([
        mClass.find_nearest(rR[T], x01[0])[0],
        mClass.find_nearest(rZ[T], x01[1])[0]]))
    ind2 = int(np.mean([
        mClass.find_nearest(rR[T], x02[0])[0],
        mClass.find_nearest(rZ[T], x02[1])[0]]))

    D1 = np.sqrt(  # distance entry
        (rR[T][ind1] - rR[T][-1])**2 + (rZ[T][ind1] - rZ[T][-1])**2)
    D2 = np.sqrt(  # distance exit
        (rR[T][ind2] - rR[T][-1])**2 + (rZ[T][ind2] - rZ[T][-1])**2)

    indx_1 = mClass.find_nearest(L, D1)[0]  # match enttry
    indx_2 = mClass.find_nearest(L, D2)[0]  # match exit
    x1, y1, z1 = X[:, indx_1], Y[:, indx_1], Z[:, indx_1]
    x2, y2, z2 = X[:, indx_2], Y[:, indx_2], Z[:, indx_2]

    # entry and and exit pyramid section
    A1 = calculate_area(np.array([x1, y1, z1]))
    A2 = calculate_area(np.array([x2, y2, z2]))

    dX = np.array([  # section of pyramid
        x1[0] - x2[0], y1[0] - y2[0], z1[0] - z2[0]])
    h = np.sqrt(np.dot(dX, dX))  # height of pyramid

    V = 1. / 3. * h * (A1 + np.sqrt(A1 * A2) + A2)
    if (h == 0) or (A1 == 0) or (A2 == 0) or (V == 0):
        h, A1, A2, V = .0, .0, .0, .0

    if debug:
        print('ind1:', ind1, 'ind2:', ind2,
              'indx1:', indx_1, 'indx2:', indx_2,
              'h:', format(h, '.2e'),
              'V:', format(V, '.2e'),
              'A1:', format(A1, '.2e'),
              'A2:', format(A2, '.2e'))

    return (V, A1, A2, h)


def have_intersection(
        arr1=Z((2, 2)),  # in m
        arr2=Z((2, 2)),  # in m
        debug=False,
        diagnose=False,
        cartesian=False):
    """ intersection finding of two (ideally) linear arrays
    Args:
        arr1 (0, ndarray): one of the lines to search the intersection on
        arr2 (1, ndarray): other one of the lines to search intersections on
        diag (2, bool): print debug
        cartesian (3, bool): is the mesh cartesian?
    Returns:
        results (0, list): 0-if intersect, 1-x point, 2-y point
    """
    if debug:
        print('x1=[', format(np.min(arr1[0]), '.3f'),
              format(np.max(arr1[0]), '.3f'), ']',
              'x2=[', format(np.min(arr2[0]), '.3f'),
              format(np.max(arr2[0]), '.3f'), ']',
              'y1=[', format(np.min(arr1[1]), '.3f'),
              format(np.max(arr1[1]), '.3f'), ']',
              'y2=[', format(np.min(arr2[1]), '.3f'),
              format(np.max(arr2[1]), '.3f'), ']')

    if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
        if diagnose:
            print('no array(s) given', end=' ')
        return (False, None, None)

    elif np.shape(arr1)[0] != 2 or np.shape(arr2)[0] != 2:
        if diagnose:
            print('wrong array(s) given', end=' ')
        return (False, None, None)

    # linear regression approach with
    # f1_arr1 = a2 * x + b2 in m
    # f2_arr2 = a2 * x + b2 in m
    # f1_arr1 == f2_arr2
    # yields x_0 = (b2 - b1) / (a1 - a2) in m
    try:
        [x_0, y_0, y_02, a1, a2, b1, b2] = [0.0] * 7
        if not cartesian:
            a1, b1 = stats.linregress(arr1[0], arr1[1])[:2]  # in unitless, m
        a2, b2 = stats.linregress(arr2[0], arr2[1])[:2]  # in unitless, m

    except Exception:
        if debug:
            print('linregress failed', end=' ')
        return (False, None, None)

    if not cartesian:
        x_0 = (b2 - b1) / (a1 - a2)  # in m
        y_0 = a1 * x_0 + b1  # in m
        y_02 = a2 * x_0 + b2  # in m

    elif cartesian:
        if arr1[0, 0] == arr1[0, 1]:  # vertical
            if diagnose:
                print('vertical', end=' ')
            x_0 = arr1[0, 0]  # in m
            y_0 = y_02 = a2 * x_0 + b2  # in m

        elif arr1[1, 0] == arr1[1, 1]:  # horizontal
            if diagnose:
                print('horizontal', end=' ')
            x_0 = (arr1[1, 0] - b2) / a2  # in m
            y_0 = y_02 = arr1[1, 0]  # in m

    if diagnose:
        print('x_0:', format(x_0, '.4f'), 'y_0:', format(y_0, '.4f'),
              'y_02:', format(y_02, '.4f'), end=' ')

    # two line always intersect, unless parallel, hence check and leave
    if (y_0 - 1e8 <= y_02 <= y_0 + 1e8) and (
            np.min(arr1[0]) <= x_0 <= np.max(arr1[0])) and (
            np.min(arr2[0]) <= x_0 <= np.max(arr2[0])) and (
            np.min(arr1[1]) <= y_0 <= np.max(arr1[1])) and (
            np.min(arr2[1]) <= y_0 <= np.max(arr2[1])):
        if debug:
            print('inside array ranges', end=' ')
        return (True, x_0, y_0)  # in m

    else:
        if debug:
            print(y_0, y_02, y_0 - y_02, end='\t')
            print('not in ranges:', (y_0 - 1e8 <= y_02 <= y_0 + 1e8),
                  (np.min(arr1[0]) <= x_0 <= np.max(arr1[0])),
                  (np.min(arr2[0]) <= x_0 <= np.max(arr2[0])),
                  (np.min(arr1[1]) <= y_0 <= np.max(arr1[1])),
                  (np.min(arr2[1]) <= y_0 <= np.max(arr2[1])), end=' ')
        return (False, None, None)


def calculate_area(
        c=Z((3, 4))):  # 3 coords, 4 points, triangle in m
    """ calculate the area between the given points
    Keyword Arguments:
        c {ndarray} -- list rectangle of corners (default: {Z((3, 4, 4))})
    Returns:
            area {float}: given 3d corners, calculated area
    """
    b = np.asarray(c).transpose()
    hull = CxH(b, qhull_options='QJ')
    return (hull.area)


def get_normal(
        rectangle=Z((3, 3))):  # in m, 3 coords and 3 points
    """ calculating normal vector
    Keyword Arguments:
        rectangle {ndarray} -- 5 point rectangle (3D) (default: {Z((3, 5))})
    Returns:
        normal {ndarray} -- normal vector
        coeffs {ndarray} -- surface equation coeffecients
        u {ndarray} -- vectorts spanning the surface
        v {ndarray} -- vectors spanning the surface
    """
    # vectors of corners of detector
    u = np.array([rectangle[0, 1] - rectangle[0, 0],  # in m
                  rectangle[1, 1] - rectangle[1, 0],
                  rectangle[2, 1] - rectangle[2, 0]])
    v = np.array([rectangle[0, 2] - rectangle[0, 0],  # in m
                  rectangle[1, 2] - rectangle[1, 0],
                  rectangle[2, 2] - rectangle[2, 0]])
    return (np.cross(u, v))  # in m


def get_angles(
        a=Z((3, 4)),  # in m, 3 coords, 4 points
        b=Z((3, 4)),  # in m, 3 coords, 4 points
        a_normal=Z((3)),  # in m
        b_normal=Z((3))):  # in m
    """ angles between normals of detector slit and line of sight from camera
    Args:
        detector (ndarray): array of corners from detector
        slit (ndarray): array of corner from slit
        detector_normal (ndarray): detector normal vector
        slit_normal (ndarray): slit normal vector
    Returns:
        alpha (float): angle one between normal and los vector
        beta (float): angle between normals
        lofs_vec (3d array): line of sight vector
    """
    lofs_vec = np.array([
        a[0, 0] - b[0, 0],
        a[1, 0] - b[1, 0],
        a[2, 0] - b[2, 0]])

    u, v, w = \
        np.linalg.norm(a_normal), \
        np.linalg.norm(b_normal), \
        np.linalg.norm(lofs_vec)

    alpha = np.rad2deg(np.arccos(np.dot(
        a_normal, lofs_vec) / (u * w)))
    beta = np.rad2deg(np.arccos(np.dot(
        b_normal, lofs_vec) / (v * w)))

    if np.isnan(alpha):
        alpha = 0.0
    if np.isnan(beta):
        beta = 0.0

    if alpha > 90.:
        alpha = np.abs(180. - alpha)
    if beta > 90.:
        beta = np.abs(180. - beta)

    return (alpha, beta)  # deg, deg, in m


def store_read_geom_factors(
        name='_EIM_beta000_new_tN2_slanted_HBCm_75_10',
        factors_angles=None,  # {'none': None},
        emissivity=None,  # np.zeros((128, 16, 31, 75)),
        geometry=None,  # np.zeros((128, 16, 31, 75)),
        line_sections=None,  # np.zeros((128, 16, 31, 75)),
        polygons=None,  # np.zeros((128, 16, 31, 75, 2, 2)),
        LoS_geometry=None,  # np.zeros((128, 16, 31, 75)),
        cell_phi=None,  # {np.zeros((128, 16, 31, 75, ))}
        overwrite=False):
    base = '../results/INVERSION/MESH/TRIAGS/'
    data = []

    def load_store_prop(
            label, P):
        loc = base + label + name + '.npz'
        if os.path.isfile(loc) and not overwrite:
            P = np.load(loc)['arr_0']
            print('\t\t\\\ load ' + label.replace('_', ' '), np.shape(P),
                  format(sys.getsizeof(P) / (1024. * 1024.), '.2f') + 'MB')
        else:
            if P is not None:
                if os.path.isfile(loc):
                    print('\t\t\\\ overwriting ' + label.replace('_', ' '),
                          np.shape(P),
                          format(sys.getsizeof(
                              P) / (1024. * 1024.), '.2f') + 'MB')
                else:
                    print('\t\t\\\ save ' + label.replace('_', ' '),
                          np.shape(P),
                          format(sys.getsizeof(
                              P) / (1024. * 1024.), '.2f') + 'MB')
                np.savez_compressed(loc, P)
            else:
                print('\t\t\\\ ' + label + ' is None')
        return (P)

    labels = [
        'emissivity', 'geometry', 'line_sections', 'LoS_geometry', 'polygons']
    properties = [
        emissivity, geometry, line_sections, LoS_geometry, polygons]

    for l, p in zip(labels, properties):
        data.append(load_store_prop(l, p))

    # factors and angles
    file = base + 'geom_factors' + name + '.json'
    if os.path.isfile(file) and not overwrite:
        with open(file, 'r') as infile:
            factors_angles = json.load(infile)
        infile.close()
        factors_angles = mClass.dict_transf(
            factors_angles, list_bool=False)
    else:
        if factors_angles is not None:
            if overwrite:
                print('\t\t\\\ overwrite factors angles',
                      format(sys.getsizeof(
                          factors_angles) / (1024. * 1024.), '.2f') + 'MB')
            else:
                print('\t\t\\\ save factors angles',
                      format(sys.getsizeof(
                          factors_angles) / (1024. * 1024.), '.2f') + 'MB')
            with open(file, 'w') as outfile:
                json.dump(mClass.dict_transf(
                    factors_angles, list_bool=True), outfile,
                    indent=4, sort_keys=False)
            outfile.close()
            factors_angles = mClass.dict_transf(
                factors_angles, list_bool=False)
        else:
            print('\t\t\\\ factors angles is None')

    # cell phi
    file = base + 'cell_phi' + name + '.json'
    if os.path.isfile(file) and not overwrite:
        with open(file, 'r') as infile:
            cell_phi = json.load(infile)
        infile.close()
        cell_phi = mClass.dict_transf(
            cell_phi, list_bool=False)
    else:
        if cell_phi is not None:
            if overwrite:
                print('\t\t\\\ overwrite cell_phi',
                      format(sys.getsizeof(
                          cell_phi) / (1024. * 1024.), '.2f') + 'MB')
            else:
                print('\t\t\\\ save cell_phi',
                      format(sys.getsizeof(
                          cell_phi) / (1024. * 1024.), '.2f') + 'MB')
            with open(file, 'w') as outfile:
                json.dump(mClass.dict_transf(
                    cell_phi, list_bool=True), outfile,
                    indent=4, sort_keys=False)
            outfile.close()
            cell_phi = mClass.dict_transf(
                cell_phi, list_bool=False)
        else:
            print('\t\t\\\ cell_phi is None')

    return (
        factors_angles,
        data[0],  # emiss
        data[1],  # geom
        data[2],  # line sec
        data[3],  # LoS geom
        data[4],  # poly
        cell_phi)  # cell phi
