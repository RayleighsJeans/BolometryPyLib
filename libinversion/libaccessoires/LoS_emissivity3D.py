""" **************************************************************************
    start of file """

from os.path import isfile
import sys
from tqdm import tqdm
import json
import mClass

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import LineString, Polygon

""" eo header
************************************************************************** """


def voxel_from_FS(
        p, s, l, mesh):
    voxel = np.zeros((3, 8))  # coords, 8 points
    # this layer
    voxel[:, 0] = mesh[:3, p, s, l]  # OG point
    voxel[:, 1] = mesh[:3, p + 1, s, l]  # next slice
    voxel[:, 2] = mesh[:3, p + 1, s + 1, l]  # next slice, further out
    voxel[:, 3] = mesh[:3, p, s + 1, l]  # same slice, further out
    # upper layer
    voxel[:, 4] = mesh[:3, p, s + 1, l + 1]  # OG point, further out
    voxel[:, 5] = mesh[:3, p + 1, s + 1, l + 1]  # out, up, and forward
    voxel[:, 6] = mesh[:3, p + 1, s, l + 1]  # up, forward
    voxel[:, 7] = mesh[:3, p, s, l + 1]  # up, OG
    return (voxel.transpose())


def facets_of_voxel(
        voxel):  # 8 points, 3 coords
    triangles = ConvexHull(voxel).simplices
    return (triangles)


def plane_from_square(
        points):
    # These two vectors are in the plane
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp
    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, points[0])
    return (a, b, c, d)


def vectors_through_corners(
        N, aperture, detector_center):
    vectors = np.zeros((3, 2, N**2))
    for a in range(N):
        for d in range(N):
            vectors[:, 0, a * N + d] = np.array([
                detector_center[0, d],
                detector_center[1, d],
                detector_center[2, d]])
            vectors[:, 1, a * N + d] = np.array([
                aperture[0, a],
                aperture[1, a],
                aperture[2, a]])
    return (vectors)


def line3D(
        p1, p2):
    # xy plane
    a = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p2[1] - a * p2[0]
    # xz plamne
    c = (p2[2] - p1[2]) / (p2[0] - p1[0])
    d = p2[2] - c * p2[0]
    return (a, b, c, d)


def solve_x_LoS(
        cam, a, b, c, d, q, r, s, L, new_type='VBCm'):
    # solve L = sqrt((x - r)^2 + (a * x + b - s)^2 + (c * x + d - q)^2)
    solution = [
        (-1. * np.sqrt((2. * a * b - 2. * a * r + 2. * c * d -
         2. * c * s - 2. * q)**2 - 4. * (a**2 + c**2 + 1.) * (
         b**2 - 2. * b * r + d**2 - 2. * d * s - L**2 + q**2 +
         r**2 + s**2)) - 2. * a * b + 2. * a * r - 2. * c * d +
         2. * c * s + 2. * q) / (2. * (a**2 + c**2 + 1.)),
        (1. * np.sqrt((2. * a * b - 2. * a * r + 2. * c * d -
         2. * c * s - 2. * q)**2 - 4. * (a**2 + c**2 + 1.) * (
         b**2 - 2. * b * r + d**2 - 2. * d * s - L**2 + q**2 +
         r**2 + s**2)) - 2. * a * b + 2. * a * r - 2. * c * d +
         2. * c * s + 2. * q) / (2. * (a**2 + c**2 + 1.))]
    if (cam == 'HBCm'):
        return (solution[1])
    elif (cam == 'ARTf'):
        if (a > .0) and (new_type == 'MIRh'):
            return (solution[1])
        if (c < .0) and (new_type == 'VBCm'):
            return (solution[1])
    elif (((cam == 'VBCl') or (cam == 'VBCr')) and
            (c * solution[1] + d > q)):
        return (solution[1])
    return (solution[0])


def LoS_vec_from_points(
        cam, p1, p2, new_type='VBCm'):
    a, b, c, d = line3D(p1, p2)
    # define cut of LoS
    q, r, s = p2
    tx = solve_x_LoS(cam, a, b, c, d, q, r, s, 2.2, new_type)
    if (b < .0) and (cam == 'VBCl'):
        tx = p2[0] + np.abs(p2[0] - tx)
    # calculate endpoint
    pE = np.array([  # endpoint
        tx, a * tx + b, c * tx + d])
    return (pE - p2, p2, pE)


def sort_mesh3D(
        mesh, p1, p2):
    res = []
    lineXY = LineString([p1[[0, 1]], p2[[0, 1]]])
    lineXZ = LineString([p1[[0, 2]], p2[[0, 2]]])
    # find cells we need
    for P in range(np.shape(mesh)[1] - 1):
        for S in range(np.shape(mesh)[2] - 1):
            for L in range(np.shape(mesh)[3] - 1):
                x, y, z = \
                    mesh[0, P:P + 2, S:S + 2, L:L + 2].reshape(-1), \
                    mesh[1, P:P + 2, S:S + 2, L:L + 2].reshape(-1), \
                    mesh[2, P:P + 2, S:S + 2, L:L + 2].reshape(-1)
                polygonXY = Polygon(np.array([x, y]).transpose())
                if lineXY.intersects(polygonXY):
                    polygonXZ = Polygon(np.array([x, z]).transpose())
                    if lineXZ.intersects(polygonXZ):
                        res.append([P, S, L])
    return (np.array(res))


def barycenter(
        points):  # n points x 3 dims
    bc = np.zeros((np.shape(points)[1]))
    for k in range(np.shape(points)[1]):
        bc[k] = np.mean(points[:, k])
    return (bc)  # x, y, z


def s_factor(
        n, c, null, v):
    return (np.dot(n, c - null) / np.dot(n, v))


def face_hit(
        polygon, null, v):
    # norm and barycenter of triangle
    norm = plane_from_square(polygon)[:3]
    res, s = None, None
    # line, plane  not perpendicular
    s = s_factor(norm, polygon[0], null, v)
    if (s >= .0) and not (np.isnan(s)) and not (np.isinf(s)):
        res = null + s * v
    return (res)


def length_res(
        p1, p2):
    dX = np.array([
        p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
    return (np.sqrt(np.dot(dX, dX)))


def area_triangle(
        a, b, c):
    return (np.sqrt(np.dot(
        np.cross(b - a, c - a), np.cross(b - a, c - a))) / 2.)


def barycentric_method(
        triangle, point):
    T = area_triangle(triangle[0], triangle[1], triangle[2])
    # now with additional point, fractions
    alpha = area_triangle(point, triangle[1], triangle[2]) / T
    if (.0 <= alpha <= 1.):
        beta = area_triangle(point, triangle[2], triangle[0]) / T
        if (.0 <= beta <= 1.):
            gamma = area_triangle(point, triangle[0], triangle[1]) / T
            if (.0 <= gamma <= 1.):
                if (1. - 1e-15 <= alpha + beta + gamma <= 1. + 1e-15):
                    return (True)
    return (False)


def voxel_face_hit(
        voxel, null, v, indices):
    status = False
    results = np.zeros((2, 3))
    for f, find in enumerate(indices):
        P = face_hit(voxel[find], null, v)
        if (P is not None):
            if barycentric_method(voxel[find], P):
                results[0] = P
                for g, gind in enumerate(indices[f + 1:]):
                    P = face_hit(voxel[gind], null, v)
                    if (P is not None):
                        if barycentric_method(voxel[gind], P):
                            results[1] = P
                            status = True
                            break
                break
    return (status, results)


def get_lines3D(
        mesh=np.zeros((3, 100, 25, 125)),  # 3 coords, n phi, nFS, nL
        N=2,
        geometry={'none': None},
        cams=['HBCm'],
        camera_info={'none': None},
        vmec_label='test',
        new_type=None,
        saving=True,
        debug=False):

    nP, nFS, nL = np.shape(mesh)[1:]
    print('\tFinding line intersections in 3D [N² x phi x FS x L]:',
          N**2, 'x', nP, 'x', nFS - 1, 'x', nL - 1, '...')

    # check if existent
    if saving:
        line_sections = store_read_line_sections(name=vmec_label)
        if line_sections is not None:
            return (line_sections)

    # storage
    line_sections = np.zeros((128, N**2, nFS - 1, nL - 1))

    det_centers = np.array([
        geometry['values']['x'][:, :, 0],
        geometry['values']['y'][:, :, 0],
        geometry['values']['z'][:, :, 0]])

    # loop over cams, channels, LCFS's and points in FS2
    for cn, cam in enumerate(cams):

        apt_centers = np.array([
            geometry['aperture'][cam]['x'][0, :],
            geometry['aperture'][cam]['y'][0, :],
            geometry['aperture'][cam]['z'][0, :]])

        nCh = camera_info['channels']['eChannels'][cam]
        for dn, ch in enumerate(nCh):

            xyz = vectors_through_corners(
                N=N, aperture=apt_centers,
                detector_center=det_centers[:, ch])

            for T in range(N**2):
                # let the parametric equation of the line be
                # P(s) = P0 + s * (P1 - P0) = P0 + s * U
                u, p0, pE = LoS_vec_from_points(
                    cam=cam, p1=xyz[:, 0, T], p2=xyz[:, 1, T],
                    new_type=new_type)

                nRange = sort_mesh3D(mesh[:3], p0, pE)
                if (np.shape(nRange) == (0,)):  # no hits
                    continue

                for i in tqdm(range(np.shape(nRange)[0]),
                              desc=cam + ' ch#' + str(dn) + ' T' + str(T)):
                    P, S, L = nRange[i]

                    voxel = voxel_from_FS(
                        p=P, s=S, l=L, mesh=mesh)

                    # blindtest
                    # p0 = np.array([-0.35, -0.5, 1.1])
                    # pE = np.array([0.35, 1.1, -.1])
                    # u = pE - p0
                    # voxel = np.array([
                    #     [0.0, 0.0, 0.1],  # 0
                    #     [1.0, 0.0, 0.2],  # 1
                    #     [1.0, 1.0, 0.3],  # 2
                    #     [0.0, 1.0, 0.4],  # 3
                    #     [0.0, 1.0, 0.5],  # 4
                    #     [1.0, 1.0, 0.6],  # 5
                    #     [1.0, 0.0, 0.7],  # 6
                    #     [0.0, 0.0, 0.8]]) # 7

                    tris = facets_of_voxel(voxel)
                    status, results = voxel_face_hit(
                        voxel, p0, u, tris)
                    if status:
                        line_sections[ch, T, S, L] += \
                            length_res(results[0], results[1])

    if saving:
        store_read_line_sections(
            name=vmec_label, data=line_sections)
    return (line_sections)


def store_read_line_sections(
        data=None,  # np.zeros((128, 64, 25, 125)),
        name='test',
        base='../results/INVERSION/MESH/TRIAGS/',
        overwrite=False):

    def shape_size_str(
            data):
        return (
            str(np.shape(data)) + ' ' + format(
                sys.getsizeof(data) / (1024. * 1024.),
                '.2f') + 'MB')

    def load_store_prop(
            label='test',
            P=None):
        loc = base + label + name + '.npz'
        if isfile(loc) and not overwrite:
            P = np.load(loc)['arr_0']
            print('\t\t\\\ load ' + label.replace('_', ' '),
                  shape_size_str(P))

        elif isfile(loc) and overwrite and (P is not None):
            print('\t\t\\\ overwriting ' + label.replace('_', ' '),
                    shape_size_str(P))
            np.savez_compressed(loc, P)

        elif not isfile(loc) and (P is not None):
            print('\t\t\\\ save ' + label.replace('_', ' '),
                  shape_size_str(P))
            np.savez_compressed(loc, P)
        return (P)

    data = load_store_prop('line_sections3D_', data)
    return (data)  # line sections


def calculate_area(
        c):
    hull = ConvexHull(c.transpose(), qhull_options='QJ')
    return (hull.area / 2.)


def get_normal(
        rectangle):  # in m, 3 coords and 3 points
    # vectors of corners of detector
    u = np.array([rectangle[0, 1] - rectangle[0, 0],  # in m
                  rectangle[1, 1] - rectangle[1, 0],
                  rectangle[2, 1] - rectangle[2, 0]])
    v = np.array([rectangle[0, 2] - rectangle[0, 0],  # in m
                  rectangle[1, 2] - rectangle[1, 0],
                  rectangle[2, 2] - rectangle[2, 0]])
    return (np.cross(u, v))  # in m


def get_angles(
        a, b, a_normal, b_normal):  # in m
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


def length(
        p1, p2):
    return (np.sqrt(np.dot(p1 - p2, p1 - p2)))


def debug_empty_channels(
        data):
    zS = np.where(np.sum(data, axis=(1, 2, 3)) == 0.0)[0]
    return (zS)


def get_emissivity(
        lines=np.zeros((128, 16, 30, 125)),
        corners={'none': None},
        camera_info={'none': None},
        cams=['HBCm', 'VBCl', 'VBCr'],
        vmec_label='EIM_000',
        pre='../results/INVERSION/MESH/',
        input_shape='3D',
        saving=False,
        overwrite=False,
        debug=False):

    print('\tCalculating emissivity factors in ' + input_shape + ' ...')
    # check if existent
    if saving:
        factors, emissivity = store_read_emissivity(
            name=vmec_label, suffix=input_shape)

        if emissivity is not None:
            for c, cam in enumerate(cams):
                nCh = camera_info['channels']['eChannels'][cam]
                zS = debug_empty_channels(emissivity[nCh])
                if not (np.shape(zS) == (0,)):
                    empty = [nCh[i] for i in zS]
                    print(
                        '\t\t\\\ warning: channels with ' +
                        'zero emissivty:', empty)
            return (factors, emissivity)

    N, nFS, nL = np.shape(lines)[1:]
    emissivity = np.zeros((128, N, nFS, nL))
    N = int(np.sqrt(N))

    factors = {
        'label': 'emissivity factors',
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

    # corners of
    cv, ca = corners['values'], corners['aperture']

    # for the geometrical factors need angles
    # surface normals plus areas
    for c, cam in enumerate(cams):

        slit = np.array([  # 3 dims, 4 points, N triags
            ca[cam]['x'], ca[cam]['y'], ca[cam]['z']])

        for p in range(N):  # all triangles

            factors['values']['areas']['A_s'][cam][p] = calculate_area(
                c=slit[:, :, p])  # pth triag, m^2

            if debug:
                v1 = slit[:, 1, p] - slit[:, 0, p]
                v2 = slit[:, 3, p] - slit[:, 0, p]

                d1, d2 = np.sqrt(np.dot(v1, v1)), np.sqrt(np.dot(v2, v2))
                aB = d1 * d2
                print(cam, p, 'cxHull',
                      factors['values']['areas']['A_s'][cam][p] * 1000.**2,
                      'a,b:', format(d1 * 1000., '.3f'), 'm',
                      format(d2 * 1000., '.3f'), 'm',
                      format(aB * 1000.**2, '.3f'), 'mm^2')

            factors['values']['normal_slit'][cam][p] = get_normal(
                rectangle=slit[:, :, p])  # m, pth triangle

        for n, ch in enumerate(
                camera_info['channels']['eChannels'][cam]):
            det = np.array([
                cv['x'][ch], cv['y'][ch], cv['z'][ch]]).transpose(0, 2, 1)

            for d in range(N):  # triangles

                # again areas
                factors['values']['areas']['A_d'][ch, d] = calculate_area(
                    c=det[:, :, d])  # m^2, dth triangle

                factors['values']['normals'][ch, d] = get_normal(
                    rectangle=det[:, :, d])  # in m, dth triang

            for p in range(N):  # slit

                for d in range(N):  # detector
                    factors['values']['d'][ch, p * N + d] = \
                        length(det[:, 0, d], slit[:, 0, p])

                    # get them normals and angles
                    factors['values']['angles']['alpha'][ch, p * N + d], \
                        factors['values']['angles']['beta'][ch, p * N + d] = \
                        get_angles(
                            det[:, :, d], slit[:, :, p],
                            factors['values']['normals'][ch, d],
                            factors['values']['normal_slit'][cam][p])

                    if debug:
                        print(cam, ch, d, p,
                              'alpha', format(
                                  factors['values']['angles']['alpha'][
                                      ch, p * N + d], '.3f'),
                              'beta', format(
                                  factors['values']['angles']['beta'][
                                      ch, p * N + d], '.3f'),
                              'area_d', format(
                                  factors['values']['angles']['A_d'][
                                      ch, d] * 1.e6, '.3f'),
                              'sum_d', format(np.sum(
                                  factors['values']['angles']['A_d'][
                                      ch, :]) * 1.e6, '.3f'))

        # units are m * cos(rad) * cos(rad) * m^2 * m^2 / m^2 = m^3
        nCh = camera_info['channels']['eChannels'][cam]
        for n, ch in enumerate(nCh):

            for p in range(N):
                for d in range(N):
                    i = p * N + d
                    # area ratios so that it keeps consistent
                    R_Ad = factors['values']['areas']['A_s'][cam][p]
                    R_As = factors['values']['areas']['A_d'][ch, d]
                    areaF = R_Ad * R_As

                    # angles between normals
                    alpha = np.abs(np.cos(np.deg2rad(
                        factors['values']['angles']['alpha'][ch, i])))
                    beta = np.abs(np.cos(np.deg2rad(
                        factors['values']['angles']['beta'][ch, i])))
                    delta = alpha * beta

                    # out of plane length sin(angl) = line / act_line
                    # act_line = line / sin(angl)
                    gamma = 1. / np.cos(np.deg2rad(corners['angles'][ch, i]))

                    if debug:
                        print(
                            cam, n, i,
                            format(alpha, '.3f'),
                            format(beta, '.3f'),
                            format(R_As * 1e4, '.4f'),
                            format(R_As * 1e4, '.4f'),
                            format(factors['values']['d'][ch, i] * 1e2, '.3f'),
                            format(gamma, '.3f'),
                            format(1e6 * gamma * (delta * areaF / (
                                4 * np.pi * np.square(
                                    factors['values']['d'][ch, i]))), '.6f'),
                            format(np.sum(lines[ch], axis=(0, 1, 2)), '.6f'))

                    emissivity[ch, i] = (delta * areaF / (
                        4 * np.pi * np.square(factors['values']['d'][ch, i]))
                        ) * lines[ch, i]
                    if input_shape == '2D':
                        emissivity[ch, i] *= gamma

        zS = debug_empty_channels(emissivity[nCh])
        if not (np.shape(zS) == (0,)):
            empty = [nCh[i] for i in zS]
            print('\t\t\\\ warning: channels with zero emissivty:', empty)

    if saving:
        factors, emissivity = store_read_emissivity(
            name=vmec_label, suffix=input_shape,
            data=emissivity, factors=factors)
    return (factors, emissivity)


def store_read_emissivity(
        data=None,  # np.zeros((128, 64, 25, 125)),
        factors=None,  # {'none': None},
        name='test',
        suffix='3D',
        base='../results/INVERSION/MESH/TRIAGS/',
        overwrite=False):

    def shape_size_np(
            data):
        return (
            str(np.shape(data)) + ' ' + format(
                sys.getsizeof(data) / (1024. * 1024.),
                '.2f') + 'MB')

    def shape_size_dict(
            data):
        return (str(format(sys.getsizeof(
            data) / (1024. * 1024.), '.2f') + 'MB'))

    def load_store_prop(
            label='test',
            P=None):
        loc = base + label + name + '.npz'
        if isfile(loc) and not overwrite:
            P = np.load(loc)['arr_0']
            print('\t\t\\\ load ' + label.replace('_', ' '),
                  shape_size_np(P))

        elif isfile(loc) and overwrite and (P is not None):
            print('\t\t\\\ overwriting ' + label.replace('_', ' '),
                  shape_size_np(P))
            np.savez_compressed(loc, P)

        elif not isfile(loc) and (P is not None):
            print('\t\t\\\ save ' + label.replace('_', ' '),
                  shape_size_np(P))
            np.savez_compressed(loc, P)

        elif not isfile(loc) and (P is None):
            print('\t\t\\\ not found ' + label.replace('_', ' '))
        return (P)

    def load_store_dict(
            label='test',
            P=None):
        loc = base + label + name + '.json'
        if isfile(loc) and not overwrite:
            with open(loc, 'r') as infile:
                P = json.load(infile)
            infile.close()
            P = mClass.dict_transf(
                P, to_list=False)

        elif isfile(loc) and overwrite and (P is not None):
            print('\t\t\\\ overwriting ' + label.replace('_', ' '),
                  shape_size_dict(P))
            with open(loc, 'w') as outfile:
                json.dump(mClass.dict_transf(P, list_bool=True),
                          outfile, indent=4, sort_keys=False)
            outfile.close()
            P = mClass.dict_transf(P, list_bool=False)

        elif not isfile(loc) and (P is not None):
            print('\t\t\\\ save ' + label.replace('_', ' '),
                  shape_size_dict(P))
            with open(loc, 'w') as outfile:
                json.dump(mClass.dict_transf(P, to_list=True),
                          outfile, indent=4, sort_keys=False)
            outfile.close()
            P = mClass.dict_transf(P, to_list=False)
        return (P)

    data = load_store_prop('emissivity' + suffix + '_', data)
    factors = load_store_dict('geometry_factors_', factors)
    return (factors, data)  # line sections
