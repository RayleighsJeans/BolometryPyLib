""" **************************************************************************
    start of file """

from os.path import isfile
import sys
from tqdm import tqdm

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import LineString, Polygon

""" eo header
************************************************************************** """


def get_volume(
        vpF=1.3,
        mesh=np.zeros((3, 100, 25, 125)),  # 3 coords, n phi, nFS, nL
        corners={'none': None},
        old={'none': None},
        cams=['HBCm'],
        camera_info={'none': None},
        vmec_label='test',
        new_type=None,
        saving=False,
        debug=False):

    def square_from_FS(
            n, s, l, nL, FS):
        # OG point
        p1 = FS[:3, n, s, l]
        # same points, next slice
        p4 = FS[:3, n + 1, s, l]
        if l == nL - 1:
            l = 0  # cheat a little
        # same slice, next point
        p2 = FS[:3, n, s, l + 1]
        # next point, next slice
        p3 = FS[:3, n + 1, s, l + 1]
        return (np.array([p1, p2, p3, p4]))

    def sort_mesh3D(
            mesh, p1, p2):
        res = []
        lineXY = LineString([p1[[0, 1]], p2[[0, 1]]])
        lineXZ = LineString([p1[[0, 2]], p2[[0, 2]]])
        # find cells we need
        for P in range(np.shape(mesh)[1] - 1):
            for L in range(np.shape(mesh)[2] - 1):
                x, y, z = \
                    mesh[0, P:P + 2, L:L + 2].reshape(-1), \
                    mesh[1, P:P + 2, L:L + 2].reshape(-1), \
                    mesh[2, P:P + 2, L:L + 2].reshape(-1)
                polygonXY = Polygon(np.array([x, y]).transpose())
                if lineXY.intersects(polygonXY):
                    polygonXZ = Polygon(np.array([x, z]).transpose())
                    if lineXZ.intersects(polygonXZ):
                        res.append([P, L])
        return (np.array(res))

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
            aperture, detector):
        nK_p = np.shape(aperture)[1]
        nK_d = np.shape(detector)[1]
        vectors = np.zeros((3, 2, nK_p * nK_d))
        for k1 in range(nK_d):
            for k2 in range(nK_p):
                vectors[:, 0, k1 * nK_p + k2] = np.array([
                    detector[0, k1],
                    detector[1, k1],
                    detector[2, k1]])
                vectors[:, 1, k1 * nK_p + k2] = np.array([
                    aperture[0, k2],
                    aperture[1, k2],
                    aperture[2, k2]])
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
            cam, a, b, c, d, q, r, s, L):
        # solve L = sqrt((x - r)^2 + (a * x + b - s)^2 + (c * x + d - q)^2)
        solution = [
            (-1. * np.sqrt((2. * a * b - 2. * a * r + 2. * c * d -
            2. * c * s - 2. * q)**2 - 4. * (a**2 + c**2 + 1.) * (
            b**2 - 2. * b * r + d**2 - 2. * d * s - L**2 + q**2 +
            r**2 + s**2)) - 2. * a * b + 2. * a * r - 2. * c * d +
            2. * c * s + 2. *q) / (2. * (a**2 + c**2 + 1.)),
            (1. * np.sqrt((2. * a * b - 2. * a * r + 2. * c * d -
            2. * c * s - 2. * q)**2 - 4. * (a**2 + c**2 + 1.) * (
            b**2 - 2. * b * r + d**2 - 2. * d * s - L**2 + q**2 +
            r**2 + s**2)) - 2. * a * b + 2. * a * r - 2. * c * d +
            2. * c * s + 2. *q) / (2. * (a**2 + c**2 + 1.))]
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
            cam, p1, p2):
        a, b, c, d = line3D(p1, p2)
        # define cut of LoS
        q, r, s = p2
        tx = solve_x_LoS(cam, a, b, c, d, q, r, s, 2.2)
        # calculate endpoint
        pE = np.array([  # endpoint
            tx, a * tx + b, c * tx + d])
        p0 = p2  # first point
        return (pE - p0, p2, pE)

    def facets_of_polygon(
            polygon):  # 8 points, 3 coords
        triangles = ConvexHull(polygon).simplices
        return (triangles)

    def s_factor(
            n, c, null, v):
        return (np.dot(n, c - null) / np.dot(n, v))

    def face_hit(
            polygon,
            null,
            v):
        # norm and barycenter of triangle
        norm = plane_from_square(polygon)[:3]
        res, s = None, None
        # line, plane  not perpendicular
        s = s_factor(norm, polygon[0], null, v)
        if (s >= .0) and not (np.isnan(s)) and not (np.isinf(s)):
            res = null + s * v
        return (res)

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

    def polygon_face_hit(
            polygon, null, v, indices):
        for f, find in enumerate(indices):
            P = face_hit(polygon[find], null, v)
            if (P is not None):
                if barycentric_method(polygon[find], P):
                    return (True, P)
        return (False, None)

    def rescale_fs_number(
            vP, nFS, target):
        return (np.int(np.round(
            nFS * target / vP) - 1))

    def view_cone_volume(
            aperture, detector, ch, nL,
            nFS, camera, mesh, debug):

        nK_d = np.shape(detector)[1]
        nK_p = np.shape(aperture)[1]
        xyz = vectors_through_corners(
            aperture=aperture, detector=detector)

        results = []  # points here
        for korner in range(nK_d * nK_p):
            # let the parametric equation of the line be
            # P(s) = P0 + s * (P1 - P0) = P0 + s * U
            u, p0, pE = LoS_vec_from_points(
                cam=camera, p1=xyz[:, 0, korner],
                p2=xyz[:, 1, korner])

            nRange = sort_mesh3D(
                mesh[:3, :, nFS, :], p0, pE)
            if (np.shape(nRange) == (0,)):  # no hits
                continue

            for i in range(np.shape(nRange)[0]):
                P, L = nRange[i]

                polygon = square_from_FS(
                    P, nFS, L, nL, mesh)
                tris = facets_of_polygon(polygon)
                status, point = polygon_face_hit(
                    polygon, p0, u, tris)

                if status:
                    results.append(point)

        b = np.asarray(results)
        try:
            # hull = ConvexHull(b)
            return (b)
            # return (ConvexHull(b).volume)
        except Exception:
            if debug:
                print('\\\ failed [cam,ch]:',
                      cam, ch, end=' ')
            return (0.0)

    print('\tFinding volume of center LoS vertice inside ' +
          'fluxsurfaces ...')

    if saving:
        volume = store_read_volume(name=vmec_label, typ='raw_')
        volumeO = store_read_volume(name=vmec_label, typ='rawO_')
        if (volume is not None) and (volumeO is not None):
            return (volume, volumeO)

    nP, nFS, nL = np.shape(mesh)[1:]
    nT = np.shape(corners['values']['x'])[1]
    volume, volumeO = np.zeros((128, nT**2)), np.zeros((128))

    if vpF > 1.3:
        tFS = rescale_fs_number(vpF, nFS, 1.3)
    else:
        tFS = nFS - 1

    det = np.array([
        corners['values']['x'],
        corners['values']['y'],
        corners['values']['z']])
    detO = np.array([
        old['values']['x'],
        old['values']['y'],
        old['values']['z']])

    # loop over cams, channels, LCFS's and points in FS
    for cn, cam in enumerate(cams):
        apt = np.array([
            corners['aperture'][cam]['x'],
            corners['aperture'][cam]['y'],
            corners['aperture'][cam]['z']])

        aptO = np.array([
            old['aperture'][cam]['x'],
            old['aperture'][cam]['y'],
            old['aperture'][cam]['z']])

        nCh = camera_info['channels']['eChannels'][cam]
        for dn, ch in tqdm(enumerate(nCh[16:]), desc='cam: ' + cam):

            for p in range(nT):
                for d in range(nT):
                    # center LoS pyramid, center det --> corners apt
                    volume[ch, p * nT + d] = view_cone_volume(
                        aperture=apt[:, 1:, p],
                        detector=det[:, ch, d, :1],
                        ch=ch, nL=nL, nFS=tFS,
                        camera=cam, mesh=mesh, debug=False)

            # full LoS pyramid, corners det --> corners apt
            volumeO[ch] = view_cone_volume(
                aperture=aptO[:, 1:],
                detector=detO[:, ch, 1:],
                ch=ch, nL=nL, nFS=tFS,
                camera=cam, mesh=mesh, debug=False)

            return (tmp, b)

    if saving:
        store_read_volume(name=vmec_label, typ='raw_', data=volume)
        store_read_volume(name=vmec_label, typ='rawO_', data=volumeO)
    return (volume, volumeO)


def scale_volume(
        raw_volume=np.zeros((128, 64)),
        old_volume=np.zeros((128)),
        corners={'none]': None},
        cams=['HBCm'],
        camera_info={'none': None},
        vmec_label='test',
        new_type=None,
        saving=False,
        debug=False):

    def get_angles(
            a, b, normal):  # in m
        lofs_vec = np.array([
            a[0, 0] - b[0, 0],
            a[1, 0] - b[1, 0],
            a[2, 0] - b[2, 0]])
        u, v  = \
            np.linalg.norm(normal), \
            np.linalg.norm(lofs_vec)
        beta = np.rad2deg(np.arccos(np.dot(
            u, v) / (u * v)))

        if np.isnan(beta):
            beta = 0.0
        if beta > 90.:
            beta = np.abs(180. - beta)
        return (beta)  # deg, deg, in m

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

    def calculate_area(
            c):
        hull = ConvexHull(c.transpose(), qhull_options='QJ')
        return (hull.area / 2.)

    def scale_with_angles_areas(
            volume, detector, aperture, area_apt, area_det, p, d):
        # scale the volume according to angles
        normals_aperture = get_normal(rectangle=aperture)

        beta = get_angles(
            detector, aperture, normals_aperture)
        # angles between normals
        beta = np.abs(np.cos(np.deg2rad(beta)))
        delta = beta  # * alpha

        f_A = (area_apt[p] / np.sum(area_apt))  # * (
        #     area_det[d] / np.sum(area_det))

        return (volume * delta * f_A)

    def scale_with_angles(
            volume, detector, aperture):
        # scale the volume according to angles
        normals_aperture = get_normal(rectangle=aperture)

        beta = get_angles(
            detector, aperture, normals_aperture)
        # angles between normals
        beta = np.abs(np.cos(np.deg2rad(beta)))
        delta = beta  # * alpha

        return (volume * delta)

    print('\tScaling volume by angles and areas ...')
    if saving:
        volume = store_read_volume(name=vmec_label, typ='')
        volumeO = store_read_volume(name=vmec_label, typ='old_')
        if (volume is not None) and (volumeO is not None):
            return (volume, volumeO)

    nT = np.shape(corners['values']['x'])[1]
    volume = np.zeros((128, nT**2))
    volumeO = np.zeros((128))

    det = np.array([
        corners['values']['x'],
        corners['values']['y'],
        corners['values']['z']])

    # loop over cams, channels, LCFS's and points in FS
    for cn, cam in enumerate(cams):

        apt = np.array([
            corners['aperture'][cam]['x'],
            corners['aperture'][cam]['y'],
            corners['aperture'][cam]['z']])

        aA = np.zeros((nT))
        for p in range(nT):
            aA[p] = calculate_area(apt[:, :, p])

        nCh = camera_info['channels']['eChannels'][cam]
        for dn, ch in tqdm(enumerate(nCh), desc='cam: ' + cam):

            aD = np.zeros((nT))
            for d in range(nT):
                aD[d] = calculate_area(det[:, ch, d, :])

            for p in range(nT):
                for d in range(nT):
                    # rescale with angles
                    volume[ch, p * nT + d] = scale_with_angles_areas(
                        raw_volume[ch, p * nT + d],
                        apt[:, :, p], det[:, ch, d, :],
                        aA, aD, p, d)

            volumeO[ch] = scale_with_angles(
                old_volume[ch], apt[:, :, 0], det[:, ch, 0, :])

            if debug:
                print(cam, 'segm. -',
                      np.sum(volume[ch, :]) * 1.e2**3, ', full -',
                      volumeO[ch] * 1.e2**3)

    if saving:
        store_read_volume(name=vmec_label, typ='', data=volume)
        store_read_volume(name=vmec_label, typ='old_', data=volumeO)
    return (volume, volumeO)


def store_read_volume(
        data=None,  # np.zeros((128)),
        name='test',
        typ='raw_',
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

        elif not isfile(loc) and (P is None):
            print('\t\t\\\ not found ' + label.replace('_', ' '))
        return (P)

    data = load_store_prop('volume_' + typ, data)
    return (data)  # line sections
