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


def get_lines2D(
        mesh={'values': {'fs': {'108.': np.zeros((2, 25, 125))}}},
        N=2,
        LoS={'none': None},
        cams=['HBCm'],
        camera_info={'none': None},
        vmec_label='test',
        saving=True,
        debug=False):

    def pixel_from_mesh(
            s, l, mesh):
        pixel = np.zeros((2, 4))  # coords, 8 points
        pixel[:, 0] = mesh[:2, s, l]  # OG point
        pixel[:, 1] = mesh[:2, s, l + 1]  # next slice
        pixel[:, 2] = mesh[:2, s + 1, l + 1]  # next slice, further out
        pixel[:, 3] = mesh[:2, s + 1, l]  # same slice, further out
        return (pixel.transpose())

    def facets_of_pixel(
            pixel):  # 8 points, 3 coords
        facets = ConvexHull(pixel).simplices
        return (facets)

    def sort_mesh2D(
        mesh, p1, p2):
        res = []
        line = LineString([p1, p2])
        # find cells we need
        for S in range(np.shape(mesh)[1] - 1):
            for L in range(np.shape(mesh)[2] - 1):
                x, y = \
                    mesh[0, S:S + 2, L:L + 2].reshape(-1), \
                    mesh[1, S:S + 2, L:L + 2].reshape(-1)
                polygon = Polygon(np.array([x, y]).transpose())
                if line.intersects(polygon):
                    res.append([S, L])
        return (np.array(res))

    def LoS_vec(
            ch, T, line_data):
        p0 = np.array([
            line_data['rz_plane']['range'][ch, T][0],
            line_data['rz_plane']['line'][ch, T][0]])
        pE = np.array([
            line_data['rz_plane']['range'][ch, T][-1],
            line_data['rz_plane']['line'][ch, T][-1]])
        return (pE - p0, p0, pE)

    def s_factor(
            n, c, null, v):
        return (np.dot(n, c - null) / np.dot(n, v))

    def length_res(
            p1, p2):
        dX = np.array([
            p1[0] - p2[0], p1[1] - p2[1]])
        return (np.sqrt(np.dot(dX, dX)))

    def perp(
            a):
        return (np.array([-1. * a[1], a[0]]))

    def pixel_face_hit(
            pixel, null, u, indices):
        # LoS is P(s) = null + s * u
        status = False
        results = np.zeros((2, 2))
        for f, face in enumerate(indices):
            # face of pixel shall be
            # Q(t) = Q0 + t * v
            points = pixel[face]
            v = points[1] - points[0]
            # if not perpendicular
            if not (np.dot(perp(u), v) == .0):
                w = null - points[0]
                s1 = (np.dot(-1. * perp(v), w)) / (np.dot(perp(v), u))
                t1 = (np.dot(perp(u), w)) / (np.dot(perp(u), v))
                if (.0 <= s1 <= 1.) and (.0 <= t1 <= 1.):
                    results[0] = null + s1 * u

                    for g, gace in enumerate(indices[f + 1:]):
                        points = pixel[gace]
                        v = points[1] - points[0]
                        if not (np.dot(perp(w), v) == .0):
                            w = null - points[0]
                            s2 = (np.dot(-1. * perp(v), w)) / (
                                np.dot(perp(v), u))
                            t2 = (np.dot(perp(u), w)) / (
                                np.dot(perp(u), v))
                            if (.0 <= s2 <= 1.) and (.0 <= t2 <= 1.):
                                results[1] = null + s2 * u
                                status = True
                                break
                    break
        return (status, results)

    nFS, nL = np.shape(mesh['values']['fs']['108.'])[1:]
    print('\tFinding line intersections in 2D [N² x FS x L]:',
          N**2, 'x', nFS - 1, 'x', nL - 1, '...')

    # check if existent
    if saving:
        line_sections = store_read_line_sections(name=vmec_label)
        if line_sections is not None:
            return (line_sections)

    # storage
    line_sections = np.zeros((128, N**2, nFS - 1, nL - 1))

    # loop over cams, channels, LCFS's and points in FS2
    for cn, cam in enumerate(cams):
        m = mesh['values']['fs'][cam]

        nCh = camera_info['channels']['eChannels'][cam]
        for dn, ch in enumerate(nCh):

            for T in range(N**2):
                # let the parametric equation of the line be
                # P(s) = P0 + s * (P1 - P0) = P0 + s * U
                u, p0, pE = LoS_vec(ch, T, LoS['values'])

                nRange = sort_mesh2D(m[:2], p0, pE)
                if (np.shape(nRange) == (0,)):  # no hits
                    continue

                for i in tqdm(range(np.shape(nRange)[0]),
                        desc=cam + ' ch#' + str(dn) + ' T' + str(T)):
                    S, L = nRange[i]

                    pixel = pixel_from_mesh(s=S, l=L, mesh=m)

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

                    faces = facets_of_pixel(pixel)
                    status, results = pixel_face_hit(
                        pixel, p0, u, faces)

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

        elif not isfile(loc) and (P is not None):
            print('\t\t\\\ not found ' + label.replace('_', ' '))
        return (P)

    data = load_store_prop('line_sections2D_', data)
    return (data)  # line sections