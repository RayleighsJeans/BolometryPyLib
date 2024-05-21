""" **************************************************************************
    so header """

import sys
from os.path import isfile
import json

import numpy as np
import mClass

""" end of header
************************************************************************** """


def setup2D_mesh(
        nFS=25,
        nL=125,
        nPhi=80,
        mesh3D=np.zeros((4, 80, 25, 125)),
        tor_phi=[108.],
        geometry={'none': None},
        camera_info={'none': None},
        cams=['HBCm'],
        vmec_label='test',
        saving=True,
        debug=False):

    def define_camera_plane(
            cam,
            aperture,
            channels,
            detectors):
        A = np.array([
            aperture[cam]['x'],
            aperture[cam]['y'],
            aperture[cam]['z']])
        # channels spanning plane
        ch0, chE = channels[0], channels[-1]
        # These two vectors are in the plane
        v0, p00, p0E = LoS_vec_from_points(
            cam, detectors[:, ch0, 0], A[:, 0])
        vE, pE0, pEE = LoS_vec_from_points(
            cam, detectors[:, chE, 0], A[:, 0])
        # the cross product is a vector normal to the plane
        cp = np.cross(v0, vE)
        return (cp, A[:, 0])

    def LoS_vec_from_points(
            cam,
            p1,
            p2):
        tx = -1. if (cam == 'HBCm') else -2.  # endpoint
        # xy plane
        s_xy = (p2[1] - p1[1]) / (p2[0] - p1[0])
        c_xy = p2[1] - s_xy * p2[0]
        # xz plamne
        s_xz = (p2[2] - p1[2]) / (p2[0] - p1[0])
        c_xz = p2[2] - s_xz * p2[0]

        pE = np.array([  # endpoint
            tx, s_xy * tx + c_xy, s_xz * tx + c_xz])
        p0 = p1  # first point
        return (pE - p0, p0, pE)

    def points_from_mesh3D(
            P,
            S,
            L,
            mesh):
        p0 = np.array([
            mesh[0, P, S, L],
            mesh[1, P, S, L],
            mesh[2, P, S, L]])
        p1 = np.array([
            mesh[0, P + 1, S, L],
            mesh[1, P + 1, S, L],
            mesh[2, P + 1, S, L]])
        return (p0, p1)

    print('\tMeshing in 2D [FS x L]:',
          nFS + 1, 'x', nL + 1, '...')

    if saving:
        output = store_read_mesh2D(name=vmec_label)
        if output is not None:
            return (output)

    output = {'values': {
        'fs': {}, 'phi': {},
        'magnetic_axes': {},
        'angles': None}}

    idx, val = mClass.find_nearest(tor_phi, 108.)
    fs_108 = np.array([mesh3D[3, idx], mesh3D[2, idx]])
    tor_angles_108 = np.zeros((nFS + 1, nL + 1))
    tor_angles_108[:, :] = val

    output['values']['fs']['108.'] = fs_108
    output['values']['phi']['108.'] = tor_angles_108

    A = geometry['aperture']
    detectors = np.array([
        geometry['values']['x'],
        geometry['values']['y'],
        geometry['values']['z']])

    for cn, cam in enumerate(cams):
        nCh = camera_info['channels']['eChannels'][cam]

        results = np.zeros((2, nFS + 1, nL + 1))
        tor_angles = np.zeros((nFS + 1, nL + 1))
        pn, pA = define_camera_plane(cam, A, nCh, detectors)[:3]

        for P in range(nPhi - 1):  # nAngles

            for S in range(nFS + 1):  # nFS

                for L in range(nL + 1):  # points in FS

                    p0, p1 = points_from_mesh3D(P, S, L, mesh3D)
                    # let the parametric equation of the line be
                    # P(s) = P0 + s * (P1 - P0) = P0 + s * U
                    u = p1 - p0

                    if np.dot(pn, u) != 0.0:
                        # line and plane are not perpendicular
                        s = np.dot(pn, pA - p0) / np.dot(pn, u)
                        if 0.0 <= s <= 1.:
                            res = p0 + s * u

                            results[0, S, L] = np.sqrt(res[0]**2 + res[1]**2)
                            results[1, S, L] = res[2]
                            tor_angles[S, L] = np.mean(tor_phi[P:P + 2])

        # put results
        output['values']['fs'][cam] = results
        output['values']['phi'][cam] = tor_angles

    if saving:
        store_read_mesh2D(output, name=vmec_label)
    return (output)


def store_read_mesh2D(
        data=None,  # np.zeros((128)),
        name='test',
        base='../results/INVERSION/MESH/',
        overwrite=False):

    def shape_size_str(
            data):
        return (str(format(sys.getsizeof(
            data) / (1024. * 1024.), '.2f') + 'MB'))

    def load_store_prop(
            label='test',
            P=None):
        loc = base + label + name + '.json'

        if isfile(loc) and not overwrite:
            with open(loc, 'r') as infile:
                P = json.load(infile)
            infile.close()
            P = mClass.dict_transf(
                P, to_list=False)
            print('\t\t\\\ loaded ' + label.replace('_', ' '),
                  shape_size_str(P))

        elif isfile(loc) and overwrite and (P is not None):
            print('\t\t\\\ overwriting ' + label.replace('_', ' '),
                  shape_size_str(P))
            with open(loc, 'w') as outfile:
                json.dump(mClass.dict_transf(P, list_bool=True),
                          outfile, indent=4, sort_keys=False)
            outfile.close()
            P = mClass.dict_transf(P, list_bool=False)

        elif not isfile(loc) and (P is not None):
            print('\t\t\\\ save ' + label.replace('_', ' '),
                  shape_size_str(P))
            with open(loc, 'w') as outfile:
                json.dump(mClass.dict_transf(P, to_list=True),
                          outfile, indent=4, sort_keys=False)
            outfile.close()
            P = mClass.dict_transf(P, to_list=False)

        elif not isfile(loc) and (P is None):
            print('\t\t\\\ not found ' + label.replace('_', ' '))
        return (P)

    data = load_store_prop('mesh2D_', data)
    return (data)  # line sections
