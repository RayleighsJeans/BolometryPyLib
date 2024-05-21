""" **************************************************************************
    so header """

import requests as req
import json
from os.path import isfile
import sys

import numpy as np
import requests
from osa import Client

import factors_geometry_plots as fgp

Z = np.zeros
ones = np.ones
vmec = Client('http://esb:8280/services/vmec_v8?wsdl')

""" end of header
************************************************************************** """


def effective_radius2D(
        lines=np.zeros((128, 4, 30, 125)),
        mesh_data={'HBCm': np.zeros((2, 3 * 125))},
        camera_info={'none': None},
        vmec_label='EIM_000',
        cams=['HBCm', 'VBCl', 'VBCr', 'ARTf'],
        new_type=None,
        vmec_ID='1000_1000_1000_1000_+0000_+0000/01/00jh_l',
        w7x_ref='w7x_ref_172',
        pre='../results/INVERSION/MESH/TRIAGS/',
        saving=False,
        debug=False):
    print('\tGet effective radius along LoS ...')

    if saving:
        reff_channels, pos_lofs, minor_radius = store_read_profile(
            name=vmec_label)
        if reff_channels is not None:
            return (reff_channels, pos_lofs, minor_radius)

    minor_radius = req.get(
        'http://svvmec1.ipp-hgw.mpg.de:8080/' +
        'vmecrest/v1/geiger/w7x/' + vmec_ID +
        '/minorradius.json').json()['minorRadius']

    # load stuff
    mesh2D = mesh_data['values']['fs']
    cell_phi = mesh_data['values']['phi']
    N, nFS, nL = np.shape(lines)[1:]

    reff_channels = np.zeros((128 + 1, N, nFS, nL))
    pos_lofs = np.zeros((128 + 1, nFS, nL, 4))

    nchn = []
    for c in cams:
        nchn.append(
            camera_info['channels']['eChannels'][c])

    for S in range(nFS):
        for L in range(nL):

            # fill up position in mesh
            p1, p2, p3, p4 = fgp.poly_from_mesh(  # m
                m=mesh2D['108.'], S=S, L=L, nL=nL, nFS=nFS)

            pos_lofs[-1, S, L] = coordinate_transform_mesh(
                polygon=[p1, p2, p3, p4], phi=cell_phi['108.'][S, L])
            if np.isnan(pos_lofs[-1, S, L]).any():
                print('\t\t\\\ 108.', S, L, p1, p2, p3, p4,
                      pos_lofs[-1, S, L])

            for cam, nCH in zip(cams, nchn):
                if (cam == 'ARTf'):
                    cam = 'HBCm' if (new_type == 'MIRh') else 'VBCl'

                # fill up position in mesh
                p1, p2, p3, p4 = fgp.poly_from_mesh(  # m
                    m=mesh2D[cam], S=S, L=L, nL=nL, nFS=nFS)

                pos_lofs[nCH, S, L] = coordinate_transform_mesh(
                    polygon=[p1, p2, p3, p4], phi=cell_phi[cam][S, L])

                if np.isnan(pos_lofs[nCH, S, L]).any():
                    p1, p2, p3, p4 = fgp.poly_from_mesh(  # m
                            m=mesh2D['108.'], S=S, L=L, nL=nL, nFS=nFS)
                    pos_lofs[nCH, S, L] = coordinate_transform_mesh(
                        polygon=[p1, p2, p3, p4], phi=cell_phi['108.'][S, L])

    def async_download_reff(
            nc, positions):
        p = vmec.types.Points3D()
        V = positions[nc, :, :, :3].reshape(nFS * nL, 3)
        p.x1, p.x2, p.x3 = V[:, 0], V[:, 1], V[:, 2]
        results = np.array(vmec.service.getReff(
            w7x_ref, p))
        del p, V
        return (results.reshape(nFS, nL))

    for cam, nCH in zip(cams, nchn): # cam planes
        reff_channels[nCH] = async_download_reff(nCH[0], pos_lofs)
    # 108. deg plane for display
    reff_channels[-1] = async_download_reff(-1, pos_lofs)

    for S in range(nFS):
        for L in range(nL):

            if (np.isnan(reff_channels[-1, 0, S, L])) or (
                    np.isinf(reff_channels[-1, 0, S, L])):

                reff_channels[-1, :, S, L] = scale_SOL_reff(
                    S=S, L=L, reffs=reff_channels[-1, 0],
                    magax=mesh2D['108.'][:, 0, 0],
                    position=pos_lofs[-1], debug=False)

            for cam, nCH in zip(cams, nchn):
                if (cam == 'ARTf'):
                    cam = 'HBCm' if (new_type == 'MIRh') else 'VBCl'

                if (np.isnan(reff_channels[nCH[0], 0, S, L])) or (
                        np.isinf(reff_channels[nCH[0], 0, S, L])):
                    reff_channels[nCH, :, S, L] = scale_SOL_reff(
                        S=S, L=L, reffs=reff_channels[nCH[0], 0],
                        magax=mesh2D['108.'][:, 0, 0],
                        position=pos_lofs[nCH[0]], debug=False)

    if saving:
        reff_channels, pos_lofs, minor_radius = store_read_profile(
            name=vmec_label, position=pos_lofs, reff=reff_channels,
            minor_radius=minor_radius)
    return (reff_channels, pos_lofs, minor_radius)


def coordinate_transform_mesh(
        polygon=[Z((2))],  # m, n corners, 2 coords
        phi=108.,  # toroidal angle
        debug=False):
    """ coordinate transformation
    Keyword Arguments:
        polygon {[type]} -- polygon from where to transf (default: {Z((2, 2))})
        phi {[type]} -- toroidal phi (default: {np.pi})
        theta {[type]} -- poloidal theta (default: {np.pi})
        debug {bool} -- debugging (default: {False})
    Returns:
        (X, Y, Z, R) {tuple} -- tuple of 4 spatial coordinates
    """
    def barycenter(
            points,  # zeros((n, 3)) n points x 3 dims
            debug=False):
        bc = np.zeros((np.shape(points)[1]))
        for k in range(np.shape(points)[1]):
            bc[k] = np.mean(points[:, k])
        return (bc)  # x, y, z

    center = barycenter(np.array(
        [[v[0], v[1]] for v in polygon]))
    R, Z = center[0], center[1]

    r = np.sqrt(R**2 + Z**2)
    if Z >= 0:
        theta = np.arccos(R / r)
    else:
        theta = 2. * np.pi - np.arccos(R / r)
    theta = np.pi / 2. - theta if (theta < np.pi / 2.) \
        else 2. * np.pi - (theta - np.pi / 2.)

    X = R * np.sin(theta) * np.cos(np.deg2rad(phi))  # in m
    Y = R * np.sin(theta) * np.sin(np.deg2rad(phi))  # in m
    return (X, Y, Z, R)


def scale_SOL_reff(
        S=0,  # S
        L=0,  # L
        reffs=Z((32, 31, 75)),  # m, reduced for cam
        magax=Z((4)),  # m (X,Y,Z,R)
        position=Z((31, 75, 4)),  # m (X,Y,Z,R)
        debug=False):
    """ scale new reff based off of previous inside LCFS
    Keyword Arguments:
        S {int} -- (default: {0})
        L {int} -- (default: {0})
        reffs {[type]} -- previous r_eff
        position {[type]} -- positios where r_effs defined
        magax {dict} -- magnetic axis pos
        debug {bool} -- debugging (default: {False})
    Returns:
        reff {float} -- scaled r_eff
    """
    if debug:
        print('ids1[S,L]:', S, L, end=' ')
    mx = np.max(reffs[:S, L])  # m
    if debug:
        print('mx:', format(mx, '.3f'), end=' ')

    ids = np.where(reffs[:S, L] == mx)[0]
    if debug:
        print('ids2[S]:', ids[0], end=' ')

    d1 = np.sqrt(
        (position[S, L, 2] - magax[1])**2 +
        (position[S, L, 3] - magax[0])**2)
    d2 = np.sqrt(
        (position[ids[0], L, 2] - magax[1])**2 +
        (position[ids[0], L, 3] - magax[0])**2)
    if debug:
        print('d1:', format(d1, '.3f'),
              'd2:', format(d2, '.3f'), end=' ')
    if False:
        print('loc1[R,Z]:', format(position[S, L, 3], '.3f'),
              format(position[S, L, 2], '.3f'),
              'loc2[R,Z]:', format(position[ids[0], L, 3], '.3f'),
              format(position[ids[0], L, 2], '.3f'), end=' ')

    new_reff = mx * (d1 / d2)
    if debug:
        print('f:', format(d1 / d2, '.3f'))
        if False:
            print('new:', format(new_reff, '.3f'))
    return (new_reff)


def store_read_profile(
        name='test',
        position=None,  # np.zeros((128, 16, 31, 75, 2, 2)),
        reff=None,  # np.zeros((128, 16, 31, 75)),
        minor_radius=None,  # {'minor_radius': None},
        base='../results/INVERSION/MESH/TRIAGS/',
        overwrite=False):
    data = []

    def shape_size_np(
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

    labels = ['positions_', 'reff_']
    properties = [position, reff]
    for l, p in zip(labels, properties):
        data.append(load_store_prop(l, p))

    # minor radius
    file = base + 'minor_radius_' + name + '.json'
    if isfile(file):
        with open(file, 'r') as infile:
            print('\t\t\\\ load minor radius')
            minor_radius = json.load(infile)
        infile.close()

    else:
        if minor_radius is not None:
            with open(file, 'w') as outfile:
                print('\t\t\\\ save minor radius')
                json.dump(
                    minor_radius, outfile,
                    indent=4, sort_keys=False)
            outfile.close()
        else:
            print('\t\t\\\ minor radius is None')

    return (data[1], data[0], minor_radius)
