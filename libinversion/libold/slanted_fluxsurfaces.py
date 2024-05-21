
""" **************************************************************************
    start of file """

import json
import numpy as np
import os
import requests
import math

import matplotlib.pyplot as p
from matplotlib.pyplot import cm

import mClass
import fluxsurface_plots as fsp

Z = np.zeros
one = np.ones

""" eo header
************************************************************************** """


def FS_cut_boloplane(
        alpha=20.,
        beta=60.,
        tor_phi=np.linspace(106.0, 110.0, 20),
        camera_info={'none': None},
        camera_geometry={'none': None},
        corner_geometry={'none': None},
        lines_of_sight={'none': None},
        fluxsurface={'none': None},
        cams=['HBCm', 'VBCl', 'VBCr'],
        vmec_label='EIM_000',
        plot=True,
        saving=True,
        debug=False):
    print('\tCutting fluxsurfaces through Bolometer LOS plane ...')

    FS = fluxsurface['values']['fs']
    LoS = lines_of_sight['values']
    center = fluxsurface['values']['magnetic_axes']

    tor_angles_108 = np.zeros((np.shape(FS[0])[0], np.shape(FS[0])[2]))
    idx, val = mClass.find_nearest(tor_phi, 108.)
    fs_108, center_108 = FS[idx], center[idx]
    tor_angles_108[:, :] = val

    results = np.zeros(np.shape(FS[0]))
    tor_angles = np.zeros((np.shape(FS[0])[0], np.shape(FS[0])[2]))
    mag_ax = np.zeros((4))

    output = {'values': {
        'fs': {}, 'phi': {}, 'magnetic_axes': {}, 'angles': None}}
    output['values']['fs']['108.'] = [fs_108.tolist()]
    output['values']['magnetic_axes']['108.'] = [center_108.tolist()]
    output['values']['phi']['108.'] = [tor_angles_108.tolist()]

    # defining the bolometer plane
    print('\t\tDefining the Bolometer plane...')
    for c, camera in enumerate(cams):
        cc = corner_geometry['aperture'][camera]
        apt = np.array([cc['x'][0, 0], cc['y'][0, 0],
                        cc['z'][0, 0], cc['r'][0, 0]])

        channels = camera_info['channels']['eChannels'][camera]
        ch1, ch2 = channels[0], channels[-1]

        p1 = np.array(
            [LoS['xy_plane']['range'][ch1][0, 0],
             LoS['xy_plane']['line'][ch1][0, 0],
             LoS['xz_plane']['line'][ch1][0, 0]])
        p2 = np.array(
            [LoS['xy_plane']['range'][ch2][0, 0],
             LoS['xy_plane']['line'][ch2][0, 0],
             LoS['xz_plane']['line'][ch2][0, 0]])

        # These two vectors are in the plane
        v1, v2 = p1 - apt[:3], p2 - apt[:3]
        # the cross product is a vector normal to the plane
        cp = np.cross(v1, v2)
        a, b, c = cp
        # This evaluates a * x3 + b * y3 + c * z3 which equals d
        d = np.dot(cp, p2)
        print('\t\tThe equation from ' + camera + ' is ' +
              format(a, '.2f'), 'x + ', format(b, '.2f'),
              'y + ', format(c, '.2f'), 'z = ', format(d, '.2f'))

        for n, phi in enumerate(tor_phi[:-1]):  # nAngles
            for m, fs in enumerate(FS[n]):  # nFS
                if debug:
                    print(np.shape(FS[n, m]))

                # also match magnetic axis
                pc0 = np.array([center[n, 0], center[n, 1], center[n, 2]])
                cu = np.array([
                    center[n + 1, 0] - pc0[0],
                    center[n + 1, 1] - pc0[1],
                    center[n + 1, 2] - pc0[2]])

                if np.dot(cp, cu) != 0.0:
                    # line and plane are not perpendicular
                    s = np.dot(cp, apt[:3] - pc0) / np.dot(cp, cu)
                    if 0.0 <= s <= 1.:
                        if debug:
                            print(s, pc0 + s * cu)
                        mag_ax[:3] = pc0 + s * cu
                        mag_ax[3] = np.sqrt(np.square(
                            mag_ax[0]) + np.square(mag_ax[1]))

                for l in range(np.shape(FS[n, m])[1]):  # points in FS
                    if debug:
                        print(np.shape(FS[n, m, :, l]))
                        print([FS[n, m, 0, l], FS[n + 1, m, 0, l]])
                        print(n, m, 0, l)

                    # x, y, z of same points on same fluxsurface in
                    # consecutive angles from VMEC
                    x, y, z = \
                        np.array([FS[n, m, 0, l], FS[n + 1, m, 0, l]]), \
                        np.array([FS[n, m, 1, l], FS[n + 1, m, 1, l]]), \
                        np.array([FS[n, m, 2, l], FS[n + 1, m, 2, l]])
                    # let the parametric equation of the line be
                    # P(s) = P0 + s * (P1 - P0) = P0 + s * U
                    p0 = np.array([x[0], y[0], z[0]])
                    u = np.array([x[1] - p0[0], y[1] - p0[1], z[1] - p0[2]])

                    if np.dot(cp, u) != 0.0:
                        # line and plane are not perpendicular
                        s = np.dot(cp, apt[:3] - p0) / np.dot(cp, u)
                        if 0.0 <= s <= 1.:
                            if debug:
                                print(s, p0 + s * u)

                            results[m, :3, l] = p0 + s * u
                            results[m, 3, l] = \
                                np.sqrt(np.square(results[m, 0, l]) +
                                        np.square(results[m, 1, l]))

                            tor_angles[m, l] = np.mean(tor_phi[n:n + 2])
                            if debug:
                                print(camera, m, l, 'tor angle:',
                                      format(tor_angles[m, l], '.3f'),
                                      format(tor_phi[n], '.3f'),
                                      format(tor_phi[n + 1], '.3f'))

        if plot:
            fsp.slanted_fs_plot(
                fluxsurfaces=fluxsurface, slanted_FS=results[1:],
                axis=mag_ax, vmec_label=vmec_label,
                camera=camera, suffix='2d')

        # put results
        output['values']['fs'][camera] = [results[:].tolist()]
        output['values']['phi'][camera] = [tor_angles.tolist()]
        output['values']['magnetic_axes'][camera] = [mag_ax.tolist()]
    # only thing that stays the same
    output['values']['angles'] = [108.]

    if saving:
        print('\t\tSaving ...')

        vmec_label += '_slanted'
        outdict = mClass.dict_transf(output, list_bool=True)
        with open('../results/INVERSION/FS/' + vmec_label +
                  'fs_data.json', 'w') as outfile:
            json.dump(outdict, outfile, indent=4, sort_keys=False)
        outfile.close()
        output = mClass.dict_transf(output, list_bool=False)

    return (output, vmec_label)
