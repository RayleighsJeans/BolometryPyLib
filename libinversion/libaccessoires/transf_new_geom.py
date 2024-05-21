""" **************************************************************************
    start of file """

import json
import os

import numpy as np
import pandas as pd

import mClass
import dat_lists

import camera_geometries as cgeom
import LoS_emissivity3D as LoS3D

Z = np.zeros
one = np.ones
line = np.linspace

""" eo header
************************************************************************** """

luts = {
    'HBCm': np.array([  # first new chans, then actual eChans
        [35, 34, 33, 32, 39, 38, 37, 36, 43, 42, 41, 40, 47, 46, 45, 44,
         51, 50, 49, 48, 55, 54, 53, 52, 59, 58, 57, 56, 63, 62, 61, 60],
        [31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
         15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]),
    'VBCr': np.array([  # first new chans, then actual eChans
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
         18, 19, 20, 21, 22, 23],
        [87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72,
         71, 70, 69, 68, 67, 66, 65, 64]]),
    'VBCl': np.array([  # first new chans, then actual eChans
        [44, 45, 46, 47, 40, 41, 42, 43, 36, 37, 38, 39, 32, 33, 34, 35, 28,
         29, 30, 31, 24, 25, 26, 27],
        [88, 89, 90, 91, 92, 93, 94, 95, 48, 49, 50, 51, 52, 53, 54, 55, 56,
         57, 58, 59, 60, 61, 62, 63]])}


def load_new_geom_xlc(
        loc='../files/geom/camgeo/',
        file='Bolo-Koordinaten 2019 TH-Koordinaten.xlsx',
        sheet='Tabelle1'):
    return (
        pd.read_excel(
            io=loc + file,
            sheet_name=sheet))


def transf_new_geom(
        camera_info={'none': None},
        old=None,  # {'none': None},
        compare=True,
        loc='../results/INVERSION/CAMGEO/',
        saving=True,
        debug=False):
    file = loc + 'new_corner_geometry.json'
    if saving:
        if os.path.isfile(file):
            with open(file, 'r') as infile:
                out = json.load(infile)
            infile.close()
            corners = mClass.dict_transf(
                out, to_list=False)
            return (corners)

    if compare:
        camera_info = dat_lists.geom_dat_to_json(saving=debug)
        geometry = cgeom.camera_geometry(
            camera_info=camera_info, debug=debug)
        old = cgeom.camera_corners(
            debug=debug, camera_info=camera_info,
            saving=False, geometry=geometry)

    corners = {
        'label': 'camera_corners',
        'values': {
            'x': np.zeros((128, 5)),
            'y': np.zeros((128, 5)),
            'z': np.zeros((128, 5)),
            'r': np.zeros((128, 5))},
        'aperture': {}
    }
    for cam in ['HBCm', 'VBCl', 'VBCr']:
        corners['aperture'][cam] = {
            'x': np.zeros((5)), 'y': np.zeros((5)),
            'z': np.zeros((5)), 'r': np.zeros((5))}

    new = load_new_geom_xlc()
    N, M = np.shape(new)
    keys = new.keys()

    for n in range(N):

        cam = new[keys[0]][n]
        det = [int(s) for s in new[keys[1]][n].split()
               if s.isdigit()]
        chip = [int(s) for s in str(new[keys[2]][n]).split()
                if s.isdigit()]
        corner = int(new[keys[3]][n])

        x, y, z = \
            new[keys[4]][n] / 1000., \
            new[keys[5]][n] / 1000., \
            new[keys[6]][n] / 1000.

        lut = luts[cam]

        if (det == []) or (chip == []):  # is pinhole
            if debug:
                print(n, '/', N, cam, corner, 'pinhole')

            corners['aperture'][cam]['x'][corner] = x
            corners['aperture'][cam]['y'][corner] = y
            corners['aperture'][cam]['z'][corner] = z
            corners['aperture'][cam]['r'][corner] = np.sqrt(
                corners['aperture'][cam]['x'][corner]**2 +
                corners['aperture'][cam]['y'][corner]**2)

            if corner == 4:
                corners['aperture'][cam]['x'][0] = np.mean(
                    corners['aperture'][cam]['x'][1:])
                corners['aperture'][cam]['y'][0] = np.mean(
                    corners['aperture'][cam]['y'][1:])
                corners['aperture'][cam]['z'][0] = np.mean(
                    corners['aperture'][cam]['z'][1:])
                corners['aperture'][cam]['r'][0] = np.sqrt(
                    corners['aperture'][cam]['x'][0]**2 +
                    corners['aperture'][cam]['y'][0]**2)

                if (cam != 'HBCm'):
                    # wierdly, those cameras are noted in half of side points
                    for k in ['x', 'y', 'z']:
                        corners['aperture'][cam][k][1:] = np.array([
                            corners['aperture'][cam][k][1] +
                            corners['aperture'][cam][k][2] -
                            corners['aperture'][cam][k][0],
                            corners['aperture'][cam][k][2] +
                            corners['aperture'][cam][k][3] -
                            corners['aperture'][cam][k][0],
                            corners['aperture'][cam][k][3] +
                            corners['aperture'][cam][k][4] -
                            corners['aperture'][cam][k][0],
                            corners['aperture'][cam][k][4] +
                            corners['aperture'][cam][k][1] -
                            corners['aperture'][cam][k][0]])

        elif (n >= 132):  # here starts actual array
            chip = chip[0]
            det = det[0]
            eChan = (det - 1) * 4 + chip - 1
            old_eChan = lut[1][np.where(lut[0] == eChan)][0]
            if debug:
                print(n, '/', N, cam, 'det:', det, 'chip:', chip,
                      'eChan:', eChan,
                      'old:', old_eChan, 'K:', corner,
                      '(x, y, z): (' + format(x, '.4f') + ', ' +
                      format(y, '.4f') + ', ' + format(z, '.4f') + ')')

            corners['values']['x'][old_eChan, corner] = x
            corners['values']['y'][old_eChan, corner] = y
            corners['values']['z'][old_eChan, corner] = z
            corners['values']['r'][old_eChan, corner] = np.sqrt(
                corners['values']['x'][old_eChan, corner]**2 +
                corners['values']['y'][old_eChan, corner]**2)

            if corner == 4:
                corners['values']['x'][old_eChan, 0] = np.mean(
                    corners['values']['x'][old_eChan, 1:])
                corners['values']['y'][old_eChan, 0] = np.mean(
                    corners['values']['y'][old_eChan, 1:])
                corners['values']['z'][old_eChan, 0] = np.mean(
                    corners['values']['z'][old_eChan, 1:])

                corners['values']['r'][old_eChan, 0] = np.sqrt(
                    corners['values']['x'][old_eChan, 0]**2 +
                    corners['values']['y'][old_eChan, 0]**2)

    if old is not None:
        compare_old_new_corners(
            camera_info=camera_info, old=old, new=corners)

    if saving:
        out = mClass.dict_transf(
            corners, to_list=True)
        with open(file, 'w') as outfile:
            json.dump(out, outfile, indent=4, sort_keys=False)
        outfile.close()
        corners = mClass.dict_transf(
            out, to_list=False)

    return (corners)


def compare_old_new_corners(
        camera_info={'none': None},
        old={'none': None},
        new={'none': None},
        debug=False):

    for cam in ['HBCm', 'VBCl', 'VBCr']:
        # get opening angle of cameras to compare for artificial array
        cO, cE = camera_info['channels']['eChannels'][cam][0], \
            camera_info['channels']['eChannels'][cam][-1]

        v1 = cgeom.v_norm([  # same orientations, from apt to detector
            old['values']['x'][cO, 0] - old['aperture'][cam]['x'][0],
            old['values']['y'][cO, 0] - old['aperture'][cam]['y'][0],
            old['values']['z'][cO, 0] - old['aperture'][cam]['z'][0]])[0]
        v2 = cgeom.v_norm([  # same, apt to det
            old['values']['x'][cE, 0] - old['aperture'][cam]['x'][0],
            old['values']['y'][cE, 0] - old['aperture'][cam]['y'][0],
            old['values']['z'][cE, 0] - old['aperture'][cam]['z'][0]])[0]
        phi_old = np.rad2deg(cgeom.vector_angle(v1=v1, v2=v2))

        v1 = cgeom.v_norm([  # same orientations, from apt to detector
            new['values']['x'][cO, 0] - new['aperture'][cam]['x'][0],
            new['values']['y'][cO, 0] - new['aperture'][cam]['y'][0],
            new['values']['z'][cO, 0] - new['aperture'][cam]['z'][0]])[0]
        v2 = cgeom.v_norm([  # same, apt to det
            new['values']['x'][cE, 0] - new['aperture'][cam]['x'][0],
            new['values']['y'][cE, 0] - new['aperture'][cam]['y'][0],
            new['values']['z'][cE, 0] - new['aperture'][cam]['z'][0]])[0]
        phi_new = np.rad2deg(cgeom.vector_angle(v1=v1, v2=v2))

        # spatial shift of aperture
        dX = np.array([
            new['aperture'][cam]['x'][0] - old['aperture'][cam]['x'][0],
            new['aperture'][cam]['y'][0] - old['aperture'][cam]['y'][0],
            new['aperture'][cam]['z'][0] - old['aperture'][cam]['z'][0]])
        dR = np.sqrt(np.dot(dX, dX))

        # angle between old normal and new
        old_slit = np.array([  # m, 3 dims, 4 points (center, corners) N triags
            old['aperture'][cam]['x'], old['aperture'][cam]['y'],
            old['aperture'][cam]['z']])
        vnorm_old = cgeom.v_norm(LoS3D.get_normal(rectangle=old_slit))[0]
        new_slit = np.array([  # m, 3 dims, 4 points (center, corners) N triags
            new['aperture'][cam]['x'], new['aperture'][cam]['y'],
            new['aperture'][cam]['z']])
        vnorm_new = cgeom.v_norm(LoS3D.get_normal(rectangle=new_slit))[0]
        phi_normals = np.rad2deg(cgeom.vector_angle(
            v1=vnorm_old, v2=vnorm_new))
        if phi_normals > 90.:
            phi_normals = 180. - phi_normals

        d1 = np.sqrt(
            (new['aperture'][cam]['x'][2] - new['aperture'][cam]['x'][1])**2 +
            (new['aperture'][cam]['y'][2] - new['aperture'][cam]['y'][1])**2 +
            (new['aperture'][cam]['z'][2] - new['aperture'][cam]['z'][1])**2)
        d2 = np.sqrt(
            (new['aperture'][cam]['x'][3] - new['aperture'][cam]['x'][1])**2 +
            (new['aperture'][cam]['y'][3] - new['aperture'][cam]['y'][1])**2 +
            (new['aperture'][cam]['z'][3] - new['aperture'][cam]['z'][1])**2)
        d3 = np.sqrt(
            (new['aperture'][cam]['x'][4] - new['aperture'][cam]['x'][1])**2 +
            (new['aperture'][cam]['y'][4] - new['aperture'][cam]['y'][1])**2 +
            (new['aperture'][cam]['z'][4] - new['aperture'][cam]['z'][1])**2)
        d4 = np.sqrt(
            (new['aperture'][cam]['x'][4] - new['aperture'][cam]['x'][2])**2 +
            (new['aperture'][cam]['y'][4] - new['aperture'][cam]['y'][2])**2 +
            (new['aperture'][cam]['z'][4] - new['aperture'][cam]['z'][2])**2)

        print('\t Camera:', cam, 'opening angle: old', format(
              phi_old, '.3f') + '°, new:', format(phi_new, '.3f') + '°,',
              'shifted by ' + format(dR * 1000., '.3f') + 'mm,',
              'normal angles:' + format(phi_normals, '.3f') + '°\n\t' +
              ' a,b,c,d:', format(d1 * 1e3, '.3f') + 'mm, ' +
              format(d2 * 1e3, '.3f') + 'mm, ' + format(d3 * 1e3, '.3f') +
              'mm, ' + format(d4 * 1e3, '.3f'), 'mm, A:',
              format(d1 * d3 * 1e6, '.3f'), 'mm2')

        print('\t Angle between old, new:')
        for n, ch in enumerate(camera_info['channels']['eChannels'][cam]):
            # view axis through centers
            vnew = cgeom.v_norm([
                new['values']['x'][ch, 0] - new['aperture'][cam]['x'][0],
                new['values']['y'][ch, 0] - new['aperture'][cam]['y'][0],
                new['values']['z'][ch, 0] - new['aperture'][cam]['z'][0]])[0]
            vold = cgeom.v_norm([
                old['values']['x'][ch, 0] - old['aperture'][cam]['x'][0],
                old['values']['y'][ch, 0] - old['aperture'][cam]['y'][0],
                old['values']['z'][ch, 0] - old['aperture'][cam]['z'][0]])[0]
            phi = np.rad2deg(cgeom.vector_angle(v1=vold, v2=vnew))

            dX = np.array([
                new['values']['x'][ch, 0] - old['values']['x'][ch, 0],
                new['values']['y'][ch, 0] - old['values']['y'][ch, 0],
                new['values']['z'][ch, 0] - old['values']['z'][ch, 0]])
            dR = np.sqrt(np.dot(dX, dX))

            # angle between old normal and new of detector
            old_det = np.array([
                old['values']['x'][ch], old['values']['y'][ch],
                old['values']['z'][ch]])
            vdet_old = cgeom.v_norm(LoS3D.get_normal(rectangle=old_det))[0]
            new_det = np.array([
                new['values']['x'][ch], new['values']['y'][ch],
                new['values']['z'][ch]])
            vdet_new = cgeom.v_norm(LoS3D.get_normal(rectangle=new_det))[0]
            phi_det = np.rad2deg(cgeom.vector_angle(
                v1=vdet_old, v2=vdet_new))
            if phi_det > 90.:
                phi_det = 180. - phi_det

            print('\t\t' + cam + ' ' + str(ch) + ':',
                  format(phi, '.3f') + '°, shift ' + format(
                      dR * 1000., '.3f') + 'mm, normals ' + format(
                          phi_det, '.3f') + '°')
    return

# old from camera_geometries
# old geometry
# corner_geometry = camera_geometry(
#     camera_info=camera_info, debug=False)
# def camera_geometry(
#         camera_info={'channels': {'eChannels': {}}, 'geometry': {}},
#         loc='../results/INVERSION/CAMGEO/',
#         file_vbcl='../files/geom/camgeo/camVBCl_geo.dat',
#         file_vbcr='../files/geom/camgeo/camVBCr_geo.dat',
#         file_hbcm='../files/geom/camgeo/camHBCm_geo.dat',
#         saving=True,
#         debug=False):
#     """ creating LOFS geometry and camera corner locations in X, Y, Z, R
#     Args:
#         loc (0, str): location to save to
#         file_hbcm (1, str): loc to load coordinates center detector sides
#         file_vbcl (2, str): --"--
#         file_vbcr (3, str): --"--
#         saving (4, bool): saving
#         debug (5, bool): debugging
#     Returns:
#         geometry (0, dict): return of overall camera geometry results
#     Notes:
#         None.
#     """
#     print('\tCamera Geometry...')
#     offset = [.0, .0, .0]  # in m
#     x, y, z, r = np.zeros((128, 5)), np.zeros((128, 5)), \
#         np.zeros((128, 5)), np.zeros((128, 5))
#     geometry = {
#         'label': 'camera_geometry', 'values': None, 'aperture': {}}
#
#     # load from files
#     cg_vbcl, cg_vbcr, cg_hbcm = np.loadtxt(file_vbcl), \
#         np.loadtxt(file_vbcr), np.loadtxt(file_hbcm)
#
#     for camera, file in zip(
#             ['HBCm', 'VBCl', 'VBCr'], [cg_hbcm, cg_vbcl, cg_vbcr]):
#         N, L = np.shape(file)[0:2]
#         for i, ch in enumerate(camera_info['channels']['eChannels'][camera]):
#             if debug:
#                 print(camera, i, ch)
#             x[ch, :] = (file[i, 0::3] + offset[0])  # in m
#             y[ch, :] = (file[i, 1::3] + offset[1])  # in m
#             z[ch, :] = (file[i, 2::3] + offset[2])  # in m
#
#         if debug:
#             print(camera, N, L, N - 1, file[N - 1, 0::3], file[i, 0::3])
#         geometry['aperture'][camera] = {
#             'x': (file[N - 1, 0::3] + offset[0]),
#             'y': (file[N - 1, 1::3] + offset[1]),
#             'z': (file[N - 1, 2::3] + offset[2])}
#         geometry['aperture'][camera]['r'] = \
#             np.sqrt(geometry['aperture'][camera]['x']**2 +
#                     geometry['aperture'][camera]['y']**2)
#
#         # get opening angle of cameras to compare for artificial array
#         cO, cE = camera_info['channels']['eChannels'][camera][0], \
#             camera_info['channels']['eChannels'][camera][-1]
#
#         v1 = v_norm([  # same orientations, from apt to detector
#             x[cO, 0] - geometry['aperture'][camera]['x'][0],
#             y[cO, 0] - geometry['aperture'][camera]['y'][0],
#             z[cO, 0] - geometry['aperture'][camera]['z'][0]])[0]
#         v2 = v_norm([  # same, apt to det
#             x[cE, 0] - geometry['aperture'][camera]['x'][0],
#             y[cE, 0] - geometry['aperture'][camera]['y'][0],
#             z[cE, 0] - geometry['aperture'][camera]['z'][0]])[0]
#
#         phi = np.rad2deg(vector_angle(v1=v1, v2=v2))
#         print('\t\t Camera:', camera, 'opening angle:', format(
#               phi, '.3f'), '°')
#
#     r = np.sqrt(x**2 + y**2)
#     if debug:
#         print(np.shape(cg_hbcm), file[-1, :])
#
#     # dump to dictionary for file later
#     geometry['values'] = {'x': x, 'y': y, 'z': z, 'r': r}
#
#     if saving:
#         file = loc + 'camera_geometry.json'
#         out = mClass.dict_transf(
#             geometry, to_list=True)
#         with open(file, 'w') as outfile:
#             json.dump(out, outfile, indent=4, sort_keys=False)
#         outfile.close()
#
#     return geometry  # in m
#
#
# def camera_corners(
#         camera_info={'channels': {'eChannels': {}}, 'geometry': {}},
#         geometry={'none': None},
#         loc='../results/INVERSION/CAMGEO/',
#         saving=True,
#         debug=False):
#     """ calculating detector/slit corners based on center points of each side
#     Args:
#         camera_info (0, dict): camera data with channels and geometry
#         geometry (1, dict): lists/arrays of camera geometry
#         saving (2, bool): saving in file
#         debug (3, bool): debugging
#     Returns:
#         corner_geometry (0, dict): constructed corner points as shown below
#     Notes:
#         None.
#     """
#     # dictionary setup
#     print('\tConstructing corner coordinates...')
#     C = mClass.dict_transf(geometry, to_list=False)
#     V, A = C['values'], C['aperture']
#
#     x, y, z, r = np.zeros((128, 5)), np.zeros((128, 5)), \
#         np.zeros((128, 5)), np.zeros((128, 5))
#     X, Y, Z, R = V['x'], V['y'], V['z'], V['r']  # in m
#
#     # results
#     camera_corners = {
#         'label': 'camera_corners', 'values': None, 'aperture': {}}
#
#     if debug:
#         print(A)
#
#     for i in range(128):
#         for j, L in enumerate(
#                 [[1, 0, 2, 3], [2, 0, 2, 4],
#                  [3, 4, 0, 1], [4, 3, 0, 1]]):
#             if j <= 1:
#                 a, b = -1., 1.
#             else:
#                 a, b, = 1., -1.
#
#             if False:  # debug:
#                 print(i, j, L, a, b,
#                       L[0], L[0].__class__,
#                       x.__class__, X.__class__)
#             #   -----------Xs[:,3]------------
#             # cXs[:,4]                    cXs[:,1]
#             #   |                            |
#             #   |                            |
#             #  Xs[:,1]    cXs[:,0]         Xs[:,2]
#             #   |          Xs[:,0]           |
#             #   |                            |
#             #   |                            |
#             # cXs[:,3]                    cXs[:,2]
#             #   -----------Xs[:,4]------------
#
#             x[:, 0] = X[:, 0]
#             y[:, 0] = Y[:, 0]
#             z[:, 0] = Z[:, 0]
#             r[:, 0] = R[:, 0]
#
#             x[:, L[0]] = X[:, L[1]] + a * X[:, L[2]] + b * X[:, L[3]]  # in m
#             y[:, L[0]] = Y[:, L[1]] + a * Y[:, L[2]] + b * Y[:, L[3]]  # in m
#             z[:, L[0]] = Z[:, L[1]] + a * Z[:, L[2]] + b * Z[:, L[3]]  # in m
#             r[:, L[0]] = R[:, L[1]] + a * R[:, L[2]] + b * R[:, L[3]]  # in m
#
#     for l, camera in enumerate(['HBCm', 'VBCl', 'VBCr']):
#         camera_corners['aperture'][camera] = {
#             'x': np.zeros(5), 'y': np.zeros(5),
#             'z': np.zeros(5), 'r': np.zeros(5)}
#         X_apt, Y_apt, Z_apt, R_apt = A[camera]['x'], \
#             A[camera]['y'], A[camera]['z'], A[camera]['r']
#
#         camera_corners['aperture'][camera]['x'][0] = X_apt[0]  # in m
#         camera_corners['aperture'][camera]['y'][0] = Y_apt[0]  # in m
#         camera_corners['aperture'][camera]['z'][0] = Z_apt[0]  # in m
#         camera_corners['aperture'][camera]['r'][0] = R_apt[0]  # in m
#
#         for j, L in enumerate(
#                 [[1, 0, 2, 3], [2, 0, 2, 4],
#                  [3, 4, 0, 1], [4, 3, 0, 1]]):
#             if j <= 1:
#                 a, b = -1., 1.
#             else:
#                 a, b, = 1., -1.
#
#             if debug:
#                 print(l, camera, j, L)
#             camera_corners['aperture'][camera]['x'][L[0]] = \
#                 X_apt[L[1]] + a * X_apt[L[2]] + b * X_apt[L[3]]
#             camera_corners['aperture'][camera]['y'][L[0]] = \
#                 Y_apt[L[1]] + a * Y_apt[L[2]] + b * Y_apt[L[3]]
#             camera_corners['aperture'][camera]['z'][L[0]] = \
#                 Z_apt[L[1]] + a * Z_apt[L[2]] + b * Z_apt[L[3]]
#             camera_corners['aperture'][camera]['r'][L[0]] = \
#                 R_apt[L[1]] + a * R_apt[L[2]] + b * R_apt[L[3]]
#
#     # done, put in results
#     camera_corners['values'] = {'x': x, 'y': y, 'z': z, 'r': r}  # in m
#
#     if saving:
#         file = loc + 'corner_geometry.json'
#         out = mClass.dict_transf(
#             camera_corners, to_list=True)
#         with open(file, 'w') as outfile:
#             json.dump(out, outfile, indent=4, sort_keys=False)
#         outfile.close()
#
#     # return this dictionary
#     return (camera_corners)
