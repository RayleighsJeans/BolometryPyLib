""" **************************************************************************
    so header """

import numpy as np
import json
import os

import mClass
import factors_geometry_plots as fgp
import mfr_plot as mfrp
import mfr2D_accessoires as mfra
import geometry_factors as geom_facs
import dat_lists as dat_lists
import profile_to_lineofsight as ptlofs
import radiational_fraction as rfrac

Z = np.zeros
ones = np.ones

location = '//share.ipp-hgw.mpg.de/' + \
    'documents/pih/Documents/git/bolometer_mfr/output'

""" end of header
************************************************************************** """


def mfr2D_to_mesh(
        grid_r=Z((14, 100)),
        grid_z=Z((14, 100)),
        nr=14,
        nz=100,
        accuracy=10000,
        extrapolate=0,
        phi=108.0,
        label='tomo_mfr2D_14x100_kani_1.5_0.8_times_3.42s',
        mesh_label='mesh_EIM_beta000_slanted_HBCm_10000_50_10.json',
        extrapoalte=0,
        saving=True,
        debug=False):

    if debug:
        print('\tMesh Construction: making 2D mesh from MFR results')
    file = '../results/INVERSION/MFR/' + str(nr) + 'x' + str(
        nz) + '/mesh_' + label + '_' + str(accuracy) + '_' + \
        str(nz) + '_' + str(0) + '.json'

    if debug:
        print(file, '../results/INVERSION/MESH/' + mesh_label)

    if os.path.isfile(file):
        with open(file, 'r') as infile:
            indict = json.load(infile)
        infile.close()
        mesh = mClass.dict_transf(
            indict, list_bool=False)
        return (mesh)

    elif os.path.isfile('../results/INVERSION/MESH/' + mesh_label):
        with open('../results/INVERSION/MESH/' + mesh_label,
                  'r') as infile:
            indict = json.load(infile)
        infile.close()
        mesh = mClass.dict_transf(
            indict, list_bool=False)
        return (mesh)

    else:
        pass

    mesh = {'label': '2D mesh from VMEC fluxsurfaces',
            'values': {
                'lines': {
                    'z': np.zeros((nz, accuracy)),
                    'r': np.zeros((nz, accuracy))},
                'crosspoints': None,
                'fs_phi': None}}

    # phi of points on FS
    fs_phi = np.zeros(((nr + 1) * nz))
    fs_phi[:] = phi
    magax_dist = [0.0] * nz

    print('\t\tMeshing...')
    nos_mesh = np.zeros((2, (nr + 1) * nz))
    for i in range(nr):
        for j in range(nz):
            if debug:
                print('i/nr:', i, '/', nr - 1, '\tj/nz:', j, '/', nz - 1)

            if j < nz - 1:
                if debug:
                    print('\tN:', (i + 1) * nz + j + 1)

                # same line, fluxsurface
                nos_mesh[:, i * nz + j] = \
                    grid_r[i, j + 1, 0], grid_z[i, j + 1, 0]
                # next line, same fluxsurface
                nos_mesh[:, i * nz + j + 1] = \
                    grid_r[i, j + 1, 1], grid_z[i, j + 1, 1]
                # next line, next fluxsurface
                nos_mesh[:, (i + 1) * nz + j + 1] = \
                    grid_r[i, j + 1, 2], grid_z[i, j + 1, 2]
                # same line, next fluxsurface
                nos_mesh[:, (i + 1) * nz + j] = \
                    grid_r[i, j + 1, 3], grid_z[i, j + 1, 3]

            if i == nr - 2:
                mesh['values']['lines']['z'][j, :] = np.linspace(
                    nos_mesh[1, 0], nos_mesh[1, (i + 1) * nz + j], accuracy)
                mesh['values']['lines']['r'][j, :] = np.linspace(
                    nos_mesh[0, 0], nos_mesh[0, (i + 1) * nz + j], accuracy)

                magax_dist[j] = np.sqrt((
                    nos_mesh[0, (i + 1) * nz + j] - nos_mesh[0, 0]) ** 2 + (
                        nos_mesh[1, (i + 1) * nz + j] - nos_mesh[1, 0]) ** 2)

    # dictionary fill
    mesh['values']['crosspoints'] = nos_mesh
    mesh['values']['fs_phi'] = fs_phi
    mesh['values']['magnetic_axis_distance'] = magax_dist

    # sav the dictionary with lists instead of arrays
    if saving:
        print('\t\tWriting mesh dict to', file)
        outdict = mClass.dict_transf(mesh, list_bool=True)
        with open(file, 'w') as outfile:
            json.dump(outdict, outfile, indent=4, sort_keys=False)
        outfile.close()
        mesh = mClass.dict_transf(outdict, list_bool=False)

    return (mesh)


def mfr2D_factors_profile(
        mesh={'none': None},
        corner_geometry={'none': None},
        lines_of_sight={'none': None},
        nr=14,
        nz=100,
        extrapolate=0,
        label='tomo_mfr2D_14x100_kani_1.5_0.8_times_3.42s',
        input_LoS='lofs_profile_EIM_beta000_slanted_HBCm_50_10.json',
        factors_label='geom_factors_EIM_beta000_slanted_HBCm_50_10.json',
        debug=False):

    camera_info = dat_lists.geom_dat_to_json()
    magnetic_axis = [
        0.0, 0.0, mesh['values']['crosspoints'][1, 0],
        mesh['values']['crosspoints'][0, 0]]
    f = '../results/INVERSION/MFR/' + str(nr) + 'x' + str(
        nz) + '/lofs_profile'

    if debug:
        print('../results/INVERSION/MESH/' + input_LoS,
              '../results/INVERSION/MESH/' + factors_label)

    if os.path.isfile('../results/INVERSION/MESH/' + factors_label):
        with open('../results/INVERSION/MESH/' + factors_label,
                  'r') as infile:
            indict = json.load(infile)
        infile.close()
        factors = mClass.dict_transf(
            indict, list_bool=False)

    else:
        factors = geom_facs.geometrical_and_emissivity_factors(
            corners=corner_geometry, camera_info=camera_info,
            mag_ax=magnetic_axis, saving=True,
            mesh=mesh, linesofsight=lines_of_sight,
            extrapolate=extrapolate, lines=nz, vmec_label=label,
            cartesian=False, pre=f)

    if os.path.isfile('../results/INVERSION/MESH/' + input_LoS):
        with open('../results/INVERSION/MESH/' + input_LoS,
                  'r') as infile:
            indict = json.load(infile)
        infile.close()
        LoS_profile = mClass.dict_transf(
            indict, list_bool=False)

    else:
        LoS_profile = ptlofs.effective_radius_cell(
            mesh=mesh, cartesian=False, vmec_label=label,
            emiss_factors=factors, tor_phi=108.0,
            shift_center=magnetic_axis,
            lines=nz, extrapolate=extrapolate,
            file=f, camera_info=camera_info)

    return (factors, LoS_profile, camera_info)
