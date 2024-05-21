""" **************************************************************************
    start of file """

import sys
from os.path import isfile

import numpy as np
from scipy import interpolate

""" eo header
************************************************************************** """


def setup3D_mesh(
        fluxsurfaces={'none': None},
        vpF=1.3,
        nFS=25,
        nL=125,
        nPhi=80,
        vmec_label='test',
        saving=True,
        debug=False):

    def get_vector_lcfs(
            fs_center=np.zeros((3)),
            fs_lcfs=np.zeros((3, 80))):
        return(np.array([
            fs_lcfs[0] - fs_center[0],
            fs_lcfs[1] - fs_center[1],
            fs_lcfs[2] - fs_center[2]]))

    def interp_lcfs(
            fs_lcfs=np.zeros((3, 80)),
            nL=125):
        tck, u = interpolate.splprep(fs_lcfs, s=0)
        return (interpolate.splev(
            np.linspace(.0, 1., nL), tck))

    def bloated_fs(
            center,  # 3 coords
            vectors,  # 3 coords, nL points
            F_volume):  # 1.3
        return (np.array([
                F_volume * vectors[0] + center[0],
                F_volume * vectors[1] + center[1],
                F_volume * vectors[2] + center[2]]))

    label = vmec_label + '_' + str(nPhi) + 'x' + \
        str(nFS) + 'x' + str(nL) + '_' + str(vpF)

    print('\tMeshing in 3D x' + str(vpF) +
          ' inflated [phi x FS x L]:',
          nPhi, 'x', nFS + 1, 'x', nL + 1, '...')

    if saving:
        fs_object = store_read_mesh3D(name=label)
        if fs_object is not None:
            print('\t\\\ final calculation label:', label)
            return (fs_object, label)

    FS = fluxsurfaces['values']['fs']
    fs_object = np.zeros((4, nPhi, nFS + 1, nL + 1))

    for P in range(nPhi):  # nAngles
        # interpolate lcfs into nL
        fs_lcfs = interp_lcfs(fs_lcfs=FS[:3, P, -1], nL=nL + 1)
        # center point
        center = FS[:3, P, 0, 0]
        # vectors to lcfs
        v_lcfs = get_vector_lcfs(  # 3 coords, n_points
            fs_center=center, fs_lcfs=fs_lcfs)
        # inflate lcfs by factor with vectors
        max_fs = bloated_fs(
            center=center, F_volume=vpF, vectors=v_lcfs)
        # get new vectors
        v_max = get_vector_lcfs(  # 3 coords, n_points
            fs_center=center, fs_lcfs=max_fs)

        for S in range(nFS + 1):
            fs_object[:3, P, S] = np.array([
                center[0] + S / (nFS + 1) * v_max[0],
                center[1] + S / (nFS + 1) * v_max[1],
                center[2] + S / (nFS + 1) * v_max[2]])

    fs_object[3] = np.sqrt(fs_object[0]**2 + fs_object[1]**2)

    if saving:
        store_read_mesh3D(fs_object, name=label)
    print('\t\\\ final calculation label:', label)
    return (fs_object, label)


def store_read_mesh3D(
        data=None,  # np.zeros((128)),
        name='test',
        base='../results/INVERSION/MESH/',
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

    data = load_store_prop('mesh3D_', data)
    return (data)  # line sections