""" **************************************************************************
    start of file """

import os
import textwrap

import numpy as np

import requests
import json

import mClass

Z = np.zeros
one = np.ones

""" eo header
************************************************************************** """


def fluxsurfaces(
        FS={'none': None},
        angle_pos=np.linspace(100.0, 115.0, 50),
        vmec_label='EIM_beta000',
        saving=False,
        plot=False):

    # textwrapper, output for chosen fluxsurface
    print('\tCreating fluxsurface library for', vmec_label)
    prefix = '                FLUXSURFACE: '
    wrapper = textwrap.TextWrapper(
        initial_indent=prefix, width=96,
        subsequent_indent=' ' * len(prefix))
    print(wrapper.fill(FS['description']))

    # fluxsurfaces from VMEC
    file_loc = r'../results/INVERSION/FS/' + \
        vmec_label + '_fs_data_nP' + \
        str(np.shape(angle_pos)[0]) + '.json'
    if not os.path.isfile(file_loc) or not saving:
        fsdata_object = vmec(
            poloid=angle_pos, vmecID=FS['URI'],
            vmec_label=vmec_label, saving=saving)

    else:
        print('\t\tLoading from:', vmec_label + '_fs_data.json')
        with open(file_loc, 'r') as infile:
            fsdata_object = json.load(infile)
        infile.close()
        fsdata_object = mClass.dict_transf(
            fsdata_object, to_list=False)

    return (fsdata_object)


def vmec(
        poloid=np.linspace(106., 110.0, 20),
        vmecID='1000_1000_1000_1000_+0000_+0000/10/06_l6ns151',
        vmec_label='EIM_beta000',
        base_URI='http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/',
        saving=False):
    """ getting the VMEC fluxsurfaces based on the ID and the label
    Args:
        poloid (0, ndarray): list toroidal angles used get from archive
        vmecID (1, str): name link for specified magnetic configuration
        vmec_label (2, str): name for saving of the configuration
    Returns:
        fsdata_object (0, dict): loaded FS data object with angles
    """
    print('\t>>Fluxsurfaces for', vmecID)
    # number of toroidal slices
    nPhi = np.shape(poloid)[0]

    # URL where to load from
    URI = base_URI + vmecID + '/'
    FSURI = URI + 'fluxsurfaces.json?phi='

    # set up dictionary
    fsdata_object = {
        'label': 'fluxsurface_data', 'values': {
            'angles': poloid,
            'fs': None, 'phi': None, 'lcfs': None, 'magnetic_axes': None}
    }

    # looping over the specified angle region (toroid.)
    for P, phi in enumerate(poloid):
        # the VMEC (R, Z)-pair to (X, Y, Z)
        theta = -1. * np.cos(np.deg2rad(180. - phi))
        gamma = np.cos(np.deg2rad(phi - 90.))

        # grab via request
        resp = requests.get(FSURI + str(phi)).json()
        if P == 0:
            nFS, nL = \
                np.shape(resp['surfaces'])[0], \
                np.shape(resp['surfaces'][0]['x1'])[0]
            FluxSurf = Z((4, nPhi, nFS, nL))
            tor_phi = Z((nPhi, nFS, nL))

        # for each surface in the VMEC output dump to the arrays
        for S, pos in enumerate(resp['surfaces']):
            FluxSurf[3, P, S, :] = pos['x1']  # R, m
            FluxSurf[2, P, S, :] = pos['x3']  # Z, m
            FluxSurf[0, P, S, :] = [x * theta for x in pos['x1']]  # X, m
            FluxSurf[1, P, S, :] = [y * gamma for y in pos['x1']]  # Y, m

            tor_phi[P, S, :] = phi  # deg

    # dump to the dictionary
    fsdata_object['values']['fs'] = FluxSurf  # in m
    fsdata_object['values']['phi'] = tor_phi  # in deg
    fsdata_object['values']['lcfs'] = FluxSurf[:, :, -1, :]  # in m
    fsdata_object['values']['magnetic_axes'] = FluxSurf[:, :, 0, 0]  # in m

    if saving:  # write file
        outdict = mClass.dict_transf(fsdata_object, to_list=True)
        file_loc = r'../results/INVERSION/FS/' + \
            vmec_label + '_fs_data_nP' + \
            str(np.shape(poloid)[0]) + '.json'
        with open(file_loc, 'w') as outfile:
            json.dump(outdict, outfile, indent=4, sort_keys=False)
            outfile.close()
        fsdata_object = mClass.dict_transf(
            fsdata_object, to_list=False)

    # returns the large dictionary previously filled
    return (fsdata_object)
