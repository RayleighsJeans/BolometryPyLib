""" **************************************************************************
    so header """

import json
import numpy as np
import matplotlib.pyplot as p
import scipy
import os

from scipy.stats import multivariate_normal
import mClass as mClass

scipy.seterr(all='print')

Z = np.zeros
ones = np.ones

""" end of header
************************************************************************** """


def surrogate_radiation_profile(
        mesh2d={'none': None},
        lines=36,
        extp_nmb=4,
        fluxsurfaces={'none': None},
        fsnumber=32,
        rad_spots=Z((5, 2)),
        gauss_sigmas=Z((5)),
        plot_bool=True,
        cartesian=True):
    """ create a surrogate radiation profile on the mesh from before
    Args:
        mesh2d (0, dict): dictionary of mesh grid from before
        lines (1, int): number of horiz./vert./poloid. lines
        extp_nmb (2, int): fluxsurfaces beyond LCFS
        fluxsurfaces (3, dict): dictionary of fluxsurface info
        fsnumber (4, int): number of selected (prob. 108°) toroidal angle
        rad_spots (5, ndarray): float locations of radiaton hot spots
        gauss_sigmas (6, ndarray): float sigma values of gauss distribution
        plot_bool (7, bool): show me?
        cartesian (8, bool): is the mesh cartesian metric?
    Returns:
        rad_dict (0, dict): Dictionary of radiation profile results
    """
    print('\t>>Surrogate radiation profiles\n',
          '\t\tPlacing', len(rad_spots),
          'multivariate normalized gaussian spots around fluxsurfaces')
    # grab stuff from the imported/parsed properties
    mesh = np.array(mesh2d['values']['crosspoints'])
    # sorted_mesh = np.array([np.sort(mesh[0, :]),
    #                        mesh[1, np.argsort(mesh[0, :])]])
    FS = fluxsurfaces['values']['fs'][fsnumber]
    rK = np.linspace(mesh[0, :].min(), mesh[0, :].max(), len(mesh[0, :]))
    zK = np.linspace(mesh[1, :].min(), mesh[1, :].max(), len(mesh[1, :]))
    RK, ZK = np.meshgrid(rK, zK)

    # leftover mesh variables
    # RFS, ZFS = np.meshgrid(sorted_mesh[0, :], sorted_mesh[1, :])
    RCIRC, ZCIRC = np.meshgrid(mesh[0, :], mesh[1, :])

    # mesh stuff for the pdf gauss
    RR, ZZ = [RK, ZK]
    print('\t\tGot RR,ZZ')
    radiation = np.zeros((np.shape(RR)))
    rad = np.zeros((np.shape(RR)))
    pos = np.empty(RR.shape + (2, ))
    # the positions over which the pdf is calculated
    pos[:, :, 0] = RR
    pos[:, :, 1] = ZZ

    # every single spot is stored in spots
    spots = [np.zeros(np.shape(RR))] * len(rad_spots)
    for source in range(0, len(rad_spots)):
        # each spot is calculated
        spots[source] = multivariate_normal.pdf(x=pos,
                                                mean=rad_spots[source],
                                                cov=gauss_sigmas[source])
    # now, the spots and the overall pdf have to be normalized
    for spot in range(0, len(spots)):
        if spot == 0:
            rad = spots[spot]
        elif spot > 0:
            # normalization
            # scale through the combined and single maximum
            rad = spots[spot] * radiation.max() / spots[spot].max()
        radiation += rad
    # scale... what happened scale? ... SCAAAAAAAAAALE ?!
    radiation = radiation / radiation.max()

    # prepare figure
    if plot_bool:
        fig = p.figure()
        ax = fig.add_subplot(111)

        # plot the mesh, whether cartesian/fluxsurface bound
        line_vecs = np.array([mesh2d['values']['lines']['r'],
                              mesh2d['values']['lines']['z']])
        if not cartesian:

            for k in range(0, len(FS) - 2 + extp_nmb):
                ax.plot(np.append(mesh[0, k * lines:(k + 1) * lines],
                                  mesh[0, k * lines]),
                        np.append(mesh[1, k * lines:(k + 1) * lines],
                                  mesh[1, k * lines]), linewidth=0.2, c='k')

            for l in range(0, lines):
                ax.plot(
                    line_vecs[0, l], line_vecs[1, l], c='k', linewidth=0.5)

        elif cartesian:
            for l in range(0, 2 * lines):
                ax.plot(
                    line_vecs[0, l], line_vecs[1, l], c='k', linewidth=0.5)

        # LCFS
        ax.plot(FS[len(FS) - 2][3], FS[len(FS) - 2][2], c='r', linewidth=1.)

        # radiation profile
        ax.contourf(RR, ZZ, radiation, 10, map='hot')
        ax.set_xlim(mesh[0, :].min(), mesh[0, :].max())
        ax.set_ylim(mesh[1, :].min(), mesh[1, :].max())
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')

    # the dictionary to be saved
    rad_dict = {'name': 'radiation matrix over mesh',
                'values': {
                    'radiation_spots': rad_spots,
                    'distribution_sigmas': gauss_sigmas,
                    'position': pos,
                    'radiation_distribution': radiation}
                }

    # where to write
    file = r'../results/INVERSION/SURROGATES/radiation_' + \
        str(len(rad_spots))
    if cartesian:
        file += '_cartesian'
    file += '_spots.json'
    print('\t\tWriting results to', file)

    # check if file
    if not os.path.isfile(file):
        dumpdict = mClass.dict_transf(prints=False,
                                      dictionary=rad_dict, list_bool=True)
        with open(file, 'w') as outfile:
            json.dump(dumpdict, outfile, indent=4, sort_keys=False)
        outfile.close()
    elif os.path.isfile(file):
        print('\t\tGot radiation profile already')

    # write figure
    if plot_bool:
        file_loc = r'../results/INVERSION/SURROGATES/radiaton_' + \
            str(len(rad_spots))
        if cartesian:
            file_loc += '_cartesian'
        file_loc += '_spots.pdf'
        print('\t\tWriting image to', file_loc)
        fig.savefig(file_loc)
        p.close('all')

    # return the important stuff
    print('\t\tSurrogate gaussian radiation spots created')
    return rad_dict
