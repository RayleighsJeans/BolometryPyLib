
""" **************************************************************************
    so header """

import numpy as np
from osa import Client
import json
import os

import mClass
import mfr_plot as mfrp

Z = np.zeros
ones = np.ones

location = '//share.ipp-hgw.mpg.de/' + \
    'documents/pih/Documents/git/bolometer_mfr/output'

""" end of header
************************************************************************** """


def poincare_in_phiplane(
        phi=107.9,
        points=320,
        step=0.2,
        magconf_ID=0,
        N_3D_points=240,
        shift_space=False,
        saving=True,
        debug=False):

    file = '../results/INVERSION/POINCARE/poincare_' + str(
        phi) + '_p' + str(points) + '_s' + str(
            step) + '_mc' + str(magconf_ID) + '_N3D' + str(N_3D_points)

    if shift_space:
        file += '_shiftYZ'
    file += '.json'

    if os.path.isfile(file):
        with open(file, 'r') as infile:
            indict = json.load(infile)
        infile.close()
        P = mClass.dict_transf(
            indict, list_bool=False)

        mfrp.poincare_scatter_plot(
            poincare=P['values'], phi=phi, points=points,
            step=step, magconf_ID=magconf_ID, N_3D_points=N_3D_points,
            shift_space=shift_space)

        return (P)
    else:
        pass

    P = {
        'label': 'poincare in HBCm plane',
        'values': {'x': None, 'y': None, 'z': None, 'r': None}}

    tracer = Client(
        'http://esb.ipp-hgw.mpg.de:8280/services/FieldLineProxy?wsdl')

    pos = tracer.types.Points3D()
    pos.x1 = np.linspace(5.6, 6.2, N_3D_points)  # X in m

    if shift_space:
        if True:
            x, y, z, r = island_prep(
                phi=phi, N_3D_points=N_3D_points)
            pos.x1, pos.x2, pos.x3 = x, y, z

        else:
            pos.x1 = np.linspace(-1.4, -1.8, N_3D_points)  # X in m
            pos.x2 = np.linspace(4.3, 5.7, N_3D_points)  # Y
            pos.x3 = np.linspace(-.7, .7, N_3D_points)  # Z

    else:
        pos.x2 = np.zeros(N_3D_points)  # Y
        pos.x3 = np.zeros(N_3D_points)  # Z

    config = tracer.types.MagneticConfig()
    config.configIds = [magconf_ID]

    # 164 w7x
    # Rectangular torus to protect from
    # going outside of the standard W7X mesh:
    # (4.05, 6.75, -1.35, 1.35) with padding of 1 cm
    machine = tracer.types.Machine(1)
    # Rectangular torus model:
    machine.meshedModelsIds = [164]

    poincare = tracer.types.PoincareInPhiPlane()
    poincare.numPoints = points
    poincare.phi0 = [np.deg2rad(phi)]
    poincare.noCrossLimit = 500.

    task = tracer.types.Task()
    task.step = step
    task.poincare = poincare

    res = tracer.service.trace(pos, config, task, None, None)
    if debug:
        print(np.shape(res.surfs))
        print(np.shape(res.surfs[0]))
        print(np.shape(res.surfs[0].points))
        print(np.shape(res.surfs[0].points.x1))

    for i in range(np.shape(res.surfs)[0]):
        if i == 0:
            for label in ['x', 'y', 'z', 'r']:
                P['values'][label] = [None] * np.shape(res.surfs)[0]
        P['values']['x'][i] = res.surfs[i].points.x1
        P['values']['y'][i] = res.surfs[i].points.x2
        P['values']['z'][i] = res.surfs[i].points.x3
        if res.surfs[i].points.x1 is not None and \
                res.surfs[i].points.x2 is not None:
            P['values']['r'][i] = [
                np.sqrt(x ** 2 + res.surfs[i].points.x2[j] ** 2) for
                j, x in enumerate(res.surfs[i].points.x1)]

    if saving:
        print('\t\tWriting ...')
        outdict = mClass.dict_transf(
            P, list_bool=True)
        with open(file, 'w') as outfile:
            json.dump(outdict, outfile,
                      indent=4, sort_keys=False)
        outfile.close()
        P = mClass.dict_transf(
            P, list_bool=False)

    mfrp.poincare_scatter_plot(
        poincare=P['values'], phi=phi, points=points,
        step=step, magconf_ID=magconf_ID, N_3D_points=N_3D_points,
        shift_space=shift_space)

    return (P)


def island_prep(
        phi=107.9,
        N_3D_points=100,
        debug=True):

    r_bl, z_bl = np.linspace(4.85, 4.95, N_3D_points), \
        np.linspace(-0.575, -0.5, N_3D_points)
    x_bl, y_bl = np.zeros((N_3D_points)), np.zeros((N_3D_points))
    i = 0
    for a, b in zip(r_bl, z_bl):
        x_bl[i], y_bl[i] = ptl.coordinate_transform_mesh(
            polygon=[[a, b], [a, b]], phi=phi)[:2]
        i += 1

    r_l, z_l = np.linspace(4.58, 4.62, N_3D_points), \
        np.linspace(-0.05, 0.05, N_3D_points)
    x_l, y_l = np.zeros((N_3D_points)), np.zeros((N_3D_points))
    i = 0
    for a, b in zip(r_l, z_l):
        x_l[i], y_l[i] = ptl.coordinate_transform_mesh(
            polygon=[[a, b], [a, b]], phi=phi)[:2]
        i += 1

    r_tl, z_tl = np.linspace(4.85, 4.95, N_3D_points), \
        np.linspace(-0.575, -0.5, N_3D_points)
    x_tl, y_tl = np.zeros((N_3D_points)), np.zeros((N_3D_points))
    i = 0
    for a, b in zip(r_tl, z_tl):
        x_tl[i], y_tl[i] = ptl.coordinate_transform_mesh(
            polygon=[[a, b], [a, b]], phi=phi)[:2]
        i += 1

    x, y, z, r = np.append(x_bl, x_l), np.append(
        y_bl, y_l), np.append(z_bl, z_l), np.append(r_bl, r_l)

    x, y, z, r = np.append(x, x_tl), np.append(
        y, y_tl), np.append(z, z_tl), np.append(r, r_tl)

    return (x, y, z, r)
