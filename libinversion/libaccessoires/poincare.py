""" **************************************************************************
    start of file """

# Created on Thu Aug  4 11:53:37 2016
# @author: tya, thomsen
# Poincare plot of a mag. configuration with CAD-coils

import sys
from os.path import isfile
from osa import Client

import numpy as np

import geometry.w7x.fieldlinetracer as flt

import poincare_plots

""" eo header
************************************************************************** """


def poincare(
        config='lin',
        phi=108.,
        points=10,
        debug=False):

    tracer = flt.TRACER()
    IP = [-15.e3 for i in range(5)]
    IP.extend([0., 0.])
    Itc, Icc = np.zeros(5), np.zeros(10)
    tracer.set_currents(IP, Itc, Icc)

    # O_points = np.array([[  # O-points EJM
    #     5.643, 6.237, 5.643, 5.517, 5.517, 5.950, 6.206, 6.266],
    #     0.927, 0.000,-0.927,-0.610, 0.610, 0.000, 0.025, 0.001]])

    tracer.set_traceinit(
        config='lin', phi=phi, nr=points, res=0.01)
    tracer.poincare_phi(phi)
    res = tracer.trace()[0]

    # reshape to array with S surfaces x N points
    X = np.zeros((
        4, np.shape(res.surfs)[0],
        np.shape(res.surfs[0].points.x1)[0]))

    for i, surf in enumerate(res.surfs):
        X[0, i] = surf.points.x1
        X[1, i] = surf.points.x2
        X[2, i] = surf.points.x3
        X[3, i] = (X[0, i]**2 + X[1, i]**2)**(1/2)

    return (X, res)


def define_divWall(
        name='MeshSrv?wsdl',
        host='http://esb:8280/services/',
        ID1=9,
        ID2=4,
        phi=108.,
        points=None,
        saving=True,
        debug=False):
    file = 'phi' + format(phi, '.1f') + '_id1_' + str(ID1) + \
        '_id2_' + str(ID2)
    if saving:
        points = store_read(arg='divertors_', name=file)
    if points is None:
        # Divertor and PV
        cl = Client(host + name)

        # ID are assembly IDs from ComponentsDB:
        # 9 = CAD vacuum vessel, 4 = divertor
        ref1 = cl.types.DataReference(1)
        ref1.dataId = ID1
        ref2 = cl.types.DataReference(1)
        ref2.dataId = ID2
        sms = cl.types.SurfaceMeshSet(1)
        sms.references = [ref1, ref2]

        phi_in_rad = round(phi * np.pi / 180., 2)
        points = cl.service.intersectMeshPhiPlane(
            phi_in_rad, sms)

        x, y, z, r = [], [], [], []
        for p in points:
            a, b = np.array(p.vertices.x1), np.array(p.vertices.x2)
            x.append(a)
            y.append(b)
            z.append(np.array(p.vertices.x3))
            r.append(np.sqrt(a**2 + b**2))
        points = np.array([x, y, z, r])
        if saving:
            store_read(data=points, arg='divertors_', name=file)
    return (points)


def search_LCMS(
        config='lin',
        resolution=10,
        phi=108.,
        debug=False):

    tracer = flt.TRACER()
    IP = [-15.e3 for i in range(5)]
    IP.extend([0.,0.])
    Itc, Icc = np.zeros(5), np.zeros(10)
    tracer.set_currents(IP,Itc,Icc)

    tracer = flt.TRACER()
    tracer.set_traceinit(
        config='lin', phi=phi, nr=resolution, res=0.01)

    res = tracer.get_separatrix(
        phi=phi, nr=resolution)[0][0]

    X = np.zeros((
        4, np.shape(res.surfs)[0],
        np.shape(res.surfs[0].points.x1)[0]))

    for i, surf in enumerate(res.surfs):
        X[0, i] = surf.points.x1
        X[1, i] = surf.points.x2
        X[2, i] = surf.points.x3
        X[3, i] = (X[0, i]**2 + X[1, i]**2)**(1/2)

    return (X, res)


def get_iota_profile(
        spos=None):  # solution of points
    # Tracer address
    url = 'http://esb.ipp-hgw.mpg.de:8280/services/FieldLineProxy?wsdl'
    tracer = Client(url)

    # definition of start points
    pos = tracer.types.Points3D()
    pos.x1 = spos[0].reshape(-1)
    pos.x2 = spos[1].reshape(-1)
    pos.x3 = spos[2].reshape(-1)

    # Building the mag. configuration from the input coil currents and coils
    config = tracer.types.MagneticConfig()
    config.configIds = [9]

    task = tracer.types.Task()
    task.step = 0.2

    task.characteristics = tracer.types.MagneticCharacteristics()
    task.characteristics.axisSettings = tracer.types.AxisSettings()

    res = tracer.service.trace(pos, config, task, None, None)

    iotaprof = []
    reffprof = []
    for ch in res.characteristics:
        iotaprof.append(ch.iota)
        reffprof.append(ch.reff)

    return (np.array([reffprof, iotaprof]))


def find_island(
        iota_prof=None,
        surfaces=None,
        iota_select=1.06):
    iota = iota_prof[1].reshape(surfaces.shape[1], surfaces.shape[2])
    i0 = np.where(iota == np.min(iota))[0]

    iota0 = iota[i0[0]:, :]

    island0 = np.array(list(set(
        np.where(iota0 >= iota_select)[0])))
    return (surfaces[:, island0, :], surfaces[:, -15:, :])


def main(
        phi=108.,  # 108.
        resolution=10,  # 10+
        config='lin',
        saving=True):  # vacuum cessel
    name = 'phi' + format(phi, '.1f') + '_res' + str(resolution) + \
        '_conf_' + config

    points_phi = store_read(arg='poincare_phi_', name=name)
    if points_phi is None:
        print('\t\\\ tracing confinement points...')
        # Calling Tracer for "Poincare" task inside LCMS
        points_phi, res = poincare(
            config=config, phi=phi,
            points=resolution, debug=False)

        if saving:
            store_read(
                data=points_phi,
                arg='poincare_phi_',
                name=name)

    points_LCMS = store_read(arg='resolution_LCMS_', name=name)
    if points_LCMS is None:
        print('\t\\\ defining positions of LCF/MS')
        # Searching for the LCMS
        points_LCMS, res = search_LCMS(
            config=config, resolution=resolution,
            phi=phi, debug=False)

        if saving:
            store_read(
                data=points_LCMS,
                arg='resolution_LCMS_',
                name=name)

    iota = store_read(arg='iota_', name=name)
    if iota is None:
        print('\t\\\ iota profile for islands')
        # Calling Tracer for "mag. characteristics" task to define iota profile
        iota = get_iota_profile(spos=points_phi)

        if saving:
            store_read(
                data=iota,
                arg='iota_',
                name=name)

    islands1 = store_read(arg='islands1_', name=name)
    islands0 = store_read(arg='islands_iota1.06_', name=name)
    if islands1 is None:
        print('\t\\\ finding islands')
        # Defining internal islands
        islands0, islands1 = find_island(
            iota_prof=iota, surfaces=points_phi,
            iota_select=1.06)

        if saving:
            store_read(
                data=islands1,
                arg='islands1_',
                name=name)

            store_read(
                data=islands1,
                arg='islands_iota1.06_',
                name=name)

    poincare_plots.plot_wrapper_poincare(
        phiP=points_phi, LCMSP=points_LCMS,
        islands=islands1, phi=phi)
    return (iota, islands0, islands1, points_phi)


def store_read(
        data=None,
        arg='poincare_',
        name='test',
        base='../results/INVERSION/POINCARE/',
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

    data = load_store_prop(arg, data)
    return (data)
