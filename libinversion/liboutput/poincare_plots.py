""" **************************************************************************
    so header """

import matplotlib.pyplot as p

import numpy as np
from osa import Client
import requests

# database
import database_inv as database
import poincare
database = database.import_database()
VMEC_URI = 'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/'

Z = np.zeros
ones = np.ones

""" end of header
************************************************************************** """


def PlotPoincare(
        axis=None,
        surfaces=None,
        col='w',
        size=.25,
        style='-',
        marker='None',
        width=0.5,
        alpha=.5):

    if axis is None:
        fig, axis = p.subplots(1, 1)

    for S in range(np.shape(surfaces)[1]):
        tmp = np.array((
            surfaces[0, S, :],
            surfaces[1, S, :],
            surfaces[2, S, :])).T

        r = np.sqrt(tmp[:, 0]**2 + tmp[:, 1]**2)
        z = tmp[:, 2]
        axis.plot(
            r, z, marker=marker, ls=style,
            color=col, markersize=size,
            alpha=alpha, lw=width)

    return (axis)


def plot_div(
        axis=None,
        points=None,
        lw=.75,
        alpha=1.,
        ls='-.',
        c='k',
        phi=108.):
    if axis is None:
        fig = p.figure()
        axis = fig.add_subplot(111)
    if points is None:
        # Divertor and PV
        points = poincare.define_divWall(
            phi=phi, saving=False)
    # finally plot
    for r, z in zip(points[3], points[2]):
        axis.plot(
            r, z, c=c, ls=ls,
            lw=lw, alpha=alpha)
    return (axis)


def plot_wrapper_poincare(
        phi=108.,
        phiP=None,
        LCMSP=None,
        islands=None,
        VMID='EIM_000',
        ax=None):

    if ax is None:
        fig = p.figure()
        ax = fig.add_subplot(111)

        ax.set_title('poincare plot, $\\varphi$=' + format(phi, '.1f'))
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
    
        VMEC_ID = database['values']['magnetic_configurations'][VMID]['URI']
        URI = VMEC_URI + VMEC_ID + '/lcfs.json?phi=108.'
        LCFS = requests.get(URI).json()
    
        ax.plot(
            LCFS['lcfs'][0]['x1'], LCFS['lcfs'][0]['x3'],
            c='lightgrey', lw=1., alpha=0.5, ls='-.')

    if phiP is not None:
        PlotPoincare(axis=ax, surfaces=phiP[:, :50], col='grey')
    if LCMSP is not None:
        PlotPoincare(axis=ax, surfaces=LCMSP, col='r')
    if islands is not None:
        PlotPoincare(axis=ax, surfaces=islands, col='g')

    plot_div(axis=ax, phi=phi)
    return
