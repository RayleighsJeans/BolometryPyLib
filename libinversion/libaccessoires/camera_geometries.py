""" **************************************************************************
    start of file """

import os
import json
import sys

import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from scipy.spatial import Delaunay as triangulate

import mClass
import dat_lists
import LoS_emissivity3D as LoS3D

import geometry_plots as gp
import line_of_sight_plots as lofsp
import transf_new_geom as tng

Z = np.zeros
one = np.ones
line = np.linspace

""" eo header
************************************************************************** """


def c_geom(
        N=[8, 2, 4],
        random_error=False,
        error_scale=0.0001,  # 0.1 mm (0.5 mm)
        tilt_deg=5.,
        interp_method='square',
        new_type='VBCm',
        vmec_label='EIM_beta0000',
        fix_LoS=False,
        centered=False,
        symmetric=False,
        tilt=False,
        artificial=True,
        add_camera=False,
        plot=False,
        saving=False,
        debug=False):
    """ doing camera geometry according to plasma geometry
    Keyword Arguments:
        fluxsurfaces {dict} -- flux geometry (def.: {})
    Returns:
        [type] -- [description]
    """
    camera_info = dat_lists.geom_dat_to_json(saving=saving)
    # super-routine which throws back camera corners
    corner_geometry = tng.transf_new_geom(
        compare=False, saving=True, debug=False)

    if add_camera:  # add entirely new camera to array
        if new_type == 'VBCm':  # mirrored vertical camera
            deg, inc = -200., 5.
        if new_type == 'MIRh':  # mirrored horzontal
            open_angle = 90.  # opening angle of array
            deg = 180.  # - open_angle / 2. # once around
            inc = open_angle / 15.  # symmetric

        corner_geometry, vmec_label = add_new_camera(
            debug=debug, corners=corner_geometry, vmec_label=vmec_label,
            saving=saving, camera_info=camera_info,
            new_type=new_type, deg=deg, inc=inc)
        cams = ['HBCm', 'VBCl', 'VBCr', 'ARTf']  # added entirely new cam

    else:  # if no new camera, only based
        cams = ['HBCm', 'VBCl', 'VBCr']

    if artificial:  # construct artificially symmetric HBCm
        centered = random_aperture_error = fix_LoS = symmetric = tilt = False
        corner_geometry, vmec_label = artificial_HBCm(
            camera_info=camera_info, corners=corner_geometry,
            vmec_label=vmec_label, saving=saving, debug=debug)

    if centered or random_error:
        corner_geometry, vmec_label = trafo_dets_aptplane(
            center=centered, randomize_error=random_error,
            error_scale=error_scale,  # 0.1 mm (0.5 mm)
            camera_info=camera_info, corners=corner_geometry,
            vmec_label=vmec_label, saving=saving, debug=debug)

        if random_error:
            fix_LoS = tilt = False
            # cant have aperture errors and fix the LoS
            # or tilt them after, doesn't add up :(

    if fix_LoS:
        corner_geometry, vmec_label = fix_rotate_boloplane(
            camera_info=camera_info, corners=corner_geometry,
            vmec_label=vmec_label, saving=saving, debug=debug)
        symmetric = tilt = False  # you can't have both

    if symmetric or tilt:
        corner_geometry, vmec_label = symmetrical_det_aptplane(
            camera_info=camera_info, corners=corner_geometry,
            vmec_label=vmec_label)

        if tilt:
            corner_geometry, vmec_label = aperture_axis_tilt(
                camera_info=camera_info, corners=corner_geometry,
                vmec_label=vmec_label, deg=tilt_deg)

    if plot:
        gp.detector_plot(
            corners=corner_geometry, geometry=corner_geometry,
            label=vmec_label, add_camera=add_camera, debug=debug,
            camera_info=camera_info, projection='3d')

    # interpolating the resulting, final detectors
    if interp_method == 'triang':
        # triangulate detectors and prepare for lofs
        interp_geometry, vmec_label, N = triangulate_detector_slit(
            debug=debug, corners=corner_geometry, vmec_label=vmec_label,
            saving=saving, camera_info=camera_info, cams=cams, N=N)

    elif interp_method == 'square':
        # split detectors and pinhole in squares
        interp_geometry, vmec_label, N = split_detector_slit(
            N=N, corners=corner_geometry, vmec_label=vmec_label,
            saving=saving, camera_info=camera_info, cams=cams, debug=debug)

    lines_of_sight = lofs_geometry(
        camera_info=camera_info, corner_geometry=interp_geometry,
        base_geometry=corner_geometry, add_camera=add_camera,
        label=vmec_label, cams=cams, M=N, new_type=new_type,
        debug=debug, saving=saving, overwrite=False)

    if plot:
        lofsp.simple_lines_of_sight(
            label=vmec_label, corners=interp_geometry,
            cameras=cams, camera_info=camera_info,
            projection='2d', lines_of_sight=lines_of_sight)

        if debug:
            lofsp.compare_viewcone(
                corners=corner_geometry, N=N, LoS=lines_of_sight,
                nCh=camera_info['channels']['eChannels']['VBCl'],
                cam='VBCl', projection='2d', det_corners=[0])

    return (
        camera_info, corner_geometry, interp_geometry,
        lines_of_sight, vmec_label, N)


def add_new_camera(
        camera_info={'none': None},
        corners={'none': None},
        vmec_label='EIM_beta0000',
        loc='../results/INVERSION/CAMGEO/',
        new_type='VBCm',  # mirrored vbcl, MIRh
        deg=15.,
        inc=1.,
        saving=True,
        debug=False):
    """ adding entirely new artificial camera ARTf
    Args:
        camera_info (dict): Channels and cameras.
        corners (dict): Detector/Aperture corner geometry.
        vmec_label (str): Magnetic configuration and geometry.
        loc (str, optional): Saving
        new_type (str, optional): Defaults to 'VBCm'.
        inc (np.float, optional): Per channel increase in opening angle.
        saving (bool, optional): Saving. Defaults to True.
        debug (bool, optional): Printing. Defaults to False.
    Returns:
        Corner geometry with new camera.
    """
    print('\t\tSetting up new camera: ' + new_type + ' (12 ch)...')

    cor = np.array([  # old geometry
        corners['values']['x'], corners['values']['y'],  # in m
        corners['values']['z'], corners['values']['r']])  # in m
    added_cor = {  # results of new fan
        'label': 'camera_corners', 'values': None, 'aperture': {}}

    # data storage for later
    pos = np.zeros((4, 128, 5))
    # take VBCl camera
    apt = corners['aperture']
    new_ca = np.zeros((4, 5))

    if (new_type == 'VBCm'):  # mirrored and tilted VBCl
        tag = '_addedARTv'
        # aperture axis of VBCl pinhole
        ca = np.array([  # aperture corners
            apt['VBCl']['x'], apt['VBCl']['y'],  # in m
            apt['VBCl']['z'], apt['VBCl']['r']])  # in m
        # VBCl channels
        nCh = camera_info['channels']['eChannels']['VBCl']
        ch0, ch1 = np.min(nCh), np.max(nCh)

        new_ca[:, 0] = ca[:, 0]  # center point the same (or all)
        new_ca[2, 0] = -ca[2, 0]  # flip to up

    elif (new_type == 'MIRh'):  # mirrored HBCm but spread more
        tag = '_addedARTm'
        # aperture axis of HBCm pinhole
        ca = np.array([  # aperture corners
            apt['HBCm']['x'], apt['HBCm']['y'],  # in m
            apt['HBCm']['z'], apt['HBCm']['r']])  # in m
        nCh = camera_info['channels']['eChannels']['HBCm']
        ch0, ch1 = np.min(nCh), np.max(nCh)

        new_ca[3, 0] = 4.35  # R in m
        new_ca[2, 0] = 0.0   # Z in m
        theta = np.arccos(new_ca[2, 0] / new_ca[3, 0])  # pol phi
        # transforming in 3D
        new_ca[0, 0] = new_ca[3, 0] * np.sin(theta) * np.cos(np.deg2rad(108.))
        new_ca[1, 0] = new_ca[3, 0] * np.sin(theta) * np.sin(np.deg2rad(108.))

    rectangle = np.array([  # spanning detector fan plane for axis
        [ca[0, 0], cor[0, ch0, 0], cor[0, ch1, 0], ca[0, 0]],   # in m
        [ca[1, 0], cor[1, ch0, 0], cor[1, ch1, 0], ca[1, 0]],   # in m
        [ca[2, 0], cor[2, ch0, 0], cor[2, ch1, 0], ca[2, 0]]])  # in m
    # plane normal becomes axis to turn around
    axis = v_norm(LoS3D.get_normal(rectangle=rectangle))[0]

    for k in range(1, 5):  # each corner vector has to be turned
        vec = ca[:3, k] - ca[:3, 0]
        v = vec  # np.dot(R.from_rotvec(  # by angle and same axis
        #     np.deg2rad(deg) * axis).as_dcm(), vec)
        new_ca[:3, k] = new_ca[:3, 0] + v
        # get R component at the end for each
        new_ca[3, k] = np.sqrt(new_ca[0, k] ** 2 + new_ca[1, k] ** 2)

    added_cor['aperture']['ARTf'] = {
        'x': None, 'y': None, 'z': None, 'r': None}
    for i, lab in enumerate(['x', 'y', 'z', 'r']):
        added_cor['aperture']['ARTf'][lab] = new_ca[i]

    N = np.shape(camera_info['channels']['eChannels']['ARTf'])[0]
    for n, c in enumerate(camera_info[
            'channels']['eChannels']['ARTf'][:]):

        if (new_type == 'VBCm'):
            nc = camera_info['channels']['eChannels']['VBCl'][-(n + 1)]
        elif (new_type == 'MIRh'):
            nc = camera_info['channels']['eChannels']['HBCm'][2 * n]

        if new_type != 'MIRh':
            alpha = deg + n * inc  # symmetric around 90. deg
        else:
            alpha = deg - (N * inc) / 2. + n * inc

        vec = np.array([  # reverse vector to detector
            [cor[0, nc, 0] - ca[0, 0]],  # in m
            [cor[1, nc, 0] - ca[1, 0]],  # in m
            [cor[2, nc, 0] - ca[2, 0]]])  # in m
        rev_vec = vec  # [vec[0], vec[1], vec[2]]
        rot = np.dot(R.from_rotvec(  # turn that again
            np.deg2rad(alpha) * axis).as_dcm(), rev_vec)
        if new_type == 'MIRh':
            rot = rot * 0.25

        diff_v = np.array([  # difference to transposition the detector
            [(new_ca[0, 0] + rot[0]) - cor[0, nc, 0]],  # in m
            [(new_ca[1, 0] + rot[1]) - cor[1, nc, 0]],  # in m
            [(new_ca[2, 0] + rot[2]) - cor[2, nc, 0]]])  # in m

        for korner in range(0, 5):  # every corner has to be moved
            for i in range(0, 3):
                pos[i, c, korner] = cor[i, nc, korner] + diff_v[i]
            pos[3, c, korner] = np.sqrt(
                pos[0, c, korner] ** 2 + pos[1, c, korner] ** 2)

    for A, cam in enumerate(['HBCm', 'VBCr', 'VBCl']):
        added_cor['aperture'][cam] = apt[cam]
        for n, c in enumerate(camera_info[
                'channels']['eChannels'][cam][:]):
            for i in range(0, 4):
                pos[i, c, :] = cor[i, c, :]  # in m

    # done, put in results
    added_cor['values'] = {
        'x': pos[0], 'y': pos[1], 'z': pos[2], 'r': pos[3]}
    if saving:  # save to new file and return the new one to use
        file = loc + 'corner_geometry' + tag + '.json'
        out = mClass.dict_transf(
            added_cor, to_list=True)
        with open(file, 'w') as outfile:
            json.dump(out, outfile, indent=4, sort_keys=False)
        outfile.close()

    return (added_cor, vmec_label + tag)


def artificial_HBCm(
        camera_info={'none': None},
        corners={'none': None},
        vmec_label='EIM_beta0000',
        loc='../results/INVERSION/CAMGEO/',
        saving=True,
        debug=False):
    """ create artificial horizontal camera array
    Args:
        camera_info (dict): List of channels and cameras.
        corners (dict): Corner geometry of camera.
        vmec_label (str): Label of magnetic configuration and geometry.
        loc (str): Saving. Defaults to '../results/INVERSION/CAMGEO/'.
        saving (bool, optional): Saving. Defaults to True.
        debug (bool, optional): Printing. Defaults to False.
    Returns:
        New corner geometry with artificial HBCm.
    """
    print('\t\tSetting up artificially symmetric horizontal camera...')

    cor = np.array([  # old geometry
        corners['values']['x'], corners['values']['y'],
        corners['values']['z'], corners['values']['r']])
    artific_cor = {  # results of new fan
        'label': 'camera_corners', 'values': None, 'aperture': {}}

    # data storage for later
    pos = np.zeros((4, 128, 5))

    apt = corners['aperture']
    cam = 'HBCm'
    # aperture axis of camera pinhole
    ca = np.array([  # aperture corners
        apt[cam]['x'], apt[cam]['y'],
        apt[cam]['z'], apt[cam]['r']])

    new_ca = np.zeros((4, 5))
    new_ca[:, 0] = ca[:, 0]  # center point the same (or all)
    new_ca[2, 0] = 0.0  # reset to zero

    #  aperture rectangle
    #   1             2
    #  x*************x
    #  * \    |    / *
    #  *  \r  h   r/ *
    #  *   \  |  /   *
    #  *    \ | /    *
    #  *     \|/     *
    #  *-- d--x--d --*
    #  *     /|\     *
    #  *    / | \    *
    #  *  r/  | r\   *
    #  *  /   h   \  *
    #  * /    |    \ *
    #  x*************x
    #   3             4

    h = v_norm((ca[:3, 1] - ca[:3, 0]) + (ca[:3, 2] - ca[:3, 0]))[1]
    d = v_norm((ca[:3, 2] - ca[:3, 0]) + (ca[:3, 3] - ca[:3, 0]))[1]

    #  o-----------------------ca[0,0]->x
    #  | -
    #  |   -
    #  |     -   alpha
    #  |       -
    #  | beta    -
    #  | (phi)     -  sqrt(ca[0,0]^2 + ca[1,0]^2)
    #  |             -
    #  |               -         x p1
    #  |                 -  90° /
    #  |                   -   / d
    # ca[1,0]                -X  p0
    #  |                 90° / d
    #  |                    /
    #  |                   x p2
    #  y
    #
    # x = r * cosp
    # y = r * sinp

    def poloidal_transf(
            offset=np.zeros((3)),
            angle=0.0,
            radius=1.0,
            height=1.0,
            debug=False):
        P = np.zeros((3))
        P[0] = offset[0] + radius * np.cos(angle)
        P[1] = offset[1] + radius * np.sin(angle)
        P[2] = offset[2] + height
        return P

    v_vec = v_norm(np.array([.0, .0, 1.]))[0]
    x = ca[:3, 1] - ca[:3, 0]
    h_vec = v_norm(np.array([x[0], x[1], .0]))[0]

    alpha = np.arccos(
        ca[0, 0] / np.sqrt(ca[0, 0] ** 2 + ca[1, 0] ** 2))
    phi = alpha + np.pi / 2.

    new_ca[:3, 1] = new_ca[:3, 0] + h / 2 * v_vec + d / 2 * h_vec
    new_ca[:3, 2] = new_ca[:3, 0] + h / 2 * v_vec + -d / 2 * h_vec
    new_ca[:3, 3] = new_ca[:3, 0] + -h / 2 * v_vec + -d / 2 * h_vec
    new_ca[:3, 4] = new_ca[:3, 0] + -h / 2 * v_vec + d / 2 * h_vec

    for k in [0, 1, 2, 3, 4]:  # each corner vector has to transitioned
        # r coordinate is calculated from the others
        new_ca[3, k] = np.sqrt(new_ca[0, k] ** 2 + new_ca[1, k] ** 2)

    for C in ['HBCm', 'VBCr', 'VBCl']:
        artific_cor['aperture'][C] = {
            'x': None, 'y': None, 'z': None, 'r': None}
        for i, lab in enumerate(['x', 'y', 'z', 'r']):
            if C == 'HBCm':
                artific_cor['aperture'][C][lab] = new_ca[i]
            else:
                artific_cor['aperture'][C][lab] = apt[C][lab]

    # number of channels and where to stop flipping
    nCh = np.shape(camera_info['channels']['eChannels'][cam])[0]
    Ch = camera_info['channels']['eChannels'][cam]

    r = np.mean([v_norm(cor[:3, x, 0] - ca[:3, 0])[1] for x in Ch])
    D = np.mean([v_norm(cor[:3, x, 2] - cor[:3, x, 1])[1] for x in Ch])
    H = np.mean([v_norm(cor[:3, x, 3] - cor[:3, x, 2])[1] for x in Ch])
    L = np.sqrt((D / 2) ** 2 + (H / 2) ** 2) / 2.

    def spherical_transf(
            offset=np.zeros((3)),
            angle1=0.0,
            angle2=np.pi,
            radius=1.0,
            debug=False):
        P = np.zeros((3))
        P[0] = offset[0] + radius * np.sin(angle1) * np.cos(angle2)
        P[1] = offset[1] + radius * np.sin(angle1) * np.sin(angle2)
        P[2] = offset[2] + radius * np.cos(angle1)
        return P

    opening_angle = 52.5
    phi = phi - np.pi / 2.
    theta_lin = np.linspace(
        90. + opening_angle / 2., 90. - opening_angle / 2., nCh)
    for n, c in enumerate(camera_info[  # move all the detectors likewise
            'channels']['eChannels'][cam][:]):

        #        |  spherical coordinates with theta
        #        |  and phi as angles and r as radius
        #        |
        #        |  sey up detectors according to previous
        #        |  aperture corners
        #        |              x
        #        |_____________   x
        #       /                   x
        #      /         X          x
        #     /         apt        x
        #    /                   x
        #   /                 x
        #   x = r sint cosp
        #   y = r sint sinp
        #   z = r cost

        theta = np.deg2rad(theta_lin[n])
        gamma = np.deg2rad(60.)  # shaping of the detector opening
        pos[:3, c, 0] = spherical_transf(
            offset=new_ca[:3, 0], angle1=theta, angle2=phi, radius=r)
        # by transformation, cant actually shape detector, but absorber plane
        pseudo_det = np.zeros((3, 4))

        x0 = pos[:3, c, 0]  # based
        pseudo_det[:, 0] = spherical_transf(
            offset=x0, angle1=gamma - np.pi / 2.,
            angle2=phi + np.pi / 2., radius=L)
        pseudo_det[:, 1] = spherical_transf(
            offset=x0, angle1=gamma + np.pi / 2.,
            angle2=phi - np.pi / 2., radius=L)
        pseudo_det[:, 2] = spherical_transf(
            offset=x0, angle1=gamma + np.pi / 2.,
            angle2=phi + np.pi / 2., radius=L)
        pseudo_det[:, 3] = spherical_transf(
            offset=x0, angle1=gamma - np.pi / 2.,
            angle2=phi - np.pi / 2., radius=L)
        # vectors vor up and down of detector
        v_vec = v_norm((pseudo_det[:3, 0] - x0) + (pseudo_det[:3, 1] - x0))[0]
        h_vec = v_norm((pseudo_det[:3, 1] - x0) + (pseudo_det[:3, 2] - x0))[0]
        # shifting accordingly to width and height
        pos[:3, c, 1] = x0 + H / 2 * v_vec + D / 2 * h_vec
        pos[:3, c, 2] = x0 + H / 2 * v_vec + -D / 2 * h_vec
        pos[:3, c, 3] = x0 + -H / 2 * v_vec + -D / 2 * h_vec
        pos[:3, c, 4] = x0 + -H / 2 * v_vec + D / 2 * h_vec

        # normal of detector rectangle
        v1 = v_norm(LoS3D.get_normal(rectangle=pos[:, c, :]))[0]
        # vector to detector center from apt
        v2 = v_norm(pos[:3, c, 0] - new_ca[:3, 0])[0]

        # angle between the two for calculation
        angle = -vector_angle(v1=v1, v2=v2)
        if n >= nCh / 2.:
            angle = -angle  # other side of array

        if angle != 0.0:  # other method of fixing the angles
            # rotation axis of normal and one side normal
            rot_axis = v_norm(np.cross(
                v1, v_norm((
                    pos[:3, c, 2] + pos[:3, c, 3]
                ) / 2. - pos[:3, c, 0])[0]))[0]

            for k in [1, 2, 3, 4]:  # transform/tilt each corner
                transf = pos[:3, c, k] - pos[:3, c, 0]
                v = np.dot(R.from_rotvec(  # v2 angle round axis
                    angle * rot_axis).as_dcm(), transf)
                pos[:3, c, k] = pos[:3, c, 0] + v  # and shift

        for k in [0, 1, 2, 3, 4]:
            # r coordinate is calculated nonetheless
            pos[3, c, k] = np.sqrt(
                pos[0, c, k] ** 2 + pos[1, c, k] ** 2)

    for A, cam in enumerate(['VBCr', 'VBCl']):
        for n, c in enumerate(  # all other channels
                camera_info['channels']['eChannels'][cam]):
            pos[0, c, :] = cor[0, c, :]
            pos[1, c, :] = cor[1, c, :]
            pos[2, c, :] = cor[2, c, :]
            pos[3, c, :] = cor[3, c, :]

    # done, put in results
    artific_cor['values'] = {
        'x': pos[0], 'y': pos[1], 'z': pos[2], 'r': pos[3]}
    if saving:  # save to new file and return the new one to use
        file = loc + 'corner_geometry_artificial_HCBm_' + '.json'
        out = mClass.dict_transf(
            artific_cor, to_list=True)
        with open(file, 'w') as outfile:
            json.dump(out, outfile, indent=4, sort_keys=False)
        outfile.close()

    return (artific_cor, vmec_label + '_artificialHBCm')


def trafo_dets_aptplane(
        camera_info={'none': None},
        corners={'none': None},
        vmec_label='EIM_beta0000',
        loc='../results/INVERSION/CAMGEO/',
        center=True,
        randomize_error=False,
        error_scale=1.e-4,  # in m, 0.1 mm (0.5 mm)
        saving=True,
        debug=False):
    """ transform detectors to center of give error
    Keyword Arguments:
        camera_info {dict} -- general info (default: {{'none': None}})
        corners {dict} -- corner geometry (default: {{'none': None}})
        vmec_label {str} -- label of run (default: {'EIM_beta0000'})
        loc {str} -- location to write ({'../results/INVERSION/CAMGEO/'})
        center {bool} -- should center device? (default: {True})
        randomize_error {bool} -- give error to meas. geometry? ({False})
        error_scale {float} -- length scale to add (default: {0.0001})
        saveing {bool} -- save? (default: True)
        debug {bool} -- debugging (default: {False})
    Returns:
        center_cor {dict} -- new corner geometry
        vmec_label {str} -- new label
    """
    if center:
        print('\t\tShifting aperture to z = 0.0...')
    if randomize_error:
        print('\t\tAdding randomized error with scale ' + str(
            error_scale * 1e3) + 'mm')

    cor = np.array([  # old geometry
        corners['values']['x'], corners['values']['y'],
        corners['values']['z'], corners['values']['r']])
    center_cor = {  # results of new symm fan
        'label': 'camera_corners', 'values': None, 'aperture': {}}

    # data storage for later
    pos = np.zeros((4, 128, 5))

    apt = corners['aperture']
    for A, cam in enumerate(['HBCm', 'VBCl', 'VBCr']):
        # aperture axis of camera pinhole
        ca = np.array([  # aperture corners
            apt[cam]['x'], apt[cam]['y'],
            apt[cam]['z'], apt[cam]['r']])

        new_ca = np.zeros((4, 5))
        new_ca[:, :] = ca[:, :]  # center point the same (or all)
        if center and (cam == 'HBCm'):
            new_ca[2, 0] = 0.0  # except z component
        diff_v = new_ca[:3, 0] - ca[:3, 0]

        for k in range(1, 5):  # each corner vector has to transitioned
            if center and (cam == 'HBCm'):  # if should be centered
                new_ca[:3, k] = ca[:3, k] + diff_v[:3]  # move!

            if randomize_error:  # add error on top
                for coord in range(0, 3):  # each coordinate indiv
                    new_ca[coord, k] += error_scale * \
                        np.random.uniform(-1., 1.)

            # r coordinate is calculated from the others
            new_ca[3, k] = np.sqrt(new_ca[0, k] ** 2 + new_ca[1, k] ** 2)

        center_cor['aperture'][cam] = {
            'x': None, 'y': None, 'z': None, 'r': None}
        for i, lab in enumerate(['x', 'y', 'z', 'r']):
            center_cor['aperture'][cam][lab] = new_ca[i]

        # first off all the old positions
        pos[:, :, :] = (cor[0], cor[1], cor[2], cor[3])

        for n, c in enumerate(camera_info[  # move all the detectors likewise
                'channels']['eChannels'][cam][:]):
            for korner in range(0, 5):  # and all corners of detectors

                for i in range(0, 3):  # coordinates x, y, z
                    if center and (cam == 'HBCm'):
                        pos[i, c, korner] += diff_v[i]

                    if randomize_error:
                        pos[i, c, korner] += error_scale * \
                            np.random.uniform(-1., 1)

                # r coordinate is calculated nonetheless
                pos[3, c, korner] = np.sqrt(
                    pos[0, c, korner] ** 2 + pos[1, c, korner] ** 2)

        # measure angle error by aperture normal and see if reproducable
        if randomize_error:
            old_norm = v_norm(LoS3D.get_normal(rectangle=ca)[0])[0]
            new_norm = v_norm(LoS3D.get_normal(rectangle=new_ca)[0])[0]
            err_angle = np.rad2deg(vector_angle(v1=old_norm, v2=new_norm))
            print('\t\t\t' + cam + ': Error in camera normal ' + str(
                err_angle) + '° ')

    # done, put in results
    center_cor['values'] = {'x': pos[0], 'y': pos[1], 'z': pos[2], 'r': pos[3]}
    if saving:  # save to new file and return the new one to use
        file = loc + 'corner_geometry_'

        if center:
            file += 'centered_apt_'
        if randomize_error:
            file += 'randErr_' + str(error_scale) + '_'

        file += '.json'
        out = mClass.dict_transf(
            center_cor, to_list=True)
        with open(file, 'w') as outfile:
            json.dump(out, outfile, indent=4, sort_keys=False)
        outfile.close()

    if randomize_error:
        vmec_label += '_randErr' + str(round(error_scale * 1.e4))
    return (center_cor, vmec_label + '_centered_apt')


def fix_rotate_boloplane(
        camera_info={'none': None},
        corners={'none': None},
        vmec_label='EIM_beta0000',
        loc='../results/INVERSION/CAMGEO/',
        saving=True,
        debug=False):
    """ fix the intrinsic LoS error between halfes of fan
    Keyword Arguments:
        camera_info {dict} -- general info (default: {{'none': None}})
        corners {dict} -- corner geometries (default: {{'none': None}})
        vmec_label {str} -- run label (default: {'EIM_beta0000'})
        loc {str} -- write location (default: {'../results/INVERSION/CAMGEO/'})
        saving {bool} -- save? (default: {True})
        debug {bool} -- debugging (default: {False})
    Returns:
        rot_cor {dict} -- rotated detector corner geometry
        vmec_label {str} -- new label
    """
    print('\t\tRotated HBCm detectors so that they all are in one plane...')

    cor = np.array([  # old geometry
        corners['values']['x'], corners['values']['y'],
        corners['values']['z'], corners['values']['r']])
    rot_cor = {  # results of new symm fan
        'label': 'camera_corners', 'values': None, 'aperture': {}}
    # data storage for later
    pos = np.zeros((4, 128, 5))

    apt = corners['aperture']
    for A, cam in enumerate(['HBCm']):
        ca = np.array([  # aperture axis of camera pinhole
            apt[cam]['x'], apt[cam]['y'],
            apt[cam]['z'], apt[cam]['r']])
        axis = 1. * v_norm(LoS3D.get_normal(
            rectangle=ca))[0]  # axis of aperture

        # number of channels and where to stop flipping
        nCh = np.shape(camera_info['channels']['eChannels'][cam])[0]
        D, M = nCh - np.floor(nCh / 2.) * 2., np.int(np.floor(nCh / 2.))
        if D != 0.0:
            print('\t\t\t...  residue is', D)
        else:
            print('\t\t\t... no residue, split halfway')

        if debug:
            for i, ch in enumerate(
                    camera_info['channels']['eChannels'][cam][:M]):
                print('\t\t\t', i, ':', ch,
                      camera_info['channels']['eChannels'][cam][-(i + 1)],
                      ':', nCh - (i + 1))

        for n, c in enumerate(camera_info[
                'channels']['eChannels'][cam][:]):

            if n < M:
                # channels to compare to eachother and aperture axis
                c1, c2 = c, camera_info['channels'][
                    'eChannels'][cam][-(n + 1)]
                if debug:
                    print(c1, c2)

                # make vectors
                v1 = cor[:3, c1, 0] - ca[:3, 0]
                v2 = cor[:3, c2, 0] - ca[:3, 0]

                # rotate v1, v2 accordigly to twist
                R1 = np.dot(R.from_rotvec(  # v1_180 to other side of v2
                    np.pi * axis).as_dcm(), v1)
                R2 = np.dot(R.from_rotvec(  # v2 angle v1_180 round axis
                    2 * vector_angle(v1=R1, v2=v2) * axis).as_dcm(), v2)

                diff_v = np.array([  # transitional vector for detectors
                    [(ca[0, 0] + R2[0]) - cor[0, c2, 0]],
                    [(ca[1, 0] + R2[1]) - cor[1, c2, 0]],
                    [(ca[2, 0] + R2[2]) - cor[2, c2, 0]]])

                for korner in range(0, 5):
                    for i in range(0, 3):
                        pos[i, c, korner] = cor[i, c, korner] + diff_v[i]
                    pos[3, c, korner] = np.sqrt(
                        pos[0, c, korner] ** 2 + pos[1, c, korner] ** 2)

            else:
                for i in range(0, 4):
                    pos[i, c, :] = cor[i, c, :]

    for A, cam in enumerate(['VBCr', 'VBCl']):
        for n, c in enumerate(camera_info[
                'channels']['eChannels'][cam][:]):
            for i in range(0, 4):
                pos[i, c, :] = cor[i, c, :]

    # done, put in results
    rot_cor['aperture'] = corners['aperture']
    rot_cor['values'] = {'x': pos[0], 'y': pos[1], 'z': pos[2], 'r': pos[3]}

    if saving:  # save to new file and return the new one to use
        file = loc + 'corner_geometry_det_fix_boloplane.json'
        out = mClass.dict_transf(
            rot_cor, to_list=True)
        with open(file, 'w') as outfile:
            json.dump(out, outfile, indent=4, sort_keys=False)
        outfile.close()

    return (rot_cor, vmec_label + '_det_fix_boloplane')


def symmetrical_det_aptplane(
        camera_info={'none': None},
        corners={'none': None},
        vmec_label='EIM_beta0000',
        loc='../results/INVERSION/CAMGEO/',
        saving=True,
        debug=False):
    """ symmetrization of detector arrangement to bolometer aperture plane
    Keyword Arguments:
        camera_info {dict} -- general info (default: {{'none': None}})
        corners {dict} -- corner geometries (default: {{'none': None}})
        vmec_label {str} -- run label (default: {'EIM_beta0000'})
        loc {str} -- write location (default: {'../results/INVERSION/CAMGEO/'})
        saving {bool} -- save? (default: {True})
        debug {bool} -- debugging (default: {False})
    Returns:
        symm_cor {dict} -- 'symmetric' detector array
        vmec_label {str} -- new label
    """
    print('\t\tSymmetric detector arrangement in aperture plane...')
    cor = np.array([  # old geometry
        corners['values']['x'], corners['values']['y'],
        corners['values']['z'], corners['values']['r']])
    symm_cor = {  # results of new symm fan
        'label': 'camera_corners', 'values': None, 'aperture': {}}
    # data storage for later
    pos = np.zeros((4, 128, 5))

    apt = corners['aperture']
    for A, cam in enumerate(['HBCm', 'VBCl', 'VBCr']):
        # aperture axis of camera pinhole
        ca = np.array([  # aperture corners
            apt[cam]['x'], apt[cam]['y'],
            apt[cam]['z'], apt[cam]['r']])

        # aperture axis needed for rotation later,
        # center of torus to mid of aperture, as if symmetric
        axis = -1. * v_norm([ca[0, 0], ca[1, 0], ca[2, 0]])[0]
        apt_norm = -1. * v_norm(LoS3D.get_normal(rectangle=ca))[0]

        apt_plane_norm = v_norm(
            LoS3D.get_normal(rectangle=np.array([
                [ca[0, 0], ca[0, 0] + apt_norm[0],
                 cor[0, 0, 0], ca[0, 0]],
                [ca[1, 0], ca[1, 0] + apt_norm[1],
                 cor[1, 0, 0], ca[1, 0]],
                [ca[2, 0], ca[2, 0] + apt_norm[2],
                 cor[2, 0, 0], ca[2, 0]]])))[0]

        new_ca = np.zeros((4, 5))
        new_ca[:, 0] = ca[:, 0]
        rot_axis = np.array([apt_plane_norm[0], apt_plane_norm[1], .0])

        for k in range(1, 5):  # each corner vector has to be turned
            vec = ca[:3, k] - ca[:3, 0]
            v = np.dot(R.from_rotvec(  # v2 angle to v1_180 round axis
                -1. * vector_angle(
                    v1=rot_axis, v2=apt_plane_norm) * axis).as_dcm(),
                vec)

            new_ca[:3, k] = new_ca[:3, 0] + v
            # get R component at the end for each
            new_ca[3, k] = np.sqrt(new_ca[0, k] ** 2 + new_ca[1, k] ** 2)

        symm_cor['aperture'][cam] = {
            'x': None, 'y': None, 'z': None, 'r': None}
        for i, lab in enumerate(['x', 'y', 'z', 'r']):
            symm_cor['aperture'][cam][lab] = new_ca[i]

        # number of channels and where to stop flipping
        Ch = camera_info['channels']['eChannels'][cam]

        # set up the 'plane normal' of channel fan
        pNorm = v_norm(LoS3D.get_normal(rectangle=np.array([
            [ca[0, 0], cor[0, Ch[0], 0], cor[0, Ch[-1], 0], ca[0, 0]],
            [ca[1, 0], cor[1, Ch[0], 0], cor[1, Ch[-1], 0], ca[1, 0]],
            [ca[2, 0], cor[2, Ch[0], 0], cor[2, Ch[-1], 0], ca[2, 0]]])
        ))[0]

        for n, c in enumerate(camera_info[
                'channels']['eChannels'][cam][:]):

            # axis to measure angle to be turned by
            rot_axis = np.array([pNorm[0], pNorm[1], .0])
            rev_vec = np.array([  # vector projecting the detector
                [cor[0, c, 0] - ca[0, 0]], [cor[1, c, 0] - ca[1, 0]],
                [cor[2, c, 0] - ca[2, 0]]])
            # not turning the re_vec by the angle the pNorm has with rot_axis
            # around the apt_axis, making the boloplane at 108° symmetric
            rot = np.dot(R.from_rotvec(  # v2 angle to v1_180 round axis
                -1. * vector_angle(v1=rot_axis, v2=pNorm) * axis).as_dcm(),
                rev_vec)

            diff_v = np.array([  # transitional vector for detectors
                [(ca[0, 0] + rot[0]) - cor[0, c, 0]],
                [(ca[1, 0] + rot[1]) - cor[1, c, 0]],
                [(ca[2, 0] + rot[2]) - cor[2, c, 0]]])

            for korner in range(0, 5):
                for i in range(0, 3):
                    pos[i, c, korner] = cor[i, c, korner] + diff_v[i]
                pos[3, c, korner] = np.sqrt(
                    pos[0, c, korner] ** 2 + pos[1, c, korner] ** 2)

    # done, put in results
    symm_cor['values'] = {'x': pos[0], 'y': pos[1], 'z': pos[2], 'r': pos[3]}
    if saving:  # save to new file and return the new one to use
        file = loc + 'corner_geometry_sym_dets_aptplane.json'
        out = mClass.dict_transf(
            symm_cor, to_list=True)
        with open(file, 'w') as outfile:
            json.dump(out, outfile, indent=4, sort_keys=False)
        outfile.close()

    return (symm_cor, vmec_label + '_sym_dets_aptplane')


def aperture_axis_tilt(
        deg=5.,
        camera_info={'none': None},
        corners={'none': None},
        vmec_label='EIM_beta0000',
        loc='../results/INVERSION/CAMGEO/',
        saving=True,
        debug=False):
    """ tilt symmetrization of detector arrangement by tilt_deg
    Keyword Arguments:
        deg {float} -- degrees tilt (default: {5.})
        camera_info {dict} -- general info (default: {{'none': None}})
        corners {dict} -- corner geometries (default: {{'none': None}})
        vmec_label {str} -- run label (default: {'EIM_beta0000'})
        loc {str} -- write location (default: {'../results/INVERSION/CAMGEO/'})
        saving {bool} -- save? (default: {True})
        debug {bool} -- debugging (default: {False})
    Returns:
        tilt_cor {dict} -- tilted detector corner geometry
        vmec_label {str} -- new label
    """
    print('\t\tTilting camera aperture by angle', str(deg) + '° ...')

    cor = np.array([  # old geometry
        corners['values']['x'], corners['values']['y'],
        corners['values']['z'], corners['values']['r']])
    tilt_cor = {  # results of new symm fan
        'label': 'camera_corners', 'values': None, 'aperture': {}}
    # data storage for later
    pos = np.zeros((4, 128, 5))

    apt = corners['aperture']  # aperture corners
    for A, cam in enumerate(['HBCm', 'VBCl', 'VBCr']):
        ch = camera_info['channels']['eChannels'][cam]

        ca = np.array([  # aperture corners
            apt[cam]['x'], apt[cam]['y'],
            apt[cam]['z'], apt[cam]['r']])

        # axis to turn LoS/detector plane by angle poloidally
        rectangle = np.array([  # spanning detector fan plane for axis
            [ca[0, 0], cor[0, np.min(ch), 0], cor[0, np.max(ch), 0], ca[0, 0]],
            [ca[1, 0], cor[1, np.min(ch), 0], cor[1, np.max(ch), 0], ca[1, 0]],
            [ca[2, 0], cor[2, np.min(ch), 0], cor[2, np.max(ch), 0], ca[2, 0]]])
        # plane normal becomes axis to turn around
        axis = v_norm(LoS3D.get_normal(rectangle=rectangle))[0]

        # need to span new aperture turned by angle, like normal etc
        new_ca = np.zeros((4, 5))
        new_ca[:, 0] = ca[:, 0]

        for k in range(1, 5):  # each corner vector has to be turned
            vec = ca[:3, k] - ca[:3, 0]
            v = np.dot(R.from_rotvec(  # by angle and same axis
                np.deg2rad(deg) * axis).as_dcm(), vec)
            new_ca[:3, k] = new_ca[:3, 0] + v
            # get R component at the end for each
            new_ca[3, k] = np.sqrt(new_ca[0, k] ** 2 + new_ca[1, k] ** 2)

        tilt_cor['aperture'][cam] = {
            'x': None, 'y': None, 'z': None, 'r': None}
        for i, lab in enumerate(['x', 'y', 'z', 'r']):
            tilt_cor['aperture'][cam][lab] = new_ca[i]

        for n, c in enumerate(camera_info[
                'channels']['eChannels'][cam][:]):

            rev_vec = np.array([  # reverse vector to detector
                [cor[0, c, 0] - ca[0, 0]], [cor[1, c, 0] - ca[1, 0]],
                [cor[2, c, 0] - ca[2, 0]]])
            rot = np.dot(R.from_rotvec(  # turn that again
                np.deg2rad(deg) * axis).as_dcm(), rev_vec)

            diff_v = np.array([  # difference to transposition the detector
                [(ca[0, 0] + rot[0]) - cor[0, c, 0]],
                [(ca[1, 0] + rot[1]) - cor[1, c, 0]],
                [(ca[2, 0] + rot[2]) - cor[2, c, 0]]])

            for korner in range(0, 5):  # every corner has to be moved
                for i in range(0, 3):
                    pos[i, c, korner] = cor[i, c, korner] + diff_v[i]
                pos[3, c, korner] = np.sqrt(
                    pos[0, c, korner] ** 2 + pos[1, c, korner] ** 2)

    # done, put in results
    tilt_cor['values'] = {'x': pos[0], 'y': pos[1], 'z': pos[2], 'r': pos[3]}
    if saving:  # save to new file and return the new one to use
        file = loc + 'corner_geometry_tilt_apt_plane_deg' + \
            str(deg) + '.json'
        out = mClass.dict_transf(
            tilt_cor, to_list=True)
        with open(file, 'w') as outfile:
            json.dump(out, outfile, indent=4, sort_keys=False)
        outfile.close()

    return (tilt_cor, vmec_label + '_tilt_cor_' + str(deg) + 'deg')


def lofs_geometry(
        camera_info={'channels': {'eChannels': {}}, 'geometry': {}},
        base_geometry={'values': {'x': None, 'y': None, 'z': None}},
        corner_geometry={'values': {'x': None, 'y': None, 'z': None}},
        loc='../results/INVERSION/LOFS/TRIAGS/',
        M=8,
        new_type=None,
        cams=['HBCm', 'VBCl', 'VBCr'],
        label='EIM_beta000',
        add_camera=False,
        debug=False,
        saving=False,
        overwrite=False):
    """ lines of sight through center point of detector and slit each
    Args:
        corner_geometry (0, dict): each camera and detector corner location
        debug (1, bool): debugging
        saving (2, bool): saving
    Returns:
        lines_of_sight (0, dict): lofs as slopes and x,y pair of vectors
    Notes:
        None.
    """
    def get_plane_coeffs(v1, v2, origin):
        """
        Args:
            points (zeros((3, 3))): 3 points with 3 coords
        """
        # These two vectors are in the plane
        v1, v2 = v1 - origin[:3], v2 - origin[:3]
        # the cross product is a vector normal to the plane
        cp = np.cross(v1, v2)
        a, b, c = cp
        # This evaluates a * x3 + b * y3 + c * z3 which equals d
        d = np.dot(cp, v2)
        return (
            np.array([
                a, b, c, d]))

    def plane_intersect(a, b):
        """
        Args:
            a, b: 4-tuples/lists
                  Ax + By +Cz + D = 0
                  A,B,C,D in order
        Returns:
            2 points on line of intersection, np.arrays, shape (3,) """
        a_vec, b_vec = np.array(a[:3]), np.array(b[:3])
        aXb_vec = np.cross(a_vec, b_vec)

        A = np.array([a_vec, b_vec, aXb_vec])
        d = np.array([-a[3], -b[3], 0.]).reshape(3, 1)
        # could add np.linalg.det(A) == 0
        # test to prevent linalg.solve throwing error

        p_inter = np.linalg.solve(A, d).T
        return (
            p_inter[0],
            (p_inter + aXb_vec)[0])

    def line3D(
            p1, p2):
        # xy plane
        a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p2[1] - a * p2[0]
        # xz plamne
        c = (p2[2] - p1[2]) / (p2[0] - p1[0])
        d = p2[2] - c * p2[0]
        return (a, b, c, d)

    def solve_x_LoS(
            cam, a, b, c, d, q, r, s, L):
        # solve L = sqrt((x - r)^2 + (a * x + b - s)^2 + (c * x + d - q)^2)
        solution = [
            (-1. * np.sqrt((2. * a * b - 2. * a * r + 2. * c * d -
            2. * c * s - 2. * q)**2 - 4. * (a**2 + c**2 + 1.) * (
            b**2 - 2. * b * r + d**2 - 2. * d * s - L**2 + q**2 +
            r**2 + s**2)) - 2. * a * b + 2. * a * r - 2. * c * d +
            2. * c * s + 2. *q) / (2. * (a**2 + c**2 + 1.)),
            (1. * np.sqrt((2. * a * b - 2. * a * r + 2. * c * d -
            2. * c * s - 2. * q)**2 - 4. * (a**2 + c**2 + 1.) * (
            b**2 - 2. * b * r + d**2 - 2. * d * s - L**2 + q**2 +
            r**2 + s**2)) - 2. * a * b + 2. * a * r - 2. * c * d +
            2. * c * s + 2. *q) / (2. * (a**2 + c**2 + 1.))]
        if (cam == 'HBCm'):
            return (solution[1])
        elif (cam == 'ARTf'):
            if (a > .0) and (new_type == 'MIRh'):
                return (solution[1])
            if (c < .0) and (new_type == 'VBCm'):
                return (solution[1])
        elif (((cam == 'VBCl') or (cam == 'VBCr')) and
                (c * solution[1] + d > q)):
            return (solution[1])
        return (solution[0])

    def solve_r_LoS(
            cam, a, b, r, s, L):
        solution = [  # solve L = sqrt((x - r)^2 + (a * x + b - s)^2)
            (-1. * a * b + r + a * s - np.sqrt(-1. * b**2 + L**2 +
             a**2 * L**2 - 2. * a * b * r - a**2 * r**2 + 2. * b * s +
             2. * a * r * s - s**2)) / (1. + a**2),
            (-1. * a * b + r + a * s + np.sqrt(-1. * b**2 + L**2 +
             a**2 * L**2 - 2. * a * b * r - a**2 * r**2 + 2. * b * s +
             2. * a * r * s - s**2)) / (1. + a**2)]
        if (cam == 'ARTf'):
            if (new_type == 'MIRh'):
                return (solution[1])
            elif ((new_type == 'VBCm')):
                if ((a < .0)):
                    return (solution[1])
        elif (cam == 'VBCl') or (cam == 'VBCr'):
            if (a > .0):
                return (solution[1])
        return (solution[0])

    def LoS_vec_from_points(
            cam, p1, p2):
        a, b, c, d = line3D(p1, p2)
        # define cut of LoS
        q, r, s = p2
        tx = solve_x_LoS(cam, a, b, c, d, q, r, s, 2.2)
        # calculate endpoint
        pE = np.array([  # endpoint
            tx, a * tx + b, c * tx + d])
        return (p2, pE)

    def LoS_line_from_points(
            cam, a, b, p02D):
        tr = solve_r_LoS(cam, a, b, p02D[0], p02D[1], 2.2)
        pE = np.array([  # endpoint
            tr, a * tr + b])
        return (p02D, pE)

    print('\tConstructing lines of sight...')
    # C = mClass.dict_transf(corner_geometry, to_list=False)
    V, A = corner_geometry['values'], corner_geometry['aperture']

    if saving:
        file = 'lofs_2D_simple3D_' + label
        if os.path.isfile(loc + 'rz_line_' + file + '.npz'):
            return (store_read_los_arrays(name=file))
        else:
            pass

    # results dictionary setup
    lofs = {
        'label': 'lines_of_sight', 'values': {
            'xy_plane': {'slope': np.zeros((128, M**2)),
                         'constant': np.zeros((128, M**2)),
                         'range': np.zeros((128, M**2, 2)),
                         'line': np.zeros((128, M**2, 2))},
            'xz_plane': {'slope': np.zeros((128, M**2)),
                         'constant': np.zeros((128, M**2)),
                         'range': np.zeros((128, M**2, 2)),
                         'line': np.zeros((128, M**2, 2))},
            'rz_plane': {'slope': np.zeros((128, M**2)),
                         'constant': np.zeros((128, M**2)),
                         'range': np.zeros((128, M**2, 2)),
                         'line': np.zeros((128, M**2, 2))}}}
    L = lofs['values']
    x, y, z, r = V['x'], V['y'], V['z'], V['r']

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import axes3d
    for i, camera in enumerate(cams):
        a = np.array([
            A[camera]['x'], A[camera]['y'], A[camera]['z']])

        for j, ch in enumerate(camera_info['channels']['eChannels'][camera]):
            det = np.array([x[ch], y[ch], z[ch]])

            for p in range(M):
                for d in range(M):

                    if debug:
                        print(camera, j, ch, p, d)
                    q, r, s, t = line3D(a[:, 0, p], det[:, d, 0])
                    # xy plane
                    L['xy_plane']['slope'][ch, p * M + d] = q  # s_xy
                    L['xy_plane']['constant'][ch, p * M + d] = r  # c_xy
                    # xz plane
                    L['xz_plane']['slope'][ch, p * M + d] = s  # s_xz
                    L['xz_plane']['constant'][ch, p * M + d] = t  # c_xz

                    p0, pE = LoS_vec_from_points(
                        camera, a[:, 0, p], det[:, d, 0])

                    L['xy_plane']['range'][ch, p * M + d, :] = [p0[0], pE[0]]
                    L['xy_plane']['line'][ch, p * M + d, :] = [p0[1], pE[1]]
                    L['xz_plane']['range'][ch, p * M + d, :] = [p0[0], pE[0]]
                    L['xz_plane']['line'][ch, p * M + d, :] = [p0[2], pE[2]]

    # do projection into camera plane
    A_old, V_old = base_geometry['aperture'], base_geometry['values']
    c = np.array([  # old geometry
        V_old['x'], V_old['y'], V_old['z'], V_old['r']])

    for i, camera in enumerate(cams):
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')

        a = np.array([A[camera]['x'], A[camera]['y'],
                      A[camera]['z'], A[camera]['r']])
        a_old = np.array([
            A_old[camera]['x'], A_old[camera]['y'],
            A_old[camera]['z'], A_old[camera]['r']])

        ch0, chE = \
            camera_info['channels']['eChannels'][camera][0], \
            camera_info['channels']['eChannels'][camera][-1]

        det0 = np.array([c[0, ch0, 0], c[1, ch0, 0], c[2, ch0, 0]])
        detE = np.array([c[0, chE, 0], c[1, chE, 0], c[2, chE, 0]])

        p00, p0E = LoS_vec_from_points(camera, a_old[:, 0], det0)
        pE0, pEE = LoS_vec_from_points(camera, a_old[:, 0], detE)

        # get plane coeffs from plane span
        # of innermost, outermost LoS and aperture
        cam_plane_coeffs = get_plane_coeffs(
            v1=v_norm(p0E - p00)[0],
            v2=v_norm(pEE - pE0)[0],
            origin=a_old[:, 0])

        # plane of x,y where the other shall intersect
        cartesian_plane_coeffs = get_plane_coeffs(
            v1=np.array([0., 1., 0.]),
            v2=np.array([1., 0., 0.]),
            origin=np.array([0., 0., 0.]))

        # intersection vector defined by p1,p2
        p1, p2 = plane_intersect(
            cam_plane_coeffs, cartesian_plane_coeffs)
        intersect_norm = v_norm(p2 - p1)[0]

        # ax.plot(
        #     [a_old[0, 0], a_old[0, 0] + intersect_norm[0]],
        #     [a_old[1, 0], a_old[1, 0] + intersect_norm[1]],
        #     [a_old[2, 0], a_old[2, 0] + intersect_norm[2]],
        #     c='r', lw=0.5, alpha=0.7)

        for j, ch in enumerate(camera_info[
                'channels']['eChannels'][camera]):

            for p in range(M):
                p03D = a[:, 0, p]
                p02D = np.array([
                    np.sqrt(a[0, 0, p]**2 + a[1, 0, p]**2), a[2, 0, p]])

                for d in range(M):
                    LoS = np.array([
                        L['xy_plane']['range'][ch, p * M + d, :],
                        L['xy_plane']['line'][ch, p * M + d, :],
                        L['xz_plane']['line'][ch, p * M + d, :]])

                    gamma = vector_angle(
                        v1=intersect_norm,
                        v2=v_norm(LoS[:, -1] - LoS[:, 0])[0])

                    # ax.plot(
                    #     LoS[0], LoS[1], LoS[2],
                    #     c='k', lw=0.33, alpha=0.5)

                    if (camera == 'HBCm') and (LoS[2, -1] < LoS[2, 0]):
                        gamma = np.pi - gamma
                    elif (camera == 'VBCr'):
                        gamma = np.pi - gamma

                    D = np.sqrt(
                        (LoS[0] - p03D[0])**2 +
                        (LoS[1] - p03D[1])**2 +
                        (LoS[2] - p03D[2])**2)

                    r_new = D * np.cos(gamma) + p02D[0]
                    z_new = D * np.sin(gamma) + p02D[1]

                    s_rz = (z_new[0] - z_new[-1]) / (r_new[0] - r_new[-1])
                    c_rz = z_new[0] - s_rz * r_new[0]

                    if (camera == 'ARTf'):
                        if ((new_type == 'MIRh') and (
                                a_old[2, 0] > c[2, ch, 0])):
                            s_rz, c_rz = -1. * s_rz, -1. * c_rz
                        elif (new_type == 'VBCm'):
                            s_rz, c_rz = -1. * s_rz, -1. * c_rz

                    L['rz_plane']['slope'][ch, p * M + d] = s_rz
                    L['rz_plane']['constant'][ch, p * M + d] = c_rz

                    p0, pE = LoS_line_from_points(
                        camera, s_rz, c_rz, p02D)

                    L['rz_plane']['range'][ch, p * M + d, :] = [p0[0], pE[0]]
                    L['rz_plane']['line'][ch, p * M + d, :] = [p0[1], pE[1]]

    # import pdb
    # pdb.set_trace()

    if saving:
        lofs = store_read_los_arrays(
            name=file, los_data=lofs,
            overwrite=overwrite)

    return (lofs)


def v_norm(
        v=np.zeros((3)),
        debug=False):
    return (
        v / np.sqrt(np.dot(v, v)),
        np.sqrt(np.dot(v, v)))


def vector_angle(
        v1=Z((3)),
        v2=Z((3)),
        debug=False):

    return (np.arccos(np.dot(v1, v2) / (
        np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))))


def vector_rotation_matrix(
        axis=Z((3)),
        angle=10. * (2 * np.pi / 360.),
        debug=False):
    # norm axis
    axis = axis / np.sqrt(np.dot(axis, axis))
    # half angle
    a = np.cos(angle / 2.)
    b, c, d = -1. * axis * np.sin(angle / 2.)

    return (np.array(
        [[a * a + b * b - c * c - d * d,
          2 * (b * c + a * d), 2 * (b * d - a * c)],
         [2 * (b * c - a * d), a * a + c * c - b * b - d * d,
          2 * (c * d + a * b)],
         [2 * (b * d + a * c), 2 * (c * d - a * b),
          a * a + d * d * - b * b - c * c]]))


def store_read_los_arrays(
        name='_EIM_beta000_tN8',
        los_data=None,
        base='../results/INVERSION/LOFS/TRIAGS/',
        overwrite=False):

    if los_data is None:
        los_data = {
            'label': 'lines_of_sight', 'values': {
                'xy_plane': {'slope': None, 'constant': None,
                             'range': None, 'line': None},
                'xz_plane': {'slope': None, 'constant': None,
                             'range': None, 'line': None},
                'rz_plane': {'slope': None, 'constant': None,
                             'range': None, 'line': None}}}

    def load_store_prop(
            label, P):
        loc = base + label + name + '.npz'
        if os.path.isfile(loc):
            if not overwrite:
                P = np.load(loc)['arr_0']
                print('\t\t\\\ load ' + label.replace('_', ' '), np.shape(P),
                      format(sys.getsizeof(
                          P) / (1024. * 1024.), '.2f') + 'MB')
            elif overwrite and P is not None:
                np.savez_compressed(loc, P)
                print('\t\t\\\ overwritten ' + label.replace('_', ' '),
                      np.shape(P),
                      format(sys.getsizeof(
                          P) / (1024. * 1024.), '.2f') + 'MB')
            else:
                print('\t\t\\\ ' + label.replace('_', ' ') +
                      ' is None, no overwrite')

        else:
            if P is not None:
                np.savez_compressed(loc, P)
                print('\t\t\\\ save ' + label.replace('_', ' '), np.shape(P),
                      format(sys.getsizeof(
                          P) / (1024. * 1024.), '.2f') + 'MB')
            else:
                print('\t\t\\\ ' + label + ' is None')
        return (P)

    for plane in ['xy', 'xz', 'rz']:
        for prop in ['slope', 'constant', 'range', 'line']:
            los_data['values'][plane + '_plane'][prop] = \
                load_store_prop(plane + '_' + prop + '_',
                                los_data['values'][plane + '_plane'][prop])

    return (los_data)


def split_detector_slit(
        N=[2, 4],
        corners={'none': None},
        camera_info={'none': None},
        vmec_label='EIM_beta0000',
        loc='../results/INVERSION/CAMGEO/',
        cams=['HBCm', 'VBCr', 'VBCrl'],
        saving=True,
        debug=False):
    print('\t\tSplitting detectors and slit ' +
          str(N[0]) + 'x' + str(N[1]) + ' times')

    # short side should be split N[0] times, the longer N[1]
    # equaling to M squares
    M = N[0] * N[1]

    file = loc + 'corner_geometry' + vmec_label + \
        '_squareN' + str(M) + '.json'

    if False:  # saving:
        if os.path.isfile(file):
            with open(file, 'r') as infile:
                indict = json.load(infile)
            infile.close()
            split_cor = mClass.dict_transf(
                indict, to_list=False)

            return (split_cor, vmec_label + '_sN' + str(M), M)
        else:
            pass

    def barycenter(
            points,  # zeros((n, 3)) n points x 3 dims
            debug=False):
        bc = np.zeros((np.shape(points)[1]))
        for k in range(np.shape(points)[1]):
            bc[k] = np.mean(points[:, k])
        return (bc)  # x, y, z

    def vL(vector):
        L = np.sqrt(np.dot(vector, vector))
        return (L)  # length float

    def points_from_base(
            base=np.zeros((3)),  # starting point
            v1=np.zeros((3)),    # long vector
            v2=np.zeros((3)),    # short vector
            n10=0,               # long point
            n20=0,               # short point
            n1E=4,               # long seperation
            n2E=2):              # short seperation
        points = np.zeros((4, 3))  # 4 points, 3 dims
        for i in range(2):
            for j in range(2):
                points[i * 2 + j, :] = \
                    base + (n10 + i) / n1E * v1 + (n20 + j) / n2E * v2
        return (points[0], points[1], points[2], points[3])

    def split_rectangle(
            square=np.zeros((3, 5)),
            nS=2,
            nL=4):
        splits = np.zeros((3, nS * nL, 5))  # 3 coords, M squares, 5 points

        # 3 coords, 5 points
        # find longest side
        a = vL(square[:, 2] - square[:, 1])  # 2, 1
        b = vL(square[:, 4] - square[:, 1])  # 3, 2

        if (a > b):
            # a is longest
            L_vector = square[:, 2] - square[:, 1]
            S_vector = square[:, 4] - square[:, 1]

        else:  # (b > a)
            # b is longest
            L_vector = square[:, 4] - square[:, 1]
            S_vector = square[:, 2] - square[:, 1]

        for L in range(nL):
            for S in range(nS):
                k = L * nS + S
                p1, p2, p3, p4 = points_from_base(
                    base=square[:, 1], v1=L_vector, v2=S_vector,
                    n10=L, n20=S, n1E=nL, n2E=nS)

                splits[:, k, 1] = p1
                splits[:, k, 2] = p2
                splits[:, k, 3] = p3
                splits[:, k, 4] = p4
                splits[:, k, 0] = barycenter(
                    points=splits[:, k, 1:].transpose())

        return (splits.transpose(1, 2, 0))

    cor = np.array([  # old geometry
        corners['values']['x'], corners['values']['y'],  # in m
        corners['values']['z']])  # in m
    apt = corners['aperture']  # aperture
    split_cor = {  # results of new fan
        'label': 'camera_corners',
        'values': None,
        'aperture': {},
        'angles': None}

    # 4 coords, M squares with 4 corners and 1 center
    split = np.zeros((4, 128, M, 5))
    angles = np.zeros((128, M * M))

    print('\t\t>> End result supposed to have ' + str(M) + ' squares')
    for cam in cams:
        nCh = camera_info['channels']['eChannels'][cam][:]
        # 4 coords, N tris, 3 corners 1 center
        A = np.array([  # x y z of old apt
            apt[cam]['x'], apt[cam]['y'], apt[cam]['z']])

        # define camplane normal
        camplane_norm = v_norm(np.cross(
            np.array([
                cor[0, nCh[0], 0] - A[0, 0],
                cor[1, nCh[0], 0] - A[1, 0],
                cor[2, nCh[0], 0] - A[2, 0]]),
            np.array([
                cor[0, nCh[-1], 0] - A[0, 0],
                cor[1, nCh[-1], 0] - A[1, 0],
                cor[2, nCh[-1], 0] - A[2, 0]])))[0]
        a_splits = split_rectangle(square=A, nS=N[1], nL=N[0])

        if False:  # cam == 'HBCm':
            gp.plot_final_triangulation(
                base=A.transpose(), triangles=a_splits[:, 1:, :],
                centers=a_splits[:, 0, :], N=M,
                L='square_aperture_' + cam)

        split_cor['aperture'][cam] = {
            'x': None, 'y': None, 'z': None, 'r': None}
        for i, lab in enumerate(['x', 'y', 'z']):
            split_cor['aperture'][cam][lab] = \
                np.array([  # center, and square corners
                    a_splits[:, 0, i], a_splits[:, 1, i],
                    a_splits[:, 2, i], a_splits[:, 3, i],
                    a_splits[:, 4, i]])

        split_cor['aperture'][cam]['r'] = \
            np.sqrt(np.square(split_cor['aperture'][cam]['x']) +
                    np.square(split_cor['aperture'][cam]['y']))

        for n, c in enumerate(nCh):
            D = cor[:, c]
            d_split = split_rectangle(square=D, nS=N[1], nL=N[0])

            for d in range(M):
                split[0, c, d, :] = d_split[d, :, 0]
                split[1, c, d, :] = d_split[d, :, 1]
                split[2, c, d, :] = d_split[d, :, 2]
                split[3, c, d, :] = np.sqrt(
                    d_split[d, :, 0]**2 + d_split[d, :, 1]**2)

            if False:  # c == 0:
                gp.plot_final_triangulation(
                    base=D.transpose(), triangles=d_split[:, 1:, :],
                    centers=d_split[:, 0, :], N=M,
                    L='square_detector_eChan_' + str(c))

            # get angles of LoS out of plane
            for p in range(M):  # for pinhole squares
                for d in range(M):  # for det squares
                    vec = v_norm(np.array([
                        a_splits[p, 0, 0] - split[0, c, d, 0],
                        a_splits[p, 0, 1] - split[1, c, d, 0],
                        a_splits[p, 0, 2] - split[2, c, d, 0]]))[0]

                    angles[c, p * M + d] = np.arcsin(np.abs(np.dot(
                        camplane_norm, vec) / (np.sqrt(np.dot(
                            camplane_norm, camplane_norm)) *
                            np.sqrt(np.dot(vec, vec)))))

    # print(np.shape(split))
    split_cor['values'] = {
        'x': split[0], 'y': split[1], 'z': split[2], 'r': split[3]}
    split_cor['angles'] = angles

    if saving:  # save to new file and return the new one to use
        out = mClass.dict_transf(
            split_cor, to_list=True)
        with open(file, 'w') as outfile:
            json.dump(out, outfile, indent=4, sort_keys=False)
        outfile.close()
        split_cor = mClass.dict_transf(
            split_cor, to_list=False)

    return (split_cor, vmec_label + '_sN' + str(M), M)


def triangulate_detector_slit(
        camera_info={'none': None},
        corners={'none': None},
        vmec_label='EIM_beta0000',
        loc='../results/INVERSION/CAMGEO/',
        cams=['HBCm', 'VBCr', 'VBCrl'],
        N=4,
        saving=True,
        debug=False):
    if N != 2:
        N = 2**math.ceil(np.sqrt(N))  # make power of 2
    print('\t\tTriangulating detectors and slit ' + str(N) + ' times')

    file = loc + 'corner_geometry' + vmec_label + \
        '_triagN' + str(N) + '.json'

    if saving:
        if os.path.isfile(file):
            with open(file, 'r') as infile:
                indict = json.load(infile)
            infile.close()
            tri_cor = mClass.dict_transf(
                indict, to_list=False)

            return (tri_cor, vmec_label + '_tN' + str(N), N)
        else:
            pass

    def barycenter(
            points,  # zeros((n, 3)) n points x 3 dims
            debug=False):
        bc = np.array([
            np.mean(points[:, 0]), np.mean(points[:, 1]),
            np.mean(points[:, 2])])
        return (bc)  # x, y, z

    def two_triangles(
            points,  # zeros((5, 3)) 5 points x 3 dims
            debug=False):

        spcs = triangulate(points[:, :2], qhull_options='QJ')
        triags = points[spcs.simplices]

        if debug:
            print('triags:', np.shape(triags), triags)

        centers = np.zeros((np.shape(triags)[0], 3))
        for n in range(np.shape(triags)[0]):
            centers[n] = barycenter(
                triags[n])
        return (
            triags,   # 2 triags x 3 points x 3 dims
            centers)  # 2 centers x 3 dims

    def vL(vector):
        L = np.sqrt(np.dot(vector, vector))
        return (L)  # length float

    def two_triangles_from_triangle(
            points,  # zeros((3, 3)) 3 points x 3 dims
            debug=False):
        triags = np.zeros((2, 3, 3))

        # find longest side on points
        a = vL(points[1] - points[0])  # 1, 0
        b = vL(points[2] - points[1])  # 2, 1
        c = vL(points[0] - points[2])  # 0, 2

        if (a > b) and (a > c):
            # a is longest
            # [1,2,mid(1,0)], [0,2,min(1,0)]
            m10 = np.mean(points[[1, 0]], axis=0)
            triags[0, 0], triags[0, 1], triags[0, 2] = \
                points[1], points[2], m10
            triags[1, 0], triags[1, 1], triags[1, 2] = \
                points[0], points[2], m10

        elif (b > a) and (b > c):
            # b is longest
            # [0, 2, mid(2,1)], [1,0,mid(2,1)]
            m21 = np.mean(points[[2, 1]], axis=0)
            triags[0, 0], triags[0, 1], triags[0, 2] = \
                points[0], points[2], m21
            triags[1, 0], triags[1, 1], triags[1, 2] = \
                points[0], points[1], m21

        elif (c > a) and (c > b):
            # c is longest
            # [0, 1, mid(2,0)], [1,2,mid(2,0)]
            m20 = np.mean(points[[2, 0]], axis=0)
            triags[0, 0], triags[0, 1], triags[0, 2] = \
                points[1], points[0], m20
            triags[1, 0], triags[1, 1], triags[1, 2] = \
                points[1], points[2], m20

        centers = np.zeros((np.shape(triags)[0], 3))
        for n in range(np.shape(triags)[0]):
            centers[n] = barycenter(
                triags[n])

        return (
            triags,   # 2 triags x 3 points x 3 dims
            centers)  # 2 centers x 3 dims

    cor = np.array([  # old geometry
        corners['values']['x'], corners['values']['y'],  # in m
        corners['values']['z']])  # in m
    tri_cor = {  # results of new fan
        'label': 'camera_corners',
        'values': None,
        'aperture': {},
        'angles': None}

    if N != 2:
        f = [2**n for n in range(1, math.ceil(np.sqrt(N)) + 1)]
    else:
        f = [2]
    M = np.sum(f)
    index = math.ceil(np.sqrt(N)) if (N != 2) else 1

    # 4 coords, N triangles with 3 corners and 1 center
    tri = np.zeros((4, 128, N, 4))
    angles = np.zeros((128, N * N))

    print('\t\t>> End result supposed to have ' + str(N) + ' triangles')
    apt = corners['aperture']
    for cam in cams:
        nCh = camera_info['channels']['eChannels'][cam][:]
        # 4 coords, N tris, 3 corners 1 center
        A = np.array([  # x y z of old apt
            apt[cam]['x'], apt[cam]['y'],
            apt[cam]['z']]).transpose()

        # define camplane normal
        camplane_norm = v_norm(np.cross(
            np.array([
                cor[0, nCh[0], 0] - A[0, 0],
                cor[1, nCh[0], 0] - A[0, 1],
                cor[2, nCh[0], 0] - A[0, 2]]),
            np.array([
                cor[0, nCh[-1], 0] - A[0, 0],
                cor[1, nCh[-1], 0] - A[0, 1],
                cor[2, nCh[-1], 0] - A[0, 2]])))[0]

        # triangulate the pinhole
        a_triags = np.zeros((M, 3, 3))
        a_cs = np.zeros((M, 3))
        for n in range(index):
            if n == 0:
                a_triags[:2], a_cs[:2] = two_triangles(A[1:])

            else:
                for m in range(2**n):
                    # old triangles to take from, new to put into
                    i, j = f[n - 1] - 2 + m, f[n] + (m - 1) * 2
                    a_triags[j:j + 2], a_cs[j:j + 2] = \
                        two_triangles_from_triangle(a_triags[i])

        af_triags, af_cs = a_triags[-N:], a_cs[-N:]
        if False:  # cam == 'HBCm':
            gp.plot_final_triangulation(
                base=A, triangles=af_triags, centers=af_cs, N=N,
                L='triags_aperture_' + cam)

        tri_cor['aperture'][cam] = {
            'x': None, 'y': None, 'z': None, 'r': None}
        for i, lab in enumerate(['x', 'y', 'z']):
            tri_cor['aperture'][cam][lab] = \
                np.array([  # center, and triag corners
                    af_cs[:, i],
                    af_triags[:, 0, i],
                    af_triags[:, 1, i],
                    af_triags[:, 2, i]])
        tri_cor['aperture'][cam]['r'] = \
            np.sqrt(np.square(tri_cor['aperture'][cam]['x']) +
                    np.square(tri_cor['aperture'][cam]['y']))

        d_triags = np.zeros((128, M, 3, 3))
        d_cs = np.zeros((128, M, 3))

        for n, c in enumerate(nCh):
            D = cor[:, c].transpose()

            for n in range(index):
                if n == 0:
                    d_triags[c, :2], d_cs[c, :2] = two_triangles(D[1:])

                else:
                    for m in range(2**n):
                        # old triangles to take from, new to put into
                        i, j = f[n - 1] - 2 + m, f[n] + (m - 1) * 2
                        d_triags[c, j:j + 2], d_cs[c, j:j + 2] = \
                            two_triangles_from_triangle(d_triags[c, i])

            df_triags, df_cs = d_triags[c, -N:], d_cs[c, -N:]
            if False:  # c == 0:
                gp.plot_final_triangulation(
                    base=D, triangles=df_triags, centers=df_cs, N=N,
                    L='triags_detector_eChan_' + str(c))

            for k in range(4):  # coords
                if k < 3:
                    tri[k, c, :, 0] = df_cs[:, k]
                    tri[k, c, :, 1] = df_triags[:, 0, k]
                    tri[k, c, :, 2] = df_triags[:, 1, k]
                    tri[k, c, :, 3] = df_triags[:, 2, k]
                else:
                    tri[k, c, :, :] = np.sqrt(
                        np.square(tri[0, c, :, :]) +
                        np.square(tri[1, c, :, :]))

            # get angles of LoS out of plane
            for p in range(N):  # for pinhole squares
                for d in range(N):  # for det squares
                    vec = v_norm(np.array([
                        af_cs[p, 0] - tri[0, c, d, 0],
                        af_cs[p, 1] - tri[1, c, d, 0],
                        af_cs[p, 2] - tri[2, c, d, 0]]))[0]

                    angles[c, p * N + d] = np.arcsin(np.abs(np.dot(
                        camplane_norm, vec) / (np.sqrt(np.dot(
                            camplane_norm, camplane_norm)) *
                            np.sqrt(np.dot(vec, vec)))))

    tri_cor['values'] = {
        'x': tri[0], 'y': tri[1], 'z': tri[2], 'r': tri[3]}
    tri_cor['angles'] = angles

    if saving:  # save to new file and return the new one to use
        out = mClass.dict_transf(
            tri_cor, to_list=True)
        with open(file, 'w') as outfile:
            json.dump(out, outfile, indent=4, sort_keys=False)
        outfile.close()
        tri_cor = mClass.dict_transf(
            tri_cor, to_list=False)

    return (tri_cor, vmec_label + '_tN' + str(N), N)
