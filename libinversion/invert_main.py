""" **************************************************************************
    start of file """

import os
import numpy as np

import database_inv as database
import fluxsurfaces as vmec
import camera_geometries as cgeom
import poincare

import mesh_2D as mesh_2D
import mesh_3D as mesh_3D

import LoS_volume as vol
import LoS_emissivity3D as emiss3D
import LoS_emissivity2D as emiss2D

import profile_to_lineofsight as profile
import profile_to_lint as profile_LoS

import factors_geometry_plots as fgp
from matplotlib.pyplot import *

# import mClass
# import slanted_fluxsurfaces as sflxs
# import geometry_factors as geom_facs

import warnings
warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")
warnings.filterwarnings("ignore", "Reloaded modules")
warnings.filterwarnings("ignore", "RuntimeWarning")

""" eo header
************************************************************************** """


def main(
        nPhi=40,
        nL=20,
        nFS=20,
        vpF=1.3,
        N=[1, 2],  # 4
        tilt_deg=1.,
        error_scale=0.0001,  # 0.1 mm (0.5 mm)
        VMID='EIM_000',
        interp_method='square',  # triang
        new_type=None,  # 'HBCm'
        cartesian=False,
        add_camera=False,
        artificial_HBCm=False,
        fix_LoS=False,
        centered=False,
        symmetric=False,
        tilt=False,
        random_error=False,
        plot=False,
        debug=False):
    """ main inversion function head routine
    Returns:
        None.
    """
    # database
    databs = database.import_database()
    # fluxsurface
    FS = databs['values']['magnetic_configurations'][VMID]
    vmec_label = FS['name']

    # from where to where goes the F
    angle_pos = np.linspace(100.0, 115.0, nPhi)

    # get fluxsurfaces, plot, save and/or load
    fluxsurfaces = vmec.fluxsurfaces(  # fs data in m
        vmec_label=vmec_label, FS=FS,
        angle_pos=angle_pos, plot=plot, saving=True)

    # get camera geometries, plot save and/or load
    camera_info, geometry, interpolated, LoS, vmec_label, N = cgeom.c_geom(
        plot=plot, interp_method=interp_method, N=N,
        tilt_deg=tilt_deg, symmetric=symmetric, tilt=tilt,
        fix_LoS=fix_LoS, centered=centered, random_error=random_error,
        error_scale=error_scale, new_type=new_type,
        artificial=artificial_HBCm, add_camera=add_camera,
        vmec_label=vmec_label, saving=True, debug=False)

    if add_camera:
        cams = ['HBCm', 'VBCl', 'VBCr', 'ARTf']
    else:
        cams = ['HBCm', 'VBCl', 'VBCr']

    mesh3D, vmec_label = mesh_3D.setup3D_mesh(
        vpF=vpF, nFS=nFS, nL=nL, nPhi=nPhi, debug=False, saving=True,
        fluxsurfaces=fluxsurfaces, vmec_label=vmec_label)

    mesh2D = mesh_2D.setup2D_mesh(
        nPhi=nPhi, nFS=nFS, nL=nL, cams=cams, vmec_label=vmec_label,
        mesh3D=mesh3D, tor_phi=angle_pos, geometry=geometry,
        camera_info=camera_info, saving=True, debug=False)

    raw_volume, old_volume = vol.get_volume(
        corners=interpolated, old=geometry,
        mesh=mesh3D, cams=cams, vpF=vpF,
        camera_info=camera_info, vmec_label=vmec_label,
        new_type=new_type, saving=True, debug=False)

    volume, old_volume = vol.scale_volume(
        raw_volume=raw_volume, old_volume=old_volume,
        corners=interpolated, cams=cams,
        camera_info=camera_info, vmec_label=vmec_label,
        new_type=new_type, saving=True, debug=True)

    line_sections3D = emiss3D.get_lines3D(
        geometry=interpolated, mesh=mesh3D, cams=cams,
        camera_info=camera_info, N=N, vmec_label=vmec_label,
        new_type=new_type, saving=True, debug=False)

    line_sections2D = emiss2D.get_lines2D(
        LoS=LoS, mesh=mesh2D, cams=cams, camera_info=camera_info,
        N=N, vmec_label=vmec_label, saving=True, debug=False)

    factors, emissivity3D = emiss3D.get_emissivity(
        lines=line_sections3D, corners=interpolated, input_shape='3D',
        camera_info=camera_info, cams=cams, vmec_label=vmec_label,
        saving=True, overwrite=False, debug=False)

    emissivity2D = emiss3D.get_emissivity(
        lines=line_sections2D, corners=interpolated, input_shape='2D',
        camera_info=camera_info, cams=cams, vmec_label=vmec_label,
        saving=True, overwrite=False, debug=False)[1]

    reff, pos_lofs, minor_radius = \
        profile.effective_radius2D(
            mesh_data=mesh2D, vmec_label=vmec_label, lines=line_sections3D,
            vmec_ID=FS['URI'], w7x_ref=FS['w7x_ref'], cams=cams,
            camera_info=camera_info, new_type=new_type,
            debug=False, saving=True)

    reff_LoS = profile_LoS.reff_of_LOS(
        camera_info=camera_info, cams=cams, reffs=reff,
        position=pos_lofs, lines=line_sections3D,
        emissivity=emissivity3D, new_type=new_type,
        mag_ax=mesh2D['values']['fs']['108.'][:, 0, 0])

    if debug:
        location = '../results/INVERSION/PROFILE/' + vmec_label
        if not os.path.exists(location):
            os.makedirs(location)

        fgp.LOS_K_factor(
            camera_info=camera_info, cameras=cams,
            emiss=np.sum(emissivity3D, axis=(1, 2, 3)),
            volum=np.sum(volume, axis=(1)),
            full_volum=np.sum(raw_volume, axis=(1)),
            daz_K=camera_info['geometry']['kbolo'],
            daz_V=camera_info['geometry']['vbolo'],
            label=vmec_label, show=False)

        fgp.angles_area(
            factors=factors['values'], camera_info=camera_info,
            cameras=cams, label=vmec_label)

        fgp.LoS_reff(
            reff=reff_LoS, camera_info=camera_info, cams=cams,
            label=vmec_label, minor_radius=minor_radius, show=False)

        # load islands and divertor/vessel
        name = 'phi108.0_res10_conf_lin'
        islands = poincare.store_read(name=name, arg='islands1_')
        divertors = poincare.define_divWall(phi=108.)

        fgp.FS_channel_property(
            property=emissivity3D * (1.e3)**3, lines=line_sections3D,
            # LoS_r=LoS['values']['rz_plane']['range'],
            # LoS_z=LoS['values']['rz_plane']['line'],
            mesh=mesh2D['values']['fs']['108.'],
            islands=islands, divertors=divertors,
            nFS=nFS, nL=nL, N=N,
            channels=[int(x) for x in np.linspace(0, 127, 128)],
            property_tag='FS_emiss3D_cell',
            cam='full', label=vmec_label, x_label='R [m]',
            y_label='Z [m]', c_label='$T$ [mm$^{3}$]',
            show_LOS=False, show=False, debug=False)

        for cam in cams[:]:
            fgp.FS_channel_property(
                property=reff / minor_radius, lines=line_sections3D,
                # LoS_r=LoS['values']['rz_plane']['range'],
                # LoS_z=LoS['values']['rz_plane']['line'],
                mesh=mesh2D['values']['fs']['108.'],
                islands=islands, divertors=divertors,
                nFS=nFS, nL=nL, N=N,
                channels=camera_info['channels']['eChannels'][cam],
                property_tag='FS_reff_cell',
                cam=cam, label=vmec_label, x_label='R [m]',
                y_label='Z [m]', c_label='radius [r$_{a}$]',
                show_LOS=False, show=False, debug=False)

            for proj, prop in zip(
                    ['3D', '2D'],
                    [line_sections3D, line_sections2D]):
                fgp.FS_channel_property(
                    property=prop, lines=line_sections3D,
                    # LoS_r=LoS['values']['rz_plane']['range'],
                    # LoS_z=LoS['values']['rz_plane']['line'],
                    mesh=mesh2D['values']['fs']['108.'],
                    islands=islands, divertors=divertors,
                    nFS=nFS, nL=nL, N=N,
                    channels=camera_info['channels']['eChannels'][cam],
                    property_tag='FS_line' + proj + '_cell',
                    cam=cam, label=vmec_label, x_label='R [m]',
                    y_label='Z [m]', c_label='L (' + proj + ') [m]',
                    show_LOS=False, show=False, debug=False)

            for proj, prop in zip(
                    ['3D', '2D'],
                    [emissivity3D, emissivity2D]):
                fgp.FS_channel_property(
                    property=prop * (1.e3)**3, lines=line_sections3D,
                    # LoS_r=LoS['values']['rz_plane']['range'],
                    # LoS_z=LoS['values']['rz_plane']['line'],
                    mesh=mesh2D['values']['fs']['108.'],
                    islands=islands, divertors=divertors,
                    nFS=nFS, nL=nL, N=N,
                    channels=camera_info['channels']['eChannels'][cam],
                    property_tag='FS_emiss' + proj + '_cell',
                    cam=cam, label=vmec_label, x_label='R [m]',
                    y_label='Z [m]', c_label='T (' + proj + ') [mm$^{3}$]',
                    show_LOS=False, show=False, debug=False)

        fgp.channel_property_sum(
            property=line_sections3D, lines=line_sections3D,
            camera_info=camera_info, cameras=cams,
            property_tag='channel_line3D_sum',
            cam='HBCm', label=vmec_label, x_label='channel no [#]',
            y_label='L(3D) [m]', show=False, debug=False)

    return (
        mesh2D,           # 0
        mesh3D,           # 1
        raw_volume,       # 2
        old_volume,       # 3
        volume,           # 4
        line_sections2D,  # 5
        line_sections3D,  # 6
        emissivity2D,     # 7
        emissivity3D,     # 8
        factors,          # 9
        reff,             # 10
        pos_lofs,         # 11
        minor_radius,     # 12
        reff_LoS,         # 13
        geometry,         # 14
        interpolated,     # 15
        LoS,              # 16
        vmec_label)       # 17
