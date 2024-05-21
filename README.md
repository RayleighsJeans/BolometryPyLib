Here comes the README for the QSB_Bolometry.
(formerly known as IDL2PY-PORT in mpcdf)

## LIBPRAD SUBROUTINE #########################################################
###############################################################################
Things the routine does up until now:

    0 -- constantly looking for the last show of the given/current day
         and grabing the logbook entry
         routines: main()

    1 -- acquiring and plotting the most important diagnostics, including
         ecrh, ne, Te, wdia
         routines: logbook_json_load()
                   check_bolo_data()
                   do_before_running()

    2 -- if shot was found, calculate from the archive data of voltage from
         each detector the P_rad for each camera as well
         routines; make flip fix and calculate the calibration values:
            aux_voidTest()
            foo()
            fit_parameters()
            calculate_prad()
            magic_function()
         a -- different filtering methods on the time derivative and the
              raw voltage possible
              routines:
                magic_function()

    3 -- output stuff:
         a -- overview plot, including ECRH, WDIA, T_e, n_e
              routines:
                overview_plot()
         b -- plot of perviously calculated P_rad (HBC, VBC)
              routines:
                prad_output()
         c -- output of standard deviation for each of the 113 channels
              routines:
                stddiv_output()
         d -- linear offset coefficients output from subtraction in
              calculate_prad()
              routines:
                linoffs_output
         e -- surface plot of all channels power over time and
              corresponding P_rad in one figure with
              same plot over r_eff (somewhere inverted)
              routines:
                surface_output()
                reff_plot()
         f -- look for the least powerloss / biggest W_dia /
              biggest f_rad = P_rad/P_ECRH and collecting those points together
              with the shot info from get_infoo_shot() for scaling
              investigation routine; calculate power balance:
                dens_diag()
                plot_dens_diag()
                dens_diag_dump()
                power_balance()
         g -- plots calibration values compared to archived data for
              specified date and shot:
                calib_outputs()

## LIBINVERSION ROUTINE #######################################################
###############################################################################
Things the routine does up until now:

    1 -- Grab the most common configurations from the database.py/json file and
             download/recieve the corresponding VMEC results from the archive
             of J.Geiger
             routines: invert_main()
                       import_database()
                       vmecFluxSurf()

    2 -- Create the camera geometries based on the torus hall location of the
           detector corners & middle point
             returns the 3d and 2d lines of sight as well as connection lines
             between the corners
             routines: line_of_sight_preamb()
                       construct_linesofs()
                       construct_corners()

    3 -- create a 2D mesh in the plane of the bolometer cameras
         either the cartesian way or on the basis of the prior loaded
         vmec fluxsurfaces
         routines: mesh()
                   get_cartesian_mesh()

    4 -- get effective plasma radius for the channels
         routines: calculate_reff_los()
        
         TODO: get actual r_eff based on LOS, rather than from file

    5 -- create a surrogate radiation profile on the previously achieved mesh
             routines: surrogate_radiation_profile()

    6 -- calculate the emissivity and the geometrical factors from the LOS
         angles, detector and slit areas as well as the lines inside the mesh
         cells
         routines: geometrilca_and_emissivity_factors()
                   GetGeometricalFactors()
                   have_intersection()
                   calculates_area()
                   get_normal_angles()

    7 -- output/visualization stuff
         routines: plot_geomfactors()
                   fluxsurf()
                   pez_losight()
                   pfez_los()
                   normals_plot()
                   plot_lofs_colorcoded()
