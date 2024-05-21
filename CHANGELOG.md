Here comes the thorough changelog for the IDL2PY-PORT.


**** CHANGELOG OF THE LIBINVERSION BRANCH ************************************
____ VERSION _______ DESCRIPTION _____________________________________________

    v 1.0.0           loading fluxsurfaces from the VMEC archive (@J.Geiger)

    v 1.5.0           plots and visualization of fluxsurfaces and cameras
                      also: fluxsurface library for later tubes where the
                            line of sight plane intersects

    v 2.0.0           grid mesh creation based on the previously loaded
                      fluxsurfaces OR just a regular cartesian mesh2d

    v 2.5.0           geometrical factors based on the lines of sight in 2D
                      and the mesh cells previously defined

    v 3.0.1           proper docstrings and commentary


**** CHANGELOG OF THE LIBPRAD MASTER *****************************************
____ VERSION _______ DESCRIPTION _____________________________________________

    ??/??/???
    v 1.0.1         current 'completed' state after OP1.2a (& DPG FJT 2018)
                    see: README.md

    ??/??/???
    v 1.0.2         added OP1.1 list of operation days to calculate stuff to
                    extend to op_days for next campaign possibly as well

    ??/??/???
    v 1.1.0         added comparison in P_rad plot to archive version from
                    daihong at [HOME]/Test/raw/W7XAnalysis/BoloTest4

    ??/??/???
    v 3.0.0         large changes overall:
                        >> rework the whole docstring situation and code
                           commentary with inputs, outputs cleaned up
                        >> reduced functions down to actual appointments,
                           especially plot routines and diagnostics
                        >> re-did density/power-searching routine
                           where the max. properties are looked up and
                           then coincedented with the other vectors of
                           important experiment parameters
                        >> enclosed the failing parts, due to archive
                           throwbacks and the alike, with exception
                           handlers that catch anything and
                           return the errors with lines and files
                           to debug
                        >> digital spiking due to aquisition problems are
                           potentially removed in the magic_functions
                           by a quick fix created -> threshold approach
                           and N point width inter/extrpolation of nearest
                           data
                        >> tried epoch time download and archive access

                        FLIP FIX:
                        >> since the absolute value is a non-uniquely
                           and irreversible mathematical operation, data
                           from the 20180822.xxx - 20180905.xxx in the
                           archive has to be corrected for flipping
                           when crossing/'touching' the null line
                           during acquisition
                        >> added for both cameras in the prad calc routine
                        >> two separate functions, looking for a flip in
                           a given signal vector and a second mother routine
                           that repeats that twice for one signal/channel
                           and evaluates if this is probable

                        CALIBRATION:
                        >> added an offline calibration value routine
                           which fits an exponentially declining curve
                           to the first current step in the calibration
                           measurement and therefore calculates the
                           resistance, heat capacity and cooling time
                           of the foils according to L.Giannone et al.
                        >> plots a selection or all of the calibration current
                           vectors compared to the fit, with results included
                        >> gets uploaded to archive attached as parlog to
                           the calibration results

                        >> smaller adjustments to the mClass functions
                           and the startup import routine
                        >> added measurements of the individual bridge
                           resistances R_12, R_14, R_23, ... from
                           after the campaign as *.json
                        >> redid sudo scaling plot according to the
                           format of the new density diag routine

                        COMBINATORY:
                        >> set up a first framework and selective channel
                           combinations for future works with geometrical
                           investigations

    09/09/2019
    v 5.0.1         big skip, forgot docs:

                        >> flip fix, goes like this:
                             1. sets up or loads previous flip
                                fix database as json with shots,
                                channels and positions/values
                             2. if the same for given XPID, repeat
                                flips and levels as given per
                                channels, if all channels exist
                                go to 5., if not, go to 3. for
                                missing channel numbers
                             3. display the line and grab two clicks,
                                if same, proceed without flipping
                                if different locations, take closest
                                level and flip appropriately
                                upwards
                             4. check and display results, if ok
                                continue to possible second flip
                                else retry or skip second one
                             5. if done with all channels, return
                                fixed voltage and save flipfix database as
                                json

                        >> XICS, HEXOS, heatload:
                            1. now capable of loading the XICS Te and Ti
                               profiles and calc. the mean, div. etc.
                            2. can load the HEXOS datalines according
                               to the linesdata file from Birger,
                               filtered by the material specified and
                               get them from the according cameras
                            3. can load and add up the heat load from the
                               IR cameras pointing at the W7X targets around
                               the torus, asymmetry not included

                        >> combinatory:
                            - training routine set up with differing eval
                              methods to cross check the values
                            - distinguishing between best combinations,
                              best individual channels, and the worst
                              vice versa
                            - benchmarking with individual channels against
                              extrapolation with more from selection
                            - iteration through many amounts of channels,
                              and more then a 5 thousand combinations
                              across all selection amounts
                            - diagnosing the channel numbers with location in
                              relation to fluxsurfaces (currently just EIM)

                        >> redone the whole download section:
                            - modular download and logbook structure, can be
                              used idividually now and finally returns something that makes sense
                            - just appendable logbook info pieces, filtered
                              by humanly readable names

                        >> thomson scattering data and profiles:
                            - searching, building, scaling and plotting the
                              delicate thomson scattering data, currently
                              UNDER CONSTRUCTION


**** TODOS FOR ALL VERSIONS **************************************************
____ NR _______ DESCRIPTION __________________________________________________

    [01]        add versionion via the api over
                http://archive-webapi.ipp-hgw.mpg.de/
                more precisely
                For the creation of a new version you need a version object
                containing the following attributes:
                {
                  "versionInfo" : [
                    {
                      "reason": "version control test",
                      "producer": "micg",
                      "code_release": "v1.0",
                      "analysis_environment": "PC-OP-WS-7"
                    }
                  ]
                }
                and send such a JSON message via POST to a new datastream or
                parlog address, such as:
                Test/raw/Test/W7X/VersionTest/Test_DATASTREAM/_versions.json

    [02]        create routine that finds the best fitting ~5 channels to
                predict the total P_rad

    [03]        intertwine the found channels as selection, the radiation
                intensity lines from the individual intrinsic impurities and
                the temperature/density profiles as a means to link the local
                radiation sensitivity to a more profound understanding of
                the radiation power loss of the plasma


**** STATUS OF TODOS *********************************************************
____ NR _______ DESCRIPTION __________________________________________________

    [01]a
    >>>>        Status: Done.
                In libwebapi now exists a versioning module that looks up,
                creates and keeps the link version in check of whatever
                location you hand it over. Also creates entirely new
                archive links to put data in.
                Accordingly, the upload and load functions have been adapted
                to load up to/down from the most current locations.

    [01]b
    >>>>        The upload in itself has been changed to accomodate the
                largely different input data formats. Parlogs and data are
                created independenly and then uploaded by a single routine
                that dumps it into the archive.

    [02]
    >>>>        done for more and less than 5 channels, more eval. methods and
                full analysis with best/worst channels/combos and their
                location in regards to the EIM fluxsurface configuration
