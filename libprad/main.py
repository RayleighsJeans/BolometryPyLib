""" **************************************************************************
    start of file """

import os
import sys
import time as ttt
import warnings
import numpy as np

import dat_lists as lists
import webapi_access as api
import logbook_api as logbook_api
import upload_archive as upload_archive

import output as output

import prad_calculation as rad_calc
# import radiation_geometry as rad_geom
# import radiational_fraction as rad_frac

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")
warnings.filterwarnings("ignore", "Reloaded modules")

base = os.getcwd()
sys.path.append(os.path.abspath(base + '/libradcalc'))
sys.path.append(os.path.abspath(base + '/liboutput'))
sys.path.append(os.path.abspath(base + '/libwebapi'))

base_URI = 'http://archive-webapi.ipp-hgw.mpg.de/'

""" ********************************************************************** """


def main(
        program='20181010.032',
        date=None,
        shot=None,
        POSIX_from=None,  # in ns
        POSIX_upto=None,  # in ns
        epoch_time=False,
        cont=False,
        filter_method='raw',
        geom_input=None,
        strgrid=None,
        magconf=None,
        plot=False,
        return_pow=False,
        Archive=True,
        compare_archive=False,
        versionize=False,
        ssp=False):
    """ Main function of post processing bolometer routine that makes all
        the plots and calculations for the archive; power of all channelsp
        and each camera, plus comparison, surface an effective plasma radius
        plots.
    Args:
        date (str, optional): Date
        shot (int, optional): XPID
        POSIX_from (None, optional): In ns
        POSIX_upto (None, optional): -"-
        epoch_time (bool, optional): If to use POSIX stuff
        continue (bool, optional): Continue after finish
        Archive (bool, optional): Skip upload
        compare_archive (bool, optional): Channel comp. to archive links
        versionize (bool, optional): Create new archive version
        ssp (bool, optional): Sudo scaling plot
    Returns:
        None
    Notes:
        None
    """
    # optional input program or shot and date
    if (date is None) and (shot is None) and (program is not None):
        date, shot = program[0:8], int(program[-3:])

    elif (date is not None) and (shot is not None):
        program = date + "." + str(int(shot)).zfill(3)

    elif not date and not shot:
        # bools so we can run the entire dataset
        # for mostly all of OPs
        pass

    if not return_pow:
        print('\n>> Start routine for P_rad' +
              ' calculation and diagnostics ...')
    else:
        print('\n\t>> P_rad calculation for output ...')

    indent_level = '\t'
    try:
        """" begin the main compartment of this routine in which all the data
        are gathered
        make shotnumber (usually 1, but if a different shot is desired, just
        name it so) and get the date in this format YYYYMMDD """

        if not return_pow:
            print(indent_level + '>> Selecting date ...')
        [dateno, op_days, date, shot] = lists.operation_days(
            indent_level=indent_level, date=date, shot=shot)
        if (program is None):
            program = date + '.' + str(shot).zfill(3)

        it = 0
        bad_key_error = False

        # get comparison data from external file
        compare_shots, compare_data_names, comparison = \
            lists.compare(base=base, indent_level=indent_level,
                          printing=not return_pow)

        # get info on date from logbook api
        logbook_data = \
            logbook_api.logbook_json_load(
                date=date, shot=None, debug=False,
                indent_level=indent_level, printing=not return_pow)

        if not return_pow:
            print("\t>> Current date is ", date, "...")
        while (True):

            location = '../results/CURRENT'
            if not os.path.exists(location):
                os.makedirs(location)

            # get current date and make initial shotnumber
            # and complete the request number with that
            # and later interate through when success
            os.chdir(base)
            program = date + '.' + str(shot).zfill(3)
            program_info, req = api.xpid_info(program=program)

            if (str(req) == "<Response [404]>"):
                it += 1

                # the shot and date didn't match the server data base
                # now wait for, like 1 minute or something and try again, 'til
                # shot is there (hence the try/while loops)
                if (it == 1):
                    print("\t\\\  ERROR: Server returns [404]." +
                          ' Waiting for shot', program)
                if (shot > 60):
                    if date == ttt.strftime("%Y%m%d"):
                        ttt.sleep(5)
                    else:
                        print("\tThe date is not set to day",
                              "--> changing date")
                        if not dateno[1]:
                            date = ttt.strftime("%Y%m%d")
                        else:
                            dateno[0] += 1
                            date = op_days[dateno[0]]

                        print('\tDate is now set to', date,)
                        logbook_data = \
                            logbook_api.logbook_json_load(
                                date=date, shot=None,
                                debug=False, indent_level=indent_level)
                        shot = 1
                else:
                    if (it == 1):
                        print(indent_level +
                              "\\\  The shot number is low. " +
                              "We suspect A DUD?!")

                    if (date == ttt.strftime("%Y%m%d")):
                        ttt.sleep(5)
                    else:
                        if (it == 1):
                            print(indent_level + "Keep on interating...")
                        if cont:
                            shot += 1
                        else:
                            print("\n>> Done!")
                            return

            elif (str(req) == "<Response [200]>"):
                it = 0

                # the shot and date made a successful match with the server
                # and now get the data from it, stored ind program_info and
                # data_object, which is listed below
                if not return_pow:
                    print(indent_level +
                          ">> SUCCESS: Server returns shot info of", program)
                    print(
                        indent_level + "   NAME: " +
                        program_info["programs"][0]["name"] + "\n" +
                        indent_level + "   DESC.: ",
                        program_info["programs"][0]["description"])

                # wait on full data object from archive
                data_object, params_object = api.return_data(
                    date=date, shotno=str(int(shot)).zfill(3),
                    POSIX_from=POSIX_from, POSIX_upto=POSIX_upto,
                    epoch_time=epoch_time, printing=not return_pow)

                # which are the same for everything (constant)
                priority_object, bad_key_error = \
                    api.do_before_running(
                        program_info=program_info, program=program,
                        date=date, data_object=data_object,
                        geom_input=geom_input, magconf=magconf,
                        strgrid=strgrid, printing=not return_pow)

                if bad_key_error is True:
                    print(indent_level +
                          ">> Got no time dimension. We suspect a trigger",
                          "or program\n\t   failure which causes the ",
                          "absolute\n\t   " +
                          "time value not being set correctly!")
                    ttt.sleep(1)
                    shot += 1

                else:
                    # the mighty magic happening and finally uploading
                    radpower_object = rad_calc.calculation(
                        prio=priority_object, dat=data_object,
                        program_info=program_info,
                        date=date, shotno=shot, printing=not return_pow,
                        make_linoffs=True, filter_method=filter_method)
                    if return_pow:
                        return (radpower_object)

                    try:
                        # select by
                        calcstate = radpower_object['status']
                        if (calcstate == 'valid') and (Archive is not None):
                            print(indent_level + ">> All P_rad recieved, " +
                                  'status: ' + calcstate +
                                  '\n' + indent_level + '>> Uploading ...')

                            upload_archive.uploads(
                                power=radpower_object, dat=data_object,
                                prio=priority_object, debug=False,
                                versionize=versionize, upload_test=Archive,
                                date=date)

                        else:
                            print(indent_level + '>> Shall not upload to' +
                                  ' archive since calibration values' +
                                  ' are missing and upload prohibited\n',
                                  indent_level + '\tArchive:', Archive,
                                  ', calcstate:', calcstate)

                    except (IndexError, TypeError):
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(
                            exc_tb.tb_frame.f_code.co_filename)[1]
                        print(indent_level + '\\\ ', exc_type, fname,
                              exc_tb.tb_lineno, '\n' +
                              indent_level + '\t\\\  Interrupted in upload')

                    # calculation routines
                    #  print(indent_level + '>> Calculations ...')
                    # if program == '20181010.032':
                    #     fracs = [0.33, 0.66, 0.9, 1.0]
                    # else:
                    #     fracs = [0.1, 0.25, 0.35]

                    # radpower_object['radiation_fraction'] = \
                    #     rad_frac.radfrac_hexo_comp_channels(
                    #         program=program, power=radpower_object,
                    #         program_info=program_info, plot=True,
                    #         indent_level=indent_level, fracs=fracs,
                    #         time=data_object['BoloSignal']['dimensions'])

                    # radpower_object['core_v_sol'] = \
                    #     rad_geom.core_SOL_radiation(
                    #         prio=priority_object, plot=False,
                    #         magconf=magconf, strgrid=strgrid,
                    #         power=radpower_object, program=program,
                    #         program_info=program_info)

                    # radpower_object['core_v_sol']['ratios'] = \
                    #     rad_geom.core_v_SOL_ratios(
                    #         program=program, program_info=program_info,
                    #         prio=priority_object, plot=False,
                    #         power=radpower_object)

                    # output function
                    if plot:
                        output.output(
                            prio=priority_object, dat=data_object,
                            power=radpower_object, params=params_object,
                            program=program, program_info=program_info,
                            compare_shots=compare_shots, comparison=comparison,
                            compare_data_names=compare_data_names,
                            compare_archive=False, logbook_data=logbook_data)

            # iterate shot number
            if cont:
                # if the day set is current day, keep on looking
                if (ttt.strftime('%Y%m%d') == date):
                    if (str(req) == "<Response [404]>"):
                        ttt.sleep(10)
                        continue
                    elif (str(req) == "<Response [200]>"):
                        # if (bad_key_error):
                        #     ttt.sleep(1)
                        # elif not (bad_key_error):
                        shot += 1
                else:
                    shot += 1
            else:
                print("\n>>Done!")
                return ({'prio': priority_object,
                         'data': data_object,
                         'radpow': radpower_object,
                         'param': params_object})

    except KeyboardInterrupt:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print('\\\ ', exc_type, fname,
              exc_tb.tb_lineno, "\n\\\  Interrupted by Keyboard")
