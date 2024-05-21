""" **************************************************************************
    so header """

import numpy as np
import matplotlib.pyplot as p
import plot_funcs as plot_funcs
import prad_outputs as prad_outputs
import calib_outputs as calib_outputs
import warnings
import sys

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")

stdwrite = sys.stdout.write
stdflush = sys.stdout.flush

""" eo header
************************************************************************** """


def output(
        prio={'none': {'none': None}},
        dat={'none': {'none': None}},
        power={'none': {'none': None}},
        params={'none': {'none': None}},
        logbook_data=[],
        program='20181010.032',
        program_info={'none': {'none': None}},
        comparison=[],
        compare_shots=[],
        compare_data_names=[],
        compare_archive=False,
        indent_level='\t'):
    """ Queue for all the plot functions.
    Args:
        prio (0, list): Constants and pre-defined quantities for calcs
        dat (1, list): Archive data downloaded prior
        power (2, list): Calculated radiation/bolometer properties
        logbook_data (3, list): List logbook info on the date selected/XP ID
        date (4, str): Date selected
        shotno (5, int): XP ID selected
        program_info (7, dict): HTTP resp of logbook
        compare_archive (9, bool): Bool whether to plot the archive raw data
        comparison_shots (10, list): List of XP IDs to plot with files
        comparison (11, list): Bool list what to compare at the XP ID
        data_names (12, list): File names/locations where to load
        indent_level (13, str): indentation level
    Returns:
        None.
    Notes:
        None.
    """
    stdwrite(indent_level + '>> Output routines ...\n')
    stdflush()
    indent_level = indent_level + '\t'

    plot_binary = [
        1,  # 0  prad_output
        0,  # 1  core_v_sol
        0,  # 2  core sol ratios
        1,  # 3  overview_plot
        1,  # 4  trigger_check
        0,  # 5  reff_plot
        1,  # 6  comparison_to_shot
        1,  # 7  calib_outputs
        1,  # 8  stddiv_output
        1]  # 9  linoffs_output

    if (plot_binary[0] == 1):
        # prad_picture and file output
        prad_outputs.prad_output(
            prio=prio,
            dat=dat,
            power=power,
            program=program,
            program_info=program_info,
            solo=False)

    if (plot_binary[1] == 1):
        # core versus SOL radiation
        prad_outputs.core_v_sol_prad(
            time=power['time'],
            core_SOL=power['core_v_sol'],
            ecrh_off=prio['ecrh_off'],
            program=program,
            program_info=program_info)

    if (plot_binary[2] == 1):
        # core versus SOL radiation
        prad_outputs.core_sol_ratios(
            time=power['time'],
            ratios=power['core_v_sol']['ratios'],
            ecrh_off=prio['ecrh_off'],
            program=program,
            program_info=program_info)

    if (plot_binary[3] == 1):
        # overview plot of most important stuff
        plot_funcs.overview_plot(
            last_panel='divertor load',
            valve_type='QSQ',
            prio=prio,
            dat=dat,
            power=power,
            program=program,
            program_info=program_info,
            solo=False)

    if (plot_binary[4] == 1):
        # trigger check on ECRH onset
        plot_funcs.trigger_check(
            prio=prio,
            dat=dat,
            power=power,
            params=params,
            program=program,
            program_info=program_info,
            sigma=1.0)

    if (plot_binary[5] == 1):
        # over effectivce radius
        prad_outputs.reff_plot(
            prio=prio,
            power=power,
            program=program,
            program_info=program_info)

    if (plot_binary[6] == 1):
        # making individual channel plots and comparison to other data
        plot_funcs.comparison_to_shot(
            prio=prio,
            dat=dat,
            power=power,
            program=program,
            program_info=program_info,
            compare_archive=False)

    if (plot_binary[7] == 1):
        # plot calibration properties, current timelines and ref/meas foil
        calib_outputs.output(
            prio=prio,
            dat=dat,
            power=power,
            program=program,
            program_info=program_info)

    if (plot_binary[8] == 1):
        # standard deviation over chanel number output
        prad_outputs.stddiv_output(
            prio=prio,
            dat=dat,
            power=power,
            program=program,
            program_info=program_info)

    if (plot_binary[9] == 1):
        # linear offset slopes and corrections
        prad_outputs.linoffs_output(
            prio=prio,
            dat=dat,
            power=power,
            program=program,
            program_info=program_info)

    p.close("all")
    return
