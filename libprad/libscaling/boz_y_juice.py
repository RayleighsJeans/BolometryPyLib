""" **************************************************************************
    start of file """

import warnings
import numpy as np
import pandas as pd

import plot_funcs as pf

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")
warnings.filterwarnings("ignore", "Reloaded modules")

""" ********************************************************************** """


def loading_pds_boz(
        debug=False):
    """Summary
        Load path strings for juice files from boz datashare
    Args:
        debug (bool, optional): Debugging placeholder
    Returns:
        files (list, len=2xY): List of avocado/apricot file names
    """

    avocado = [
        'thomson_fit_step_0.10_v201903171930.csv',
        'sppts_comb_fit_step_0.10_v201903242330.csv',
        'bolotest6_step_0.10_v201901311830.csv',
        'double_pellets_v201903151730.csv',
        'equilibrium_20180807_to_20181018_step_0.20_v201903081930.csv',
        'equilibrium_20180807_to_20181018_step_0.20_v201903091200.csv',
        'equilibrium_20180807_to_20181018_step_0.20_v201903222330.csv',
        'op12a_step_0.10_v201903242100.csv',
        'op12a_thomson_fit_step_0.10_v201903242100.csv',
        'op12bpreboron_step_0.10_v201903242100.csv',
        'pelletinfo_step_0.10_v201903151130.csv',
        'postboronization_20180807_to_20181018_step_0.10_v201901281300.csv',
        'postboronization_20180807_to_20181018_step_0.10_v201901311840.csv',
        'postboronization_20180807_to_20181018_step_0.10_v201903151140.csv',
        'postboronization_20180807_to_20181018_step_0.10_v201903151700.csv',
        'postboronization_20180807_to_20181018_step_0.10_v201903171400.csv',
        'sppts_comb_fit_step_0.10_v201903242330.csv',
        'sppts_comb_profile_step_0.10_v201903211000.csv',
        'sppts_fit_step_0.10_v201903242030.csv',
        'sppts_profile_step_0.10_v201903211000.csv',
        'thomson_fit_step_0.10_v201903092350.csv',
        'thomson_fit_step_0.10_v201903171930.csv',
        'thomson_step_0.10_v201903171400.csv',
        'thomsonv10_step_0.10_v201901311830.csv',
        'xics_line_integrated_profile_step_0.10_v201903201800.csv',
        'xics_reduced_profile_step_0.10_v201902051730.csv']

    apricot = [
        'sppts_comb_fit_step_0.20_v201903231030.csv',
        'postboronization_20180807_to_20181018_step_0.20_v201901311300.csv',
        'bolotest6_step_0.20_v201901291430.csv',
        'cxrs_dualgauss_step_0.20_v20190304.csv',
        'cxrs_twoBranch-step_0.20_v20190306.csv',
        'double_pellets_v201903151730.csv',
        'equilibrium_20180807_to_20181018_step_0.20_v201903081400.csv',
        'equilibrium_20180807_to_20181018_step_0.20_v201903091100.csv',
        'equilibrium_20180807_to_20181018_step_0.20_v201903222300.csv',
        'nbi_neutraliser_step_0.20_v201901291430.csv',
        'op12a_step_0.20_v201903232330.csv',
        'op12a_thomson_fit_step_0.20_v201903232330.csv',
        'op12bpreboron_step_0.20_v201903232030.csv',
        'postboronization_20180807_to_20181018_step_0.20_v201901281000.csv',
        'postboronization_20180807_to_20181018_step_0.20_v201901291530.csv',
        'postboronization_20180807_to_20181018_step_0.20_v201901311300.csv',
        'postboronization_20180807_to_20181018_step_0.20_v201903151800.csv',
        'postboronization_20180807_to_20181018_step_0.20_v201903191000.csv',
        'sppts_comb_fit_step_0.20_v201903231030.csv',
        'sppts_comb_profile_step_0.20_v201903214000.csv',
        'sppts_fit_step_0.20_v201903231030.csv',
        'sppts_step_0.20_v201903201300.csv',
        'thomson_fit_step_0.20_v201902122345.csv',
        'thomson_fit_step_0.20_v201902281220.csv',
        'thomson_fit_step_0.20_v201903222300.csv',
        'thomson_step_0.20_v201903182359.csv',
        'thomsonv10_step_0.20_v201901311300.csv',
        'xics_line_integrated_profile_step_0.20_v201903201300.csv',
        'xics_reduced_profile_step_0.20_v201902051400.csv']

    avocado = ['juice/juice/avocado/adb_juice_' + file for file in avocado]
    apricot = ['juice/juice/apricot/adb_juice_' + file for file in apricot]

    files = []
    files.extend(avocado)
    files.extend(apricot)

    return files


def get_file_pd_dict(
        file=None,
        debug=False,
        head='hebeam',
        F=[7, 9, 11, 12, 13, 14, 15, 27, 36, 38, 39, 40, 41, 42, 43]):
    """Summary
         Loading file previously defined and sorting for columns I want
    Args:
        file (None, optional): Placeholder filename
        debug (bool, optional): Debugging placeholder
        F (list, optional): Index list of files with interesting content
    Returns:
        data (unknown): Loaded data
        inds (list): List of indices where filter applied
    todos:
        get ONLY interesting files and apply filter, then return values as x
    """
    files = loading_pds_boz()

    if not F:
        i, F = 0, []
        print('\t>> locating interesting column heads...', end='\n')

        for file in files:
            target = '../../' + file
            x = pd.read_csv(target, index_col=0)

            y = [i for i, x in enumerate(x.columns.values) if head in x]
            if (y != []):
                F.extend([i])
            i += 1

            print('\t>> files with relevant column heads (' +
                  head + '):', F, end='\n')

            return F

    elif F:

        print('\t>> reading data according to column head (' +
              head + ')', end='\n')

        xpid, t, Prad = [], [], []

        for j in F[0:1]:
            x = pd.read_csv('../../' + files[j],
                            index_col=0, low_memory=False)
            inds = filterX(
                x=x,
                shot_with_hebeam=0,
                t_hebeam_time=1.,
                t_hebeam_margin=0.3,
                pradMW_hbc_max=1e5,
                pradMW_hbc_min=-1e5)

            xpid.extend([np.int(
                s.replace('W7X', '').replace('.', ''))
                for s in x[inds]['shot'].tolist()])
            t.extend(x[inds].index.values.tolist())
            Prad.extend(x[inds]['prad_hbc_MW'].tolist())

        pf.bozy_plot()

        return xpid, t, Prad


def filterX(
        x=pd.DataFrame(),
        t_in_shot=.2,
        t_shot_length=.1,
        shot_with_pellets=0,
        t_pellet_timing=.1,
        shot_with_tespel=0,
        t_tespel_timing=.1,
        shot_with_lbo=0,
        t_lbo_timing=.1,
        shot_with_nbi=0,
        t_nbi_timing=.1,
        shot_with_hebeam=1,
        t_hebeam_time=0.1,
        t_hebeam_margin=0.2,
        main_HE=0,
        main_NE=0,
        main_N2=0,
        ecrh_MW=.25,
        line_dens1e19=.25,
        line_dens1e19_var=1.,
        wdiaMJ_var=1.,
        wdiaMJ_lim=1.,
        line_dens1e19_lim=1.,
        pradMW_hbc_max=12.,
        pradMW_hbc_min=.1):
    """Summary
        Filtering the given pandas dataframe from previously
    Args:
        x (pandas dataframe, None, optional): dataframe for filtering
        t_in_shot (float, optional): time rel after T1 in s
        t_shot_length (float, optional): time before end of XP
        shot_with_pellets (int, optional): (0/1) if shot with pellets
        t_pellet_timing (float, optional):
            (pos) time before pellet, (neg) time after pellet
        shot_with_tespel (int, optional): (0/1) if shot with tespel
        t_tespel_timing (float, optional):
            (pos) time before tespel, (neg) time after tespel
        shot_with_lbo (int, optional): (0/1) if shot with lbo
        t_lbo_timing (float, optional):
            (pos) time before lbo, (neg) time after lbo
        shot_with_nbi (int, optional): (0/1) if shot with nbi
        t_nbi_timing (float, optional):
            (pos) time before nbi, (neg) time after nbi
        shot_with_hebeam (int, optional): (0/1) if shot with hebeam
        t_hebeam_time (float, optional): duration of hebeam puff
        t_hebeam_margin (float, optional): timing margin around hebeam
        main_HE (int, optional): (0/1) if DCH helium
        main_NE (int, optional): (0/1) if DCH neon
        main_N2 (int, optional): (0/1) if DCH nitrogen
        ecrh_MW (float, optional): amount of power on
        line_dens1e19 (float, optional): line integrated density in 1e19 m^-2
        line_dens1e19_var (float, optional):
            (ne^-1 * d_ne/dt) measure of variation in line density
        wdiaMJ_var (float, optional):
            (wdia^-1 * d_wdia/dt) measure of variation in diamag energy
        wdiaMJ_lim (float, optional):
            ((max_wdia - min_wdia) / wdia) limit of wdia in ref to max/min
        line_dens1e19_lim (float, optional):
            ((max_ne - min_ne) / ne) limit of ne in ref to max/min
        pradMW_hbc_max (float, optional): rad power horizontal camera
        pradMW_hbc_min (float, optional): rad power horizontal camera
    Returns:
        inds (list): List of indices (shots, POSIX times) where applicable
    todos:
        find better filter than the usual easy one from README
    """
    inds = (
        (x["t_in_shot"] > t_in_shot) &
        (x["t_shot_stop"] - x["t_in_shot"] > t_shot_length) & (
            (x["shot_with_pellets"] == shot_with_pellets) |
            (x["t_in_shot"] < x["t_pellet_start"] - t_pellet_timing)) & (
            (x["shot_with_tespel"] == shot_with_tespel) |
            (x["t_in_shot"] < x["t_tespel"] - t_tespel_timing)) & (
            (x["shot_with_lbo"] == shot_with_lbo) |
            (x["t_in_shot"] < x["t_lbo_1"] - t_lbo_timing)) & (
            (x["shot_with_nbi"] == shot_with_nbi) |
            (x["t_in_shot"] < x["t_nbi_start"] - t_nbi_timing)) &
        (((x["shot_with_hebeam"] == shot_with_hebeam) &
            (x["t_hebeam_stop"] - x["t_hebeam_start"] > t_hebeam_time)) |
            (x["t_hebeam_start"] > x["t_in_shot"] + t_hebeam_margin) |
            (x["t_hebeam_stop"] < x["t_in_shot"] - t_hebeam_margin)) &
        (x["shot_with_main_he_injection"] == main_HE) &
        (x["shot_with_main_ne_injection"] == main_NE) &
        (x["shot_with_main_n2_injection"] == main_N2) &
        (x["ecrh_total_MW"] > ecrh_MW) &
        (x["neL_1e19_m2"] > line_dens1e19) &
        (np.abs(x["dneL_dt_1e19_m2_s"] /
                x["neL_1e19_m2"]) < line_dens1e19_var) &
        (np.abs(x["dwdia_dt_MJ_s"] / x["wdia_MJ"]) < wdiaMJ_var) &
        (np.abs(x["max_neL_1e19_m2"] - x["min_neL_1e19_m2"]) /
            x["neL_1e19_m2"] < line_dens1e19_lim) &
        (np.abs(x["max_wdia_MJ"] - x["min_wdia_MJ"]) /
            x["wdia_MJ"] < wdiaMJ_lim) &
        (x["prad_hbc_MW"] < pradMW_hbc_max) &
        (x["prad_hbc_MW"] > pradMW_hbc_min))

    return inds
