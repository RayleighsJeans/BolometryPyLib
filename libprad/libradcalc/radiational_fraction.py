""" **************************************************************************
    so header """

import numpy as np
import sys
import requests
import warnings
# import archivedb

import thomsonS_access
import mClass
import plot_funcs
# import hexos_getter as hexos_get
import dat_lists
import webapi_access as api

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")

Z = np.zeros
one = np.ones

stdflush = sys.stdout.flush
stdwrite = sys.stdout.write

""" eo header
************************************************************************** """


def radiational_frac(
        program='20181010.032',
        ecrh=None,
        prad=None,
        debug=False):
    """ download and calculate the radiational fraction
    Args:
        program (str, optional): program
        power (dict, optional): power object from main
        data (dict, optional): data object download
        program_info (dict, optional): info
        time (list, optional): time vector
    Returns:
        radiational_fraction (ndarray):
        Prad
        ECRH
    """
    if (ecrh is None) or (prad is None):
        program_info, req = api.xpid_info(program=program)
        t0 = program_info['programs'][0]['trigger']['1'][0]  # in ns
        t4 = program_info['programs'][0]['trigger']['4'][0]  # in ns

    if (ecrh is None):
        try:
            # ECRH
            ecrh = api.download_single(
                api.download_link(name='ECRH'),
                program_info=None, filter=None,
                start_POSIX=t0, stop_POSIX=t4)
            ecrh['values'] = [x / 1.e3 for x in ecrh['values']]

        except Exception:
            print('\\\ failed ECRH download')
            return (None, None, None)

    else:
        if (('dimensions' not in ecrh.keys()) or
                ('values' not in ecrh.keys())):
            return (None, None, None)

    if prad is None:
        try:
            prad = api.download_single(
                api.download_link(name='PradHBCm'),
                program_info=None, filter=None,
                start_POSIX=t0, stop_POSIX=t4)
            prad['values'] = [x / 1.e6 for x in prad['values']]

            if prad['label'] == 'EmptySignal':
                prad = api.download_single(
                    api.download_link(name='Prad HBC'),
                    program_info=None, filter=None,
                    start_POSIX=t0, stop_POSIX=t4)
                prad['values'] = [x / 1.e3 for x in prad['values'][0]]

        except Exception as e:
            print('\\\ failed P_rad download')
            return (None, None, None)

    else:
        if (('dimensions' not in prad.keys()) or
                ('values' not in prad.keys())):
            return (None, None, None)

    if len(ecrh['dimensions']) > len(prad['dimensions']):
        prad['values'] = np.interp(
            ecrh['dimensions'], prad['dimensions'], prad['values'])
        prad['dimensions'] = [(t - t0) / 1.e9 for t in ecrh['dimensions']]
        ecrh['dimensions'] = [(t - t0) / 1.e9 for t in ecrh['dimensions']]
    else:
        ecrh['values'] = np.interp(
            prad['dimensions'], ecrh['dimensions'], ecrh['values'])
        ecrh['dimensions'] = [(t - t0) / 1e9 for t in prad['dimensions']]
        prad['dimensions'] = [(t - t0) / 1e9 for t in prad['dimensions']]

    f_rad = np.array([
        [(t - t0) / 1.e9 for t in prad['dimensions']],
        [x / y if ((x > .1) & (y > .1)) and
         (x < 12.) and (x / y < 2.) else np.nan
         for x, y in zip(prad['values'], ecrh['values'])]])
    return (f_rad, prad, ecrh)


def radfrac_hexo_comp_channels(
        program='20181010.032',
        power=None,  # {'none': None}
        program_info=None,  # {}
        time=None,  # in ns
        fracs=[0.33, 0.66, 0.9, 1.0],
        channels=None,
        material='C',
        indent_level='\t',
        plot=True,
        debug=False):
    """ wrapper for radiational fraction analysis, hexos and channels
        comparison fucntions
    Args:
        program (str, optional):
        fracs (list, optional):
    Returns:
        None
    """
    print(indent_level + '\t>> radiational fraction...')
    rad_fraction, prad, ECRH = radiational_frac(
        program=program, power=power,
        program_info=program_info, time=time)

    times = None
    if fracs is not None:
        times = [None] * len(fracs)
        for i, frac in enumerate(fracs):
            idx = mClass.find_nearest(rad_fraction, frac)[0]
            times[i] = prad['dimensions'][idx]

    species, photon_energies = None, None
    if material is not None:
        try:
            S = ['_II', '_IV', '_V', '_VI']
            species = [material + v for v in S]
            hexos_dat = hexos_get.get_hexos_xics(
                date=int(program[0:8]), shot=int(program[-3:]),
                mat=material, saving=True, debug=False,
                hexos=True, xics=False)[0]

            photon_energies = []
            for S in species:
                photon_energies.append(
                    1.2398 / (hexos_dat['lambda_lit'][S] / 1e3))

        except Exception:
            print('failed with species and photon photon energies')
            pass

    try:
        TS_data = thomsonS_access.thomson_scaled_grab(
            shotno=program, debug=False, scaling='gauss',
            evalStart=None, evalEnd=None, plot=False)[0]
    except Exception:
        TS_data = None

    if plot:
        plot_funcs.plot_radFraction(
            program=program, ECRH=ECRH, Prad=prad,
            rad_fraction=rad_fraction, avg_TS_datas=None,
            TS_data=TS_data, fracs=fracs, times=times,
            species=None, photon_energies=None)

    chordal_profile(
        power=power, program_info=program_info,
        time=time, program=program, fracs=fracs, times=times,
        indent_level=indent_level, best_chans=channels,
        plot=plot)

    if times is not None:
        channel_hexos_comp(
            program=program, material=material, eChannels=channels,
            times=[times[-1] - 1., times[-1] + 1.], species=species,
            photon_energies=photon_energies)

    return (rad_fraction)


def chordal_profile(
        program='20181010.032',
        power=None,
        time=None,
        reff=None,
        program_info=None,
        fracs=[0.33, 0.66, 0.9, 1.0],
        times=[0.621, 2.2205, 7.1539, 3.421],
        best_chans=[],
        cams=None,
        indent_level='\t',
        plot=True):
    """ chordial brightness profiles for times and cam
    Args:
        program (str, optional):
        times (list, optional):
        cam (str, optional):
        plot (bool, optional):
        best_chans (list, optional):
    Returns:
        chordial_profile (dict):
    """
    print(indent_level + '\t>> chordal profiles...')
    camera_info = dat_lists.geom_dat_to_json()
    if reff is None:
        reff = camera_info['radius']['reff']

    base_URI = 'http://archive-webapi.ipp-hgw.mpg.de/'
    if program_info is None:
        req = requests.get(
            base_URI + 'programs.json' + '?from=' + program)
        program_info = req.json()

    start = str(program_info['programs'][0]['from'])
    stop = str(program_info['programs'][0]['upto'])

    if cams is None:
        cams = ['HBCm', 'VBC']

    if power is not None:
        chord = power['volscaled']  # in W

    for c, cam in enumerate(cams):
        if power is None:
            try:
                chordal_profile = api.download_single(  # in W
                    base_URI + "Test/raw/W7XAnalysis/QSB_Bolometry/" +
                    'ChordalProfile_' + cam + '_DATASTREAM/V4/',
                    program_info=program_info,
                    start_POSIX=start, stop_POSIX=stop)
            except Exception:
                print('failed chrodal_profile')
                return

        else:
            nCh = reff['channels']['eChannels'][cam]
            chordal_profile = chord[nCh, :].transpose()  # in W

            if time is not None:
                chordal_profile = {
                    'values': chordal_profile,  # in W
                    'dimensions': time}  # in ns

            else:
                print('failed in chordal with time')
                return

        if plot:
            plot_funcs.chordal_profile(
                program=program, chordal_data=chordal_profile,
                best_chans=best_chans, times=times, fracs=fracs,
                camera_info=camera_info, reff=reff, cam=cam)

    if power is None:
        return (chordal_profile)
    else:
        return


def channel_hexos_comp(
        program='20181010.032',
        material='C',
        eChannels=[5, 7, 23],
        times=[3., 5.],
        plot=False,
        species=['C_II', 'C_IV', 'C_V', 'C_VI'],
        photon_energies=None):
    """ comparing channels and HEXOS lines
    Args:
        program (str, optional):
        material (str, optional):
        eChannels (list, optional):
        times (list, optional):
        plot (bool, optional):
        species (list, optional):
        photon_energies (None, optional):
    Returns:
        None
    """
    base_URI = 'http://archive-webapi.ipp-hgw.mpg.de/'
    req = requests.get(
        base_URI + 'programs.json' + '?from=' + program)
    program_info = req.json()
    start = str(program_info['programs'][0]['from'])
    stop = str(program_info['programs'][0]['upto'])

    try:  # material = 'C'
        hexos_dat = hexos_get.get_hexos_xics(
            date=int(program[0:8]), shot=int(program[-3:]),
            mat=material, saving=True,
            debug=False, hexos=True, xics=False)[0]
    except Exception:
        print('failed Prad')
        return None

    try:  # own Prad
        channel_dat = api.download_single(
            base_URI + "Test/raw/W7XAnalysis/QSB_Bolometry/" +
            'PradChannels_DATASTREAM/V2/', program_info=program_info,
            start_POSIX=start, stop_POSIX=stop)
    except Exception:
        print('failed Prad')
        return None

    if plot:
        plot_funcs.hexos_and_channels(
            program=program, material=material, eChannels=eChannels,
            times=times, hexos_dat=hexos_dat, channel_data=channel_dat,
            species=species, photon_energies=photon_energies)

    return
