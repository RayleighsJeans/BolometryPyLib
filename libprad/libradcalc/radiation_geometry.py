""" **************************************************************************
    so header """

import numpy as np
import sys
import requests
import warnings

import prad_calculation
import prad_outputs
import dat_lists
import webapi_access as api
import mClass

import LoS_emissivity3D as LoS3D
import LoS_volume as LoSv
import profile_to_lineofsight as profile

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")

Z = np.zeros
one = np.ones

stdflush = sys.stdout.flush
stdwrite = sys.stdout.write

""" eo header
************************************************************************** """


def core_SOL_radiation(
        program='20181010.032',
        program_info=None,
        prio=None,
        power=None,
        vmecID='1000_1000_1000_1000_+0000_+0000/01/00jh_l/',
        strgrid='tN4_50x30x100_1.4',
        magconf='EIM_beta000',
        filter_method='raw',
        indent_level='\t',
        plot=True,
        debug=False):
    """ core and SOL radiation levels from cameras
    Keyword Arguments:
        program {str} -- XP ID (default: {'20181010.032'})
        program_info {[type]} -- XP info (default: {None})
        prio {[type]} -- priority data (default: {None})
        power {[type]} -- calculated radiation (default: {None})
        vmecID {str} -- (def.: {'1000_1000_1000_1000_+0000_+0000/01/00jh_l/'})
        filter_method {str} -- filter method of prad (default: {'raw'})
        indent_level {str} -- printing indentation (default: {'\t'})
        debug {bool} -- debugging (default: {False})
    Returns:
        power {dict} -- appended power object with core/SOL
        time {ndarray} -- time vector
        prad {dict} -- camera core and SOL radiation
    """
    if program_info is None:
        base_URI = 'http://archive-webapi.ipp-hgw.mpg.de/'
        req = requests.get(
            base_URI + 'programs.json' + '?from=' + program)
        program_info = req.json()

    start = str(program_info['programs'][0]['from'])
    stop = str(program_info['programs'][0]['upto'])

    trigger = program_info["programs"][0]["trigger"]
    ecrh_off = (trigger["4"][0] - trigger['1'][0]) / (1e9) + 1.0

    if power is None:  # calculated power object
        try:
            req = api.download_single(
                base_URI + "Test/raw/W7XAnalysis/QSB_Bolometry/" +
                'PradChannels_DATASTREAM/V2/', program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)
            power_channels = np.array(req['values'], dtype='float')
            time = [(t - program_info["programs"][0]["trigger"]["1"][0]) / 1e9
                    for t in req['dimensions']]

        except Exception:
            print('\t\\\ power download failed')
            return

    else:
        try:
            time, power_channels = power['time'], power['power']
        except Exception:
            print('\t\\\ power import failed')
            return

    if prio is None:  # priority data input
        camera_info = dat_lists.geom_dat_to_json()
        volume_torus = 45.  # m^3

    else:
        try:
            camera_info = prio['geometry']
            volume_torus = prio['volume_torus']
        except Exception:
            print('\t\\\ prio import failed')
            return

    if ((magconf is None) or (strgrid is None)):
        magconf, strgrid = 'EIM_beta000', 'tN4_50x30x100_1.4'

    print(power_channels.__class__, np.shape(power_channels),
          power_channels.dtype)

    # get data
    volume = LoSv.store_read_volume(name=magconf + '_' + strgrid)
    emiss = LoS3D.store_read_emissivity(name=magconf + '_' + strgrid)[1]
    line_secs = LoS3D.store_read_line_sections(
        name=magconf + '_' + strgrid)
    reff, position, minor_radius = profile.store_read_profile(
        name=magconf + '_' + strgrid)
    
    N, nFS, nL = np.shape(emiss)[1:]

    print(indent_level + '\t>> core v sol: ', end='')
    prad = {'HBCm': {}, 'VBC': {}}
    for cam in ['HBCm', 'VBC']:
        print(cam, end=' - ')

        # channel numbers sorting for core and SOL
        broken, M  = \
            camera_info['channels']['droplist'], \
            np.shape(power_channels)[1]

        core_sol = Z((128, M, 2))
        core_vol, sol_vol, core_kbolo, sol_kbolo = \
            Z((128)), Z((128)), Z((128)), Z((128))

        for ch in camera_info['channels']['eChannels'][cam]:
            if ch in broken:
                continue

            for T in range(N):
                tot_line = np.sum(line_secs[ch, T])
                nZ = np.where(line_secs[ch, T] != 0.0)[:2]

                # get aperture factors
                k_facs_ch = np.mean(
                    emiss[ch, T][nZ] / line_secs[ch, T][nZ])

                for S in range(nFS):
                    for L in range(nL):
                        if line_secs[ch, T, S, L] == .0:
                            continue

                        f = line_secs[ch, T, S, L] / tot_line
                        if np.abs(reff[ch, T, S, L]) < minor_radius:
                            core_vol[ch] += volume[ch]  # m^3
                            core_kbolo[ch] += line_secs[ch, T, S, L]  # m
                            core_sol[ch, :, 0] += power_channels[ch] * f  # W

                        elif np.abs(reff[ch, T, S, L]) >= minor_radius:
                            sol_vol[ch] += volume[ch]  # m^3
                            sol_kbolo[ch] += line_secs[ch, T, S, L]  # m
                            core_sol[ch, :, 1] += power_channels[ch] * f  # W

            sol_kbolo[ch] *= k_facs_ch  # in m^2 * m -> m^3
            core_kbolo[ch] *= k_facs_ch  # in m^2 * m -> m^3

        print('core ', end='... ')
        prad[cam]['P_rad_core'], prad[cam]['volume_sum_core'] = \
            prad_calculation.calculate_prad(
                time=time, power=core_sol[:, :, 0],  # in s and W
                volume=core_vol, k_bolo=core_kbolo,  # in m^3
                volume_torus=volume_torus,  # in m^3
                channels=camera_info['channels']['eChannels'][cam],  # in m
                date=int(program[:8]), shotno=int(program[9:]), brk_chan=broken,
                camera_tag=cam + '_core', saving=True, method=filter_method,
                camera_list=camera_info['channels']['eChannels'][cam],
                indent_level=indent_level, debug=False)
        print('SOL ', end='... ')
        prad[cam]['P_rad_sol'], prad[cam]['volume_sum_sol'] = \
            prad_calculation.calculate_prad(
            time=time, power=core_sol[:, :, 1],
            volume=sol_vol, k_bolo=sol_kbolo,  # in m^3
            volume_torus=volume_torus,  # in m^3
            channels=camera_info['channels']['eChannels'][cam],  # in m
            date=int(program[:8]), shotno=int(program[9:]), brk_chan=broken,
            camera_tag=cam + '_SOL', saving=True, method=filter_method,
            camera_list=camera_info['channels']['eChannels'][cam],
            indent_level=indent_level, debug=False)
    print(' done!')

    if plot:
        prad_outputs.core_v_sol_prad(
            time=time, core_SOL=prad, ecrh_off=ecrh_off,
            program=program, program_info=program_info)
    return (prad)


def core_v_SOL_ratios(
        program='20181010.032',
        program_info=None,
        power=None,
        prio=None,
        core_v_sol=None,
        time=None,
        indent_level='\t',
        plot=True,
        debug=False):
    """ ratios of core and SOL prad values
    Keyword Arguments:
        program {str} -- XP (default: {'20181010.032'})
        program_info {[type]} -- XP info (default: {None})
        power {[type]} -- power object (default: {None})
        prio {[type]} -- prio object (default: {None})
        core_v_sol {[type]} -- core_v_sol data (default: {None})
        time {[type]} -- time vector (default: {None})
        indent_level {str} -- printing indentation (default: {'\t'})
        debug {bool} -- debugging (default: {False})
    Returns:
        power {dict} -- appended power object with core/SOL
        prad {dict} -- camera core and SOL radiation ratios
    """
    print(indent_level + '\t>> core v sol ratios: ', end='')
    base_URI = 'http://archive-webapi.ipp-hgw.mpg.de/'
    if power is not None:
        core_v_sol = power['core_v_sol']
        time = power['time']

    if program_info is None:
        req = requests.get(
            base_URI + 'programs.json' + '?from=' + program)
        program_info = req.json()

    if power is None:
        start = str(program_info['programs'][0]['from'])
        stop = str(program_info['programs'][0]['upto'])

    if prio is None:
        trigger = program_info["programs"][0]["trigger"]
        ecrh_off = (trigger["4"][0] - trigger['1'][0]) / (1e9) + 1.0
    elif prio is not None:
        ecrh_off = prio['ecrh_off']

    a, b = mClass.find_nearest(np.array(time), 0.1)[0], \
        mClass.find_nearest(np.array(time), ecrh_off)[0]

    ratios = {'HBCm': {}, 'VBC': {}}
    for cam in ['HBCm', 'VBC']:
        print(cam, end=' - ')
        if power is None:
            prad_tot = api.download_single(
                base_URI + "Test/raw/W7XAnalysis/QSB_Bolometry/" +
                'Prad' + cam + '_DATASTREAM/V2/',
                program_info=program_info,
                start_POSIX=start, stop_POSIX=stop)['values'][0]

        elif power is not None:
            prad_tot = power['P_rad_' + cam.lower()[:3]]

        print('core ', end='... ')
        ratios[cam]['core'] = [
            v / prad_tot[i] if ((a < i < b) & (prad_tot[i] > 1e5) & (v > .0))
            else 0.0 for i, v in enumerate(core_v_sol[cam]['P_rad_core'])]

        print('SOL ', end='... ')
        ratios[cam]['sol'] = [
            v / prad_tot[i] if ((a < i < b) & (prad_tot[i] > 1e5) & (v > .0))
            else 0.0 for i, v in enumerate(core_v_sol[cam]['P_rad_sol'])]

    print(' done!')
    if plot:
        prad_outputs.core_sol_ratios(
            time=time, ecrh_off=ecrh_off, ratios=ratios,
            program=program, program_info=program_info)
    return (ratios)
