""" **************************************************************************
    so header """

import time as ttt
import datetime
import warnings
import numpy as np
import multichannel_upload as multich
import versioning as vers

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")

""" eo header
************************************************************************** """


def uploads(
        power={'none': None},
        dat={'none': None},
        prio={'none': None},
        versionize=False,
        debug=False,
        upload_test=True,
        date='20181010',
        indent_level='\t'):
    """ Upload parent function to set up data, upload and versionize.
    Args:
        power (0, list): radiated power and adjustments etc.
        dat (1, list): Downloaded objects from the archive
        prio (2, list): Predefined and constant values
        versionize (3, bool): Bool archive version is to be checked
        date (4, str): current date
        raw_time (5, list): Time vector
        indent_level (6, str): Indentation level
    Returns:
        versionize (0, bool): skip the consecutive runs, hence turn to False
    Notes:
        None
    """
    if upload_test:
        # print(indent_level + '\t\\\ upload test to SandboxDB initiated')
        # base_URI = 'http://archive-webapi.ipp-hgw.mpg.de/' + \
        #     'Sandbox/raw/W7XAnalysis/'

        print(indent_level + '\t\\\ upload test to W7XAnalysis initiated')
        base_URI = 'http://archive-webapi.ipp-hgw.mpg.de/' + \
            'Test/raw/W7XAnalysis/'
    elif not upload_test:
        print(indent_level + '\t\\\ no upload/test')
        return (versionize)

    if versionize:
        reason = 'Improved calibration procedure; ' + \
            'changed time basis individually for different ' + \
            'sample times based off of D. Fu\'s labtests to account for ' + \
            'system load stretching; new geometry input from AS ' + \
            'measured pinhole position; K and volume factors from ' + \
            'own geometry calculations in 30x20x150 grid to 1.3x ' + \
            'inflated plasma; standard configuration EIM beta=0.0, ' + \
            'ref 1; 31.56m^3 x 1.3 = 41.028m^3; manual fix for ' + \
            'flipped channels when bugged DAQ routine was used'
        vers.increase_versioning(
            indent_level=indent_level + '\t',
            user='pih', reason=reason,
            environment='PC-CLD-WS-535, Win10 Py3.7.4',
            base=base_URI[37:], specifier='QSB_Bolometry/')
        versionize = False
    raw_time = dat['BoloSignal']['dimensions']  # time POSIX

    indent_level = indent_level + '\t'
    [upload_object_dat, upload_object_par,
     corrected_object_dat, corrected_object_par] = \
        set_up_data(power=power, dat=dat, prio=prio, raw_time=raw_time)

    links = \
        ['PradHBCm', 'PradVBC', 'PradChannels', 'BoloAdjusted', 'LInAlch',
         'LInSXRh', 'LInSXRv', 'LInVBCl', 'LInVBCr', 'LInHBCm',
         'MKappa', 'MTau', 'MRes', 'RKappa', 'RTau', 'RRes',
         'ChordalProfile_HBCm', 'ChordalProfile_VBC']

    links_dat = ['QSB_Bolometry/' + x + '_DATASTREAM/' for x in links]
    links_par = ['QSB_Bolometry/' + x + '_PARLOG/' for x in links]
    [dat_stat_arr, par_stat_arr] = [['none'] * len(links)] * 2

    for n in range(0, len(links_dat)):
        if debug:
            print(indent_level + '>> Writing to ' +
                  links_dat[n].replace(base_URI, ""), end=' ')

        [dat_stat_arr[n], par_stat_arr[n]] = multich.upload_archive(
            links=[links_dat[n], links_par[n]], data=upload_object_dat[n],
            parms=upload_object_par[n], indent_level=indent_level,
            base_URI=base_URI)
        if debug:
            print(' DATA: ' + dat_stat_arr[n] +
                  ' PARAM: ' + par_stat_arr[n], end='\n')

    if (dat_stat_arr == ['GREEN'] * len(links)) and \
       (par_stat_arr == ['GREEN'] * len(links)):
        print(indent_level + '>> Writing succeeded at QSB_Bolometry ...')

    elif (dat_stat_arr == ['FAIL'] * len(links)) and \
         (par_stat_arr == ['FAIL'] * len(links)):
        print(indent_level + '>> Failed all uploads at QSB_Bolometry ...')

    else:
        print(indent_level + '>> Failed to upload:', end=' ')
        for n in range(0, len(links)):
            if (dat_stat_arr[n] != 'GREEN'):
                print(links[n] + '_DATASTREAM', end=' ')
            if (par_stat_arr[n] != 'GREEN'):
                print(links[n] + '_PARLOG', end=' ')

    if (int(date) <= 20180927):
        corrected_links = \
            ['BoloCalibMeasFoilCurrent', 'BoloCalibRefFoilCurrent']
        [corrected_links_dat, corrected_links_par] = \
            [['QSB_Bolometry/' + x + '_DATASTREAM/' for x in corrected_links],
             ['QSB_Bolometry/' + x + '_PARLOG/' for x in corrected_links]]
        [corstat_arr, corpar_arr] = [['none'] * len(corrected_links)] * 2

        for n in range(0, len(corrected_links_dat)):
            if debug:
                print(indent_level + '>> Writing to ' +
                      corrected_links_dat[n].replace(base_URI, ""), end=' ')

            [corstat_arr[n], corpar_arr[n]] = multich.upload_archive(
                links=[corrected_links_dat[n], corrected_links_par[n]],
                data=corrected_object_dat[n], parms=corrected_object_par[n],
                indent_level=indent_level, base_URI=base_URI)
            if debug:
                print(' DATA: ' + corstat_arr[n] +
                      ' PARAM: ' + corpar_arr[n], end='\n')

        if (corstat_arr == ['GREEN'] * len(corrected_links)) and \
           (corpar_arr == ['GREEN'] * len(corrected_links)):
            print(indent_level +
                  '>> Correction for QSB_Bolometry succeeded ...')

        elif (corstat_arr == ['FAIL'] * len(corrected_links)) and \
             (corpar_arr == ['FAIL'] * len(corrected_links)):
            print(indent_level +
                  '>> Correction at QSB_Bolometry all failed ...')

        else:
            print(indent_level + '>> Failed to upload:', end=' ')
            for n in range(0, len(corrected_links)):
                if (corstat_arr[n] is not 'GREEN'):
                    print(corrected_links[n] + '_DATASTREAM', end=' ')
                if (corpar_arr[n] is not 'GREEN'):
                    print(corrected_links[n] + '_PARLOG', end=' ')

    return versionize


def set_up_data(
        power={'none': None},
        dat={'none': None},
        prio={'none': None},
        raw_time=np.linspace(0, 1, 10000)):
    """ Sets up the data and parameter objects to be uploaded later on
        with the single one archive access function.
    Args:
        radpower_object (0, list): radiated power and the alike etc.
        data_object (1, list): Downloaded objects from the archive.
        priority_object (2, list): Predefined and constant values
        raw_time (3, list): Time vector
    Returns:
        upload_object_dat (0, list): Calculated/created results to upload
        upload_object_par (1, list): Fitting parameter blocks to the data
        corrected_object_dat (2, list): Data of the original repo corrected
        corrected_object_par (3, list): Parameter blocks to corrections.
    Notes:
        None
    """
    activity = [1] * 128

    # PradHBCm
    prad_hbc = \
        {"label": "PradHBCm", "unit": "W",
         "values": power['P_rad_hbc'],
         "dimensions": raw_time}
    param_hbc = \
        {"label": "parms",
         "values": [{"chanDescs":
                     {"[0]": {"name": "PradHBCm",
                              "physicalQuantity":
                              {"type": "W"}}}}],
         "VolumeUse": prio['geometry']['geometry']['vbolo'],
         "KBoloUse": prio['geometry']['geometry']['kbolo'],
         "VolumeSumHBCm": power['volume_sum_hbc'],
         "dimensions": raw_time}

    # PradVBC
    prad_vbc = \
        {"label": "PradVBC", "unit": "W",
         "values": power['P_rad_vbc'],
         "dimensions": raw_time}
    param_vbc = \
        {"label": "parms",
         "values": [{"chanDescs":
                     {"[0]": {"name": "PradVBC",
                              "physicalQuantity":
                              {"type": "W"}}}}],
         "VolumeUse": prio['geometry']['geometry']['vbolo'],
         "KBoloUse": prio['geometry']['geometry']['kbolo'],
         "VolumeSumVBC": power['volume_sum_vbc'],
         "dimensions": raw_time}

    # PradChannels
    pradch = {"datatype": "float", "unit": "W",
              "values": power['power'],
              "dimensions": raw_time}
    param_chan = {"label": "parms",
                  "values": [{"chanDescs": {}}],
                  "NChannels": 128,
                  "dimensions": raw_time}
    for c in range(0, 128):
        param_chan["values"][0]["chanDescs"]["[" + str(c) + "]"] = \
            {"name": "PradChannels_" + str(c),
             "description": 'individual channel power',
             "detector": "Au_Kapton",
             "active": activity[c],
             "OffsetSlope": power['a'][c],
             "OffsetConstant": power['b'][c],
             "ROhm": power['rohm'][c],
             "KappaM": power['kappam'][c],
             "TauM": power['taum'][c],
             "physicalQuantity":
                 {"type": "W",
                  "from": min(power['power'][c]),
                  "upto": max(power['power'][c])}}

    # BoloAdjusted
    adjust = {"datatype": "float", "unit": "V",
              "values": power['voltage adjusted'],
              "dimensions": raw_time}
    param_adjust = {"label": "parms",
                    "values": [{"chanDescs": {}}],
                    "NChannels": 128,
                    "dimensions": raw_time}
    for c in range(0, 128):
        param_adjust["values"][0]["chanDescs"]["[" + str(c) + "]"] = \
            {"name": "BoloAdjusted_" + str(c),
             "description": 'adjusted voltage per channel',
             "detector": "Au_Kapton",
             "active": activity[c],
             "OffsetSlope": power['a'][c],
             "OffsetConstant": power['b'][c],
             "physicalQuantity":
                 {"type": "V",
                  "from": min(power['voltage adjusted'][c]),
                  "upto": max(power['voltage adjusted'][c])}}

    # LInAlch
    arrAlch = []
    for i, ch in enumerate(prio['geometry']['channels']['eChannels']['ALF']):
        arrAlch.append(power['voltage adjusted'][ch].tolist())
    lin_Alch = {"datatype": "float", "unit": "V",
                "values": arrAlch,
                "dimensions": raw_time}
    param_Alch = {"label": "parms",
                  "values": [{"chanDescs": {}}],
                  "NChannels": len(prio['geometry']['channels'][
                      'eChannels']['ALF']),
                  "dimensions": raw_time}
    for i, ch in enumerate(prio['geometry']['channels']['eChannels']['ALF']):
        param_Alch["values"][0][
            "chanDescs"]["[" + str(i) + "]"] = \
            {"name": "LInAlch_" + str(prio['geometry']['channels'][
                'gChannels']['ALF'][i]),
             "description": 'aluminium coated detector',
             "detector": "Au_Kapton with Al layer",
             "active": activity[ch],
             "OffsetSlope": power['a'][ch],
             "OffsetConstant": power['b'][ch],
             "physicalQuantity":
                 {"type": "V",
                  "from": min(power['voltage adjusted'][ch]),
                  "upto": max(power['voltage adjusted'][ch])}}

    # LInSXRh
    arrSXRh = []
    for i, ch in enumerate(prio['geometry']['channels']['eChannels']['SXRh']):
        arrSXRh.append(power['voltage adjusted'][ch].tolist())
    lin_SXRh = {"datatype": "float", "unit": "V",
                "values": arrSXRh,
                "dimensions": raw_time}
    param_SXRh = {"label": "parms",
                  "values": [{"chanDescs": {}}],
                  "NChannels": len(prio['geometry']['channels'][
                      'eChannels']['SXRh']),
                  "dimensions": raw_time}
    for i, ch in enumerate(prio['geometry']['channels']['eChannels']['SXRh']):
        param_SXRh["values"][0][
            "chanDescs"]["[" + str(i) + "]"] = \
            {"name": "LInSXRh_" + str(prio['geometry']['channels'][
                'eChannels']['SXRh'][i]),
             "description": 'SXR detector horizontally',
             "detector": "Au_Kapton for SXR",
             "active": activity[ch],
             "OffsetSlope": power['a'][ch],
             "OffsetConstant": power['b'][ch],
             "physicalQuantity":
                 {"type": "V",
                  "from": min(power['voltage adjusted'][ch]),
                  "upto": max(power['voltage adjusted'][ch])}}

    # LInSXRv
    arrSXRv = []
    for i, ch in enumerate(prio['geometry']['channels']['eChannels']['SXRv']):
        arrSXRv.append(power['voltage adjusted'][ch].tolist())
    lin_SXRv = {"datatype": "float", "unit": "V",
                "values": arrSXRv,
                "dimensions": raw_time}
    param_SXRv = {"label": "parms",
                  "values": [{"chanDescs": {}}],
                  "NChannels": len(prio['geometry'][
                      'channels']['eChannels']['SXRv']),
                  "dimensions": raw_time}
    for i, ch in enumerate(prio['geometry'][
                           'channels']['eChannels']['SXRh']):
        param_SXRv["values"][0]["chanDescs"]["[" + str(i) + "]"] = \
            {"name": "LInSXRv_" + str(prio['geometry'][
                'channels']['gChannels']['SXRh'][i]),
             "description": 'SXR detector vertically',
             "detector": "Au_Kapton for SXR",
             "active": activity[ch],
             "OffsetSlope": power['a'][ch],
             "OffsetConstant": power['b'][ch],
             "physicalQuantity":
                 {"type": "V",
                  "from": min(power['voltage adjusted'][ch]),
                  "upto": max(power['voltage adjusted'][ch])}}

    # LInVBCl
    arrVBCl = []
    for i, ch in enumerate(prio['geometry']['channels']['eChannels']['VBCl']):
        arrVBCl.append(power['voltage adjusted'][ch].tolist())
    lin_VBCl = {"datatype": "float", "unit": "V",
                "values": arrVBCl,
                "dimensions": raw_time}
    param_VBCl = {"label": "parms",
                  "values": [{"chanDescs": {}}],
                  "NChannels": len(prio['geometry']['channels'][
                      'eChannels']['VBCl']),
                  "dimensions": raw_time}
    for i, ch in enumerate(prio['geometry']['channels']['eChannels']['VBCl']):
        param_VBCl["values"][0][
            "chanDescs"]["[" + str(i) + "]"] = \
            {"name": "LInVBCl_" + str(prio['geometry'][
                'channels']['gChannels']['VBCl'][i]),
             "description": 'Vertical/left detector camera',
             "detector": "Au_Kapton",
             "active": activity[c],
             "OffsetSlope": power['a'][ch],
             "OffsetConstant": power['b'][ch],
             "physicalQuantity":
                 {"type": "V",
                  "from": min(power['voltage adjusted'][ch]),
                  "upto": max(power['voltage adjusted'][ch])}}

    # LInVBCr
    arrVBCr = []
    for i, ch in enumerate(prio['geometry']['channels']['eChannels']['VBCr']):
        arrVBCr.append(power['voltage adjusted'][ch].tolist())
    lin_VBCr = {"datatype": "float", "unit": "V",
                "values": arrVBCr,
                "dimensions": raw_time}
    param_VBCr = {"label": "parms",
                  "values": [{"chanDescs": {}}],
                  "NChannels": len(prio['geometry'][
                      'channels']['eChannels']['VBCr']),
                  "dimensions": raw_time}
    for i, ch in enumerate(prio['geometry']['channels']['eChannels']['VBCr']):
        param_VBCr["values"][0][
            "chanDescs"]["[" + str(i) + "]"] = \
            {"name": "LInVBCr_" + str(prio['geometry']['channels'][
                'gChannels']['VBCr'][i]),
             "description": 'Vertical/right detector camera',
             "detector": "Au_Kapton",
             "active": activity[ch],
             "OffsetSlope": power['a'][ch],
             "OffsetConstant": power['b'][ch],
             "physicalQuantity":
                 {"type": "V",
                  "from": min(power['voltage adjusted'][ch]),
                  "upto": max(power['voltage adjusted'][ch])}}

    # LInHBCm
    arrHBCm = []
    for i, h in enumerate(prio['geometry']['channels']['eChannels']['HBCm']):
        arrHBCm.append(power['voltage adjusted'][ch].tolist())
    lin_HBCm = {"datatype": "float", "unit": "V",
                "values": arrHBCm,
                "dimensions": raw_time}
    param_HBCm = {"label": "parms",
                  "values": [{"chanDescs": {}}],
                  "NChannels": len(prio['geometry']['channels'][
                      'eChannels']['HBCm']),
                  "dimensions": raw_time}
    for i, ch in enumerate(prio['geometry']['channels']['eChannels']['HBCm']):
        param_HBCm["values"][0][
            "chanDescs"]["[" + str(i) + "]"] = \
            {"name": "LInHBCm_" + str(prio['geometry'][
                'channels']['gChannels']['HBCm'][i]),
             "description": 'Horizontal detector camera',
             "detector": "Au_Kapton",
             "active": activity[ch],
             "OffsetSlope": power['a'][ch],
             "OffsetConstant": power['b'][ch],
             "physicalQuantity":
                 {"type": "V",
                  "from": min(power['voltage adjusted'][ch]),
                  "upto": max(power['voltage adjusted'][ch])}}

    # MKappa
    activity = [1] * 128
    MKappa = {"datatype": "float",
              "values": [power['kappam'].tolist(), [0] * 128],
              "dimensions": [raw_time[0], raw_time[-1]]}
    param_MKappa = {"label": "parms", "unit": "A^2",
                    "values": [{"chanDescs": {}}],
                    "NChannels": 128,
                    "dimensions": [raw_time[0], raw_time[-1]]}
    for c in range(0, 128):
        param_MKappa["values"][0]["chanDescs"]["[" + str(c) + "]"] = \
            {"name": "MKappa_" + str(c),
             "description": 'effective detector heat capacity',
             "detector": "Au_Kapton",
             "active": activity[c],
             "fitTimeConstant": power['fit_results'][0][c][1],
             "fitLeadingFactor": power['fit_results'][0][c][0],
             "physicalQuantity":
                 {"type": "A^",
                  "from": min(power['kappam']),
                  "upto": max(power['kappam'])}}

    # MTau
    activity = [1] * 128
    MTau = {"datatype": "float", "unit": "s",
            "values": [power['taum'].tolist(), [0] * 128],
            "dimensions": [raw_time[0], raw_time[-1]]}
    param_MTau = {"label": "parms",
                  "values": [{"chanDescs": {}}],
                  "NChannels": 128,
                  "dimensions": [raw_time[0], raw_time[-1]]}
    for c in range(0, 128):
        param_MTau["values"][0]["chanDescs"]["[" + str(c) + "]"] = \
            {"name": "MTau_" + str(c),
             "description": 'detector cooling time',
             "detector": "Au_Kapton",
             "active": activity[c],
             "fitTimeConstant": power['fit_results'][0][c][1],
             "fitLeadingFactor": power['fit_results'][0][c][0],
             "physicalQuantity":
                 {"type": "s",
                  "from": min(power['taum']),
                  "upto": max(power['taum'])}}

    # MRes
    activity = [1] * 128
    MRes = {"datatype": "float", "unit": "Ohm",
            "values": [power['rohm'].tolist(), [0] * 128],
            "dimensions": [raw_time[0], raw_time[-1]]}
    param_MRes = {"label": "parms",
                  "values": [{"chanDescs": {}}],
                  "NChannels": 128,
                  "dimensions": [raw_time[0], raw_time[-1]]}
    for c in range(0, 128):
        param_MRes["values"][0]["chanDescs"]["[" + str(c) + "]"] = \
            {"name": "MRes_" + str(c),
             "description": 'detector electrical resistance',
             "detector": "Au_Kapton",
             "active": activity[c],
             "fitTimeConstant": power['fit_results'][0][c][1],
             "fitLeadingFactor": power['fit_results'][0][c][0],
             "physicalQuantity":
                 {"type": "Ohm",
                  "from": min(power['rohm']),
                  "upto": max(power['rohm'])}}

    # RKappa
    activity = [1] * 128
    RKappa = {"datatype": "float", "unit": "A^2",
              "values": [power['kappar'].tolist(), [0] * 128],
              "dimensions": [raw_time[0], raw_time[-1]]}
    param_RKappa = {"label": "parms",
                    "values": [{"chanDescs": {}}],
                    "NChannels": 128,
                    "dimensions": [raw_time[0], raw_time[-1]]}
    for c in range(0, 128):
        param_RKappa["values"][0]["chanDescs"]["[" + str(c) + "]"] = \
            {"name": "RKappa_" + str(c),
             "description": 'reference effective detector heat capacity',
             "detector": "Au_Kapton",
             "active": activity[c],
             "fitTimeConstant": power['fit_results'][1][c][1],
             "fitLeadingFactor": power['fit_results'][1][c][0],
             "physicalQuantity":
                 {"type": "A^2",
                  "from": min(power['kappar']),
                  "upto": max(power['kappar'])}}

    # RTau
    activity = [1] * 128
    RTau = {"datatype": "float", "unit": "s",
            "values": [power['taur'].tolist(), [0] * 128],
            "dimensions": [raw_time[0], raw_time[-1]]}
    param_RTau = {"label": "parms",
                  "values": [{"chanDescs": {}}],
                  "NChannels": 128,
                  "dimensions": [raw_time[0], raw_time[-1]]}
    for c in range(0, 128):
        param_RTau["values"][0]["chanDescs"]["[" + str(c) + "]"] = \
            {"name": "RTau_" + str(c),
             "description": 'reference detector cooling time',
             "detector": "Au_Kapton",
             "active": activity[c],
             "fitTimeConstant": power['fit_results'][1][c][1],
             "fitLeadingFactor": power['fit_results'][1][c][0],
             "physicalQuantity":
                 {"type": "s^-1",
                  "from": min(power['taur']),
                  "upto": max(power['taur'])}}

    # RRes
    activity = [1] * 128
    RRes = {"datatype": "float", "unit": "Ohm",
            "values": [power['rohr'].tolist(), [0] * 128],
            "dimensions": [raw_time[0], raw_time[-1]]}
    param_RRes = {"label": "parms",
                  "values": [{"chanDescs": {}}],
                  "NChannels": 128,
                  "dimensions": [raw_time[0], raw_time[-1]]}
    for c in range(0, 128):
        param_RRes["values"][0]["chanDescs"]["[" + str(c) + "]"] = \
            {"name": "RRes_" + str(c),
             "description": 'detector electrical resistance',
             "detector": "Au_Kapton",
             "active": activity[c],
             "fitTimeConstant": power['fit_results'][1][c][1],
             "fitLeadingFactor": power['fit_results'][1][c][0],
             "physicalQuantity":
                 {"type": "Ohm",
                  "from": min(power['rohr']),
                  "upto": max(power['rohr'])}}

    # ChordalProfile_HBCm
    chordalprofile_hbc = np.zeros((
        len(raw_time),
        len(prio['geometry']['channels']['eChannels']['HBCm'])))
    for i, ch in enumerate(prio['geometry'][
            'channels']['eChannels']['HBCm']):
        chordalprofile_hbc[:, ch] = power['volscaled'][ch, :]

    chordal_hbc = \
        {"label": "ChordalProfile_HBCm",
         "datatype": 'Float',
         'sampleCount': len(raw_time),
         "values": chordalprofile_hbc,
         "unit": "W*m^-3",
         "dimensions": raw_time}
    param_chordal_hbc = \
        {"label": "parms",
         "values": [{"chanDescs":
                     {"[0]": {"name": "ChordalProfile_HBCm",
                              "physicalQuantity":
                              {"type": "W*m^-3"}}}}],
         "VolumeUse": prio['geometry']['geometry']['vbolo'],
         "KBoloUse": prio['geometry']['geometry']['kbolo'],
         "dimensions": [raw_time[0], raw_time[-1]]}

    # ChordalProfile_VBC
    chordalprofile_vbc = np.zeros((
        len(raw_time),
        len(prio['geometry']['channels']['eChannels']['VBC'])))
    for i, ch in enumerate(prio['geometry']['channels']['eChannels']['VBC']):
        chordalprofile_vbc[:, i] = power['volscaled'][ch, :]
    chordal_vbc = \
        {"label": "ChordalProfile_VBC",
         "datatype": 'Float',
         'sampleCount': len(raw_time),
         "values": chordalprofile_vbc,
         "unit": "W*m^-3",
         "dimensions": raw_time}
    param_chordal_vbc = \
        {"label": "parms",
         "values": [{"chanDescs":
                     {"[0]": {"name": "ChordalProfile_VBC",
                              "physicalQuantity":
                              {"type": "W*m^-3"}}}}],
         "VolumeUse": prio['geometry']['geometry']['vbolo'],
         "KBoloUse": prio['geometry']['geometry']['vbolo'],
         "dimensions": [raw_time[0], raw_time[-1]]}

    upload_object_dat = [prad_hbc,
                         prad_vbc,
                         pradch,
                         adjust,
                         lin_Alch,
                         lin_SXRh,
                         lin_SXRv,
                         lin_VBCl,
                         lin_VBCr,
                         lin_HBCm,
                         MKappa,
                         MTau,
                         MRes,
                         RKappa,
                         RTau,
                         RRes,
                         chordal_hbc,
                         chordal_vbc]
    upload_object_par = [param_hbc,
                         param_vbc,
                         param_chan,
                         param_adjust,
                         param_Alch,
                         param_SXRh,
                         param_SXRv,
                         param_VBCl,
                         param_VBCr,
                         param_HBCm,
                         param_MKappa,
                         param_MTau,
                         param_MRes,
                         param_RKappa,
                         param_RTau,
                         param_RRes,
                         param_chordal_hbc,
                         param_chordal_vbc]

    new_raw_time = np.linspace(
        raw_time[0], raw_time[0] +
        (len(dat['BoloCalibMeasFoilCurrent']['values'][0]) * 400000),
        len(dat['BoloCalibMeasFoilCurrent']['values'][0]))

    # BoloCalibMeasFoilCurrent
    activity = [1] * 128
    MeasCurr = {"datatype": "float", "unit": "A",
                "values": dat['BoloCalibMeasFoilCurrent']['values'],
                "dimensions": new_raw_time}
    param_MeasCurr = {"label": "parms",
                      "values": [{"chanDescs": {}}],
                      "NChannels": 128,
                      "dimensions": new_raw_time}
    for c in range(0, 128):
        param_MeasCurr["values"][0]["chanDescs"]["[" + str(c) + "]"] = \
            {"name": "E_Chan_" + str(c),
             "description": 'foil current for calibration',
             "detector": "Au_Kapton",
             "active": activity[c],
             "physicalQuantity":
                 {"type": "A",
                  "from": min(
                      dat['BoloCalibMeasFoilCurrent']['values'][c]),
                  "upto": max(
                      dat['BoloCalibMeasFoilCurrent']['values'][c])}}

    # BoloCalibMeasRefCurrent
    activity = [1] * 128
    RefCurr = {"datatype": "float", "unit": "A",
               "values": dat['BoloCalibMeasFoilCurrent']['values'],
               "dimensions": new_raw_time}
    param_RefCurr = {"label": "parms",
                     "values": [{"chanDescs": {}}],
                     "NChannels": 128,
                     "dimensions": new_raw_time}
    for c in range(0, 128):
        param_RefCurr["values"][0]["chanDescs"]["[" + str(c) + "]"] = \
            {"name": "E_Chan_" + str(c),
             "description": 'reference foil current for calibration',
             "detector": "Au_Kapton",
             "active": activity[c],
             "physicalQuantity":
                 {"type": "A",
                  "from": min(
                      dat['BoloCalibMeasFoilCurrent']['values'][c]),
                  "upto": max(
                      dat['BoloCalibMeasFoilCurrent']['values'][c])}}

    corrected_object_dat = [MeasCurr,
                            RefCurr]
    corrected_object_par = [param_MeasCurr,
                            param_RefCurr]

    return [upload_object_dat, upload_object_par,
            corrected_object_dat, corrected_object_par]


def temperatures(
        upload=False,
        debug=False):
    """ upload temperature time lines
    Args:
        upload (bool, optional): upload the dicts
        debug (bool, optional): print debugging
    Returns:
        None
    """
    files = [
        '2018-06-11_18.txt', '2018-06-19_25.txt', '2018-08-01_08.txt',
        '2018-08-08_15.txt', '2018-08-16_23.txt', '2018-08-24_09-03.txt',
        '2018-09-04_05.txt', '2018-09-18_28.txt', '2018-10-01_02.txt',
        '2018-10-02_02.txt', '2018-10-02_11.txt', '2018-10-11_16.txt',
        '2018-10-16_18.txt']
    [t, ch1T, ch2T, ch3T, ch4T] = [[], [], [], [], []]

    i = 0
    for f in files:
        print(str(i) + '/' + str(len(files)), f)
        i += 1

        file = r'../files/temperature/export logging/' + f
        channels_T = np.genfromtxt(
            file, delimiter='\t', dtype=np.float, usecols=[2, 3, 4, 5])
        tmp = np.genfromtxt(
            file, delimiter='\t', dtype=str, usecols=[0, 1])

        x = [ttt.mktime(datetime.datetime.strptime(
             tmp[i, 0] + ' ' + tmp[i, 1],
             '%m/%d/%Y %H:%M:%S').timetuple()) * 1e9
             for i, s in enumerate(tmp)]

        t.extend(x)
        ch1T.extend(channels_T[:, 0].tolist())
        ch2T.extend(channels_T[:, 1].tolist())
        ch3T.extend(channels_T[:, 1].tolist())
        ch4T.extend(channels_T[:, 1].tolist())

    # all together
    test = np.vstack((t, ch1T, ch2T, ch3T, ch4T))

    N = ['1', '2', '3', '4']
    vers.increase_versioning(
        reason='transported temperature lines from *.plw to' +
        ' *.csv and uploaded them',
        links=['TemperatureChannel' + n for n in N],
        specifier='QSB_Bolometry/')

    if upload:
        i = 0
        for n in N:
            dat = \
                {"label": "TemperatureChannel" + n,
                 "unit": "°C",
                 "values": test[i + 1].tolist(),
                 "dimensions": test[0].tolist()}
            param = \
                {"label": "parms",
                 "values":
                 [{"chanDescs":
                   {"[" + n + "]": {"name": "Channel " + n,
                                    "physicalQuantity": {"type": "°C"}}}}],
                 "dimensions": test[0].tolist()}

            L = ['TemperatureChannel' + n + '_DATASTREAM/',
                 'TemperatureChannel' + n + '_PARLOG/']
            foo = multich.upload_archive(
                links=['QSB_Bolometry/' + l for l in L],
                data=dat, parms=param)
            if debug:
                print(foo)
            i += 1

    return
