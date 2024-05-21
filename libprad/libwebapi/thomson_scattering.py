""" *************************************************************************
    so HEADER
    Created on Mon Feb 18 17:01:00 2019
    Author: Hannes Damm
        git source:
            https://git.ipp-hgw.mpg.de/hdamm/ts_archive_2_profiles

    used and edited at own risk, not sanctioned by orignal creator
    Editor: Philipp Hacker (Mi Sep 25 11:03:00 2019) """

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from osa import Client
import re
import json
import os
import sys
import archivedb  # https://gitlab.mpcdf.mpg.de/kjbrunne/W7X_APIpy
import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# The script downloads Thomson data from the archive and stores it
# to hard drive. Important TS quantities like the calibration version
# and laser identifier are given as well as the vmec id applied for the
# mapping. The TS locations (volumina) are given in cartesian, cylindrical
# and vmec coordinates. LFS radii and s are multiplied by -1 to gain full
# profiles. The physics quantities T_e, n_e and p_e + errors are given. W_e
# is calculated by the script. A cross-calibration factor to the
# interferometry density signal is calculated with and without mapping. It
# can be used for scaling of n_e, p_e and W_e and is NOT automatically
# applied. Plots,  fitting methods, and other things can be adjusted by the
# user. The  minimum input required is the shot number. Examples how to
# read in the stored data file are given in the code.

# required packages:
#     sklearn >= v0.20.0
#     sys
#     urllib or urllib2
#     numpy >= 1.14.3
#     scipy >= 1.1.0
#     osa >= 1.6-p4
#     re >= 2.2.1
#     json >= 2.0.9
#     archivedb >= 0.7.6
# recommended packages
#     matplotlib >= 1.5.1

__version__ = u"0.2.1"
__status__ = "beta"
__author__ = u"W7-X thomson scattering group and friends"
__copyright__ = "FOR IPP INTERNAL USE + COLLABORATORS ONLY"
__credits__ = [
    "Daniel Boeckenhoff",
    "Sergey Bozhenkov",
    "Kai Jakob Brunner",
    "Hannes Damm",
    "Golo Fuchert",
    "Daniele Villa"]
__maintainer__ = "Golo Fuchert"
__email__ = "golo.fuchert@ipp.mpg.de"

global_debug = False  # can be changed via kwarg

# import version specific python modules:
python_version = sys.version_info.major + 0.1 * sys.version_info.minor
if python_version >= 2.7 and python_version <= 3.0:
    import urllib2
elif python_version >= 3.0:
    import urllib.request as urllib2
else:
    print('python version %s is not supported (e.g. <2.7)' % python_version)

""" eo header
************************************************************************** """


def ts_data_full_analysis(
        shotno='20181018.041',
        t_start_analysis=0,
        t_end_analysis=0,
        TS_version='',
        vmec_id='',
        calc_w_e=False,
        burst=False,
        sanity_checks=False,
        scaling='gauss',
        debug=True,
        indent_level='\t'):
    """ fully returns TS data and saves shit
    Args:
        shotno (str, optional): XPID
        t_start_analysis (int, optional): Analysis start time
        t_end_analysis (int, optional): Analysis end time
        TS_version (str, optional): data version loaded
        vmec_id (str, optional): VMEC id from archive
        calc_w_e (bool, optional): If W_e should be calculated
        burst (bool, optional): Burst mode on
        sanity_checks (bool, optional): idk
        scaling (str, optional): Way of scale fitting etcc
        debug (bool, optional): Debugging print
        indent_level (str, optional): Printing indentation
    Returns:
        data (dict): TS data
        names (list): names of TS data entries in dict
        dtypes (list): data types to be interpreted
        units (list): physical units of data types in data
        info (dict): calculated stuff used in routine
    Raises:
        ValueError: Shotno error
    Notes:
        None
    """
    if debug:
        global global_debug
        global_debug = True

    # print analysed shotno
    # TODO sanity_checks variable maybe retrieved from logging level?
    if not isinstance(shotno, str):
        raise ValueError("Shotno should be of type str.")
    print(indent_level + "\tAnalysing: ", shotno)  # TODO logger!

    # Automatically setup urls and variables referenced throughout script
    shot_year, shot_day, TS_version_address, vmec_client, vmec_base = \
        script_variables(shotno)

    loc = '../results/THOMSON/' + str(shot_day) + '/' + shotno[-3:] + '/'
    if not os.path.exists(loc):
        os.makedirs(loc)

    # t1 = start of heating, t5 = termination of heating + roughly 4s
    # if not provided by user assumed to be beginning and end of whole shot
    if t_start_analysis == 0:
        t1 = int(archivedb.get_program_from_to(shotno, useCache=True)[0])

    elif t_start_analysis != 0:
        t1 = int(time_to_timestamp(t_start_analysis, shotno=shotno))

    if t_end_analysis == 0:
        t5 = int(archivedb.get_program_from_to(shotno, useCache=True)[1])

    elif t_end_analysis != 0:
        t5 = int(time_to_timestamp(t_end_analysis, shotno=shotno,))

    # setup equilibrium geometry
    if vmec_id:
        vmec_ids = (np.array([t1, t5]), np.array([vmec_id, vmec_id]))
    else:
        vmec_ids = read_vmec_ids(t1, t5)

    if not TS_version:  # set TS version if not provided by user
        TS_version = recommended_ts_version(
            shot_year, shot_day, TS_version_address, t1, t5)

    # basic volume information
    volumes, t, params = read_active_scatter_vol_params(
        shotno, shot_year, TS_version, t1, t5)

    if global_debug:
        print(indent_level + "\tread params and activity checked")

    # setup the geometry from the basic volume information
    scatter_vol_coordinates = coordinate_transformation(
        vmec_ids, volumes, params, vmec_client, shot_year)

    if global_debug:
        print(indent_level + '\tcoordinate transformation')

    # collects and orders all the data relevant to the volumes
    if global_debug:
        print(indent_level + '\tbuilding volume data')

    data, laserno_version = build_volume_data(
        shotno, shot_year, vmec_ids, t, t1, t5, volumes,
        scatter_vol_coordinates, TS_version, scaling)

    if global_debug:
        print(indent_level + '\tbuild volume data & downloaded TS data')

    if scaling != 'unscaled':
        if global_debug:
            print(indent_level + '\tscaling TS data: ', end='')
        scale_ts_data(
            t, t1, t5, data, vmec_ids, shot_year, vmec_client,
            vmec_base, volumes, burst, sanity_checks, scaling, debug=debug)

        if global_debug:
            print('done!', end='\n')

    if calc_w_e:
        calculate_w_e(t, data, volumes)

    if global_debug:
        if scaling == 'unscaled':
            print(indent_level + '\t\tThomson Data is unscaled. ' +
                  'No scaling factors were calculated.')
        else:
            print(
                '\n' + indent_level + '\t\tTS data is stored along with the' +
                'scaled data according to the real space factor\n' +
                indent_level + '\t\tcalculated with {scaling} method. The ' +
                'error bars on density and pressure are available\n' +
                indent_level + '\t\tadjusted with error propagation. W_e ' +
                'has been calculated on the scaled quantities.\n')

    # remove 'r_real' from stored data
    quantities_to_store = list(data.dtype.names)
    # quantities_to_remove = ['r_real']
    # for quantity_to_remove in quantities_to_remove:
    #    quantities_to_store.remove(quantity_to_remove)
    tuple(quantities_to_store)

    data_unit_list = [
        't [ns];',
        'shot_time [s];',
        'shotno;',
        'laserno;',
        'TS_version;',
        'vmec_id;',
        'factor_real_{scaling};',
        'factor_mapped_{scaling};',
        'scattering_volume_number;',
        'x_real [m];',
        'y_real [m];',
        'z_real [m];',
        # 'r_real [m];',
        # will become LOS coordinate at some point,
        # specific for the individual 3 lasers!
        'r_eff [m];',
        'r_cyl [m];',
        'phi_cyl [deg];',
        's_vmec;',
        'theta_vmec [deg];',
        'phi_vmec [deg];',
        'Te_map [keV];',
        'Te_low97.5 [keV];',
        'Te_high97.5 [keV];',
        'ne_map [e19m^-3];',
        'ne_low [e19m^-3];',
        'ne_high [e19m^-3];',
        'pe_map [kPa];',
        'pe_low [kPa];',
        'pe_high [kPa];',
        'ne_map_scaled_{scaling} [e19m^-3];',
        'ne_low_scaled_{scaling} [e19m^-3];',
        'ne_high_scaled_{scaling} [e19m^-3];',
        'pe_map_scaled_{scaling} [kPa];',
        'pe_low_scaled_{scaling} [kPa];',
        'pe_high_scaled_{scaling} [kPa];',
        'We_map_scaled_{scaling} [kJ];',
        'We_low_scaled_{scaling} [kJ];',
        'We_high_scaled_{scaling} [kJ]']

    data_unit_list_string = ''.join(data_unit_list).format(**locals())

    names = [
        data_unit_list_string.replace(' ', ';').split(';')[i]
        for i in range(len(data_unit_list_string.replace(
            ' ', ';').split(';'))) if '[' not in
        data_unit_list_string.replace(' ', ';').split(';')[i]]

    units = [
        data_unit_list_string.split(';')[i].replace(
            names[i], '').replace(' ', '').replace(
            '[', '').replace(']', '') for i in range(len(names))]

    dtypes = []
    for n in data[quantities_to_store].dtype.names:
        dtypes = dtypes + [str(data.dtype[n])]

    header = '\n'.join([
        ';'.join(names), ';'.join(units),
        ';'.join(dtypes)]).format(**locals())

    data_filename = loc + \
        'thomson_data_{shotno}_{scaling}_{TS_version}.txt'.format(**locals())
    np.savetxt(
        data_filename, data[quantities_to_store],
        fmt='%s', delimiter=';', header=header)

    # save text file containing version and datastream
    # information of the analysed data for publication reference
    info = np.array([
        ('TS_data',
         'Test/raw/W7X/QTB_Profile/volume_**vol_no**_DATASTREAM',
         '{TS_version}'.format(**locals())),
        ('reference_equilibria',
         'ArchiveDB/raw/W7XAnalysis/Equilibrium/RefEq_PARLOG/V1/' +
         'parms/equilibriumID',
         '{}'.format(archivedb.get_last_version(
             'ArchiveDB/raw/W7XAnalysis/Equilibrium/RefEq_PARLOG/' +
             'V1/parms/equilibriumID', t1, t5, useCache=True))),
        ('TS_active_flag_2017',
         'Test/raw/W7X/QTB_Test_V/volume_**vol_no**_PARLOG/V1/' +
         'parms/active',
         '{}'.format(archivedb.get_last_version(
             'Test/raw/W7X/QTB_Test_V/volume_2_PARLOG/V1/' +
             'parms/active', t1, t5, useCache=True))),
        ('TS_active_flag_2018',
         'Test/raw/W7X/QTB_Profile/volume_**vol_no**_PARLOG/V1/' +
         'parms/active',
         '{TS_version}'.format(**locals())),
        ('TS_scatter_vol_params',
         'Test/raw/W7X/QTB_Profile/volume_**vol_no**_PARLOG',
         '{TS_version}'.format(**locals())),
        ('vmec_client',
         '{}'.format(str(vmec_client).split('\t')[1]), 'None'),
        ('vmec_base', '{vmec_base}'.format(**locals()), 'None'),
        ('TS_laserno',
         'Test/raw/W7X/QTB_Profile_LaserStream/' +
         'laser_number_DATASTREAM',
         '{laserno_version}'.format(**locals())),
        ('high_res_interferometry (burst mode TS)',
         'Test/raw/W7XAnalysis/QMJ_IEDDI/IED_DDR/V1/1/density',
         '{}'.format(archivedb.get_last_version(
             'Test/raw/W7XAnalysis/QMJ_IEDDI/IED_DDR/V1/1/density',
             t1, t5, useCache=True))),
        ('low_res_interferometry',
         'ArchiveDB/codac/W7X/CoDaStationDesc.16339/' +
         'DataModuleDesc.16341_DATASTREAM/' +
         '0/Line integrated density/scaled',
         '{}'.format(archivedb.get_last_version(
             'ArchiveDB/codac/W7X/CoDaStationDesc.16339/' +
             'DataModuleDesc.16341_DATASTREAM/0/' +
             'Line integrated density/scaled', t1, t5, useCache=True))), ],
        dtype=[('data_name', 'O'),
               ('data_stream', 'O'),
               ('version', 'O'), ])

    info_header = []
    for i in range(len(info.dtype.names)):
        info_header = info_header + [info.dtype.names[i]]
    info_header = ';'.join(info_header).format(**locals())

    dtime = datetime.datetime.now().strftime("%Y_%m_%d_%H.%M")
    np.savetxt(
        '../results/THOMSON/' + str(shot_day) + '/' + shotno[-3:] + '/' +
        'ts_data_log_{shotno}_{scaling}_{dtime}.txt'.format(**locals()), info,
        fmt='%s', delimiter=';', header=info_header)

    return (data, names, dtypes, units, info)


def script_variables(
        shotno='20181018.041'):
    """
    Automatically sets up a set of variables
    referenced by the main script
    """
    shot_year = int(shotno[:4])
    shot_day = int(shotno[:8])

    TS_version_adress = "Test/raw/W7X/QTB_Profile/volume_2_DATASTREAM"
    vmec_client = Client('http://esb:8280/services/vmec_v8?wsdl')
    vmec_base = 'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/'

    return shot_year, shot_day, TS_version_adress, vmec_client, vmec_base


def recommended_ts_version(
        shot_year=2018,
        shot_day=20181018,
        TS_version_address='',
        t_start=0,
        t_end=0):
    """ version of the TS data needs to be set manually until
        there is a recommended
        one for all discharges. If you are unsure about
        your specific user-case please contact the TS group!!!
    Args:
        shot_year (int, optional): Year
        shot_day (int, optional): Month + Day
        TS_version_address (str, optional): Version link
        t_start (int, optional): Start
        t_end (int, optional): End
    Returns:
        TS_version (str): Recommended (not last) TS version
    """
    # use of last available version is NOT recommended

    if shot_year == 2017:
        TS_version = "V3"
    elif shot_day > 20180727 and shot_day < 20180922:
        TS_version = "V8"
    elif shot_day > 20180922 and shot_day < 20181018:
        TS_version = "V10"
    else:
        TS_version = archivedb.get_last_version(
            TS_version_address, t_start, t_end, useCache=True)

    return TS_version


def time_to_timestamp(
        *args,
        shotno='20181018.041',
        t1=0):
    """ Convert an arbitrary number of human readable "seconds after t1"
        to nanoseconds for use with archivedb at this stage times are floats.
        returns list or single value, int.
    Args:
        *args: Arbitrary arguments input
        shotno (str, optional): XP ID
        t1 (int, optional): T1 time
    Returns:
        timestamps (list): of W7X timestamps
    """

    if t1 == 0:
        t1 = int(archivedb.get_program_t1(shotno, useCache=True))
    timestamps = []
    for t in args:
        # Case 1: Time is already in nanoseconds
        if isinstance(t, int) or isinstance(t, np.int64):
            t = int(t)
        # Do nothing, simply convert to integer the string
        # Case 2: Time is passe as minutes:seconds.fraction_of_second
        # from t1
        # This supposes input is string, and has been discarded
        # elif t.find(':') != -1:
        #    t_min=int(t[:t.find(':')])
        #    # skip position 2 as it contains ':'
        #    t_sec=float(t[ t.find(':')+1 : ]) + t_min*60
        #    t=float(t_sec)*1e9 + t1

        # Case 3: Time is passed as seconds.fractions_of_seconds from t1
        else:
            t = t * 1e9 + t1
        timestamps.append(int(t))
    if len(timestamps) == 1:
        timestamps = timestamps[0]

    return timestamps


def timestamp_to_time_after_t1(
        *args,
        shotno='20181018.041',
        t1=0):
    """ timestamps to time after t1 trigger
    Args:
        *args: Arbitrary arguments input
        shotno (str, optional): XP ID
        t1 (int, optional): T1 time
    Returns:
        time_after_t1 (list): times after t1
    """
    # Return input timestamp as string containing seconds and minutes after t1
    time_after_t1 = []
    if t1 == 0:
        t1 = int(archivedb.get_program_t1(shotno, useCache=True))
    for timestamp in args:
        mins = 0
        secs = 0

        timestamp = str(int(timestamp) - t1)
        if float(timestamp) < 1:
            temp_list = ['0']
        elif len(timestamp) <= 9:
            zeroes = ['0'] * (9 - len(timestamp))
            temp_list = [str(secs), '.', ''.join(zeroes), timestamp]

        elif int(timestamp[:-9]) >= 60:
            mins = (int(timestamp[:-9])) // 60
            secs = int((int(timestamp[:-9])) % 60)
            temp_list = [str(mins), ':', str(secs), '.', timestamp[-9:]]
        else:
            secs = int(timestamp[:-9])
            temp_list = [str(secs), '.', timestamp[-9:]]

        valid_time = ''.join(temp_list)
        time_after_t1.append(valid_time)

    if len(time_after_t1) == 1:
        time_after_t1 = time_after_t1[0]

    return time_after_t1


def read_vmec_ids(
        t1,
        t5):
    """ read matching VMEC configs for shot from archive
        (might change throughout a shot in a later stage)
        currently determined via coil currents and Wdia as estimate
        for beta at t(W_dia=max) the vmec run is picked from the VMEC
        webservice database the vmec_id gained needs to be checked by the
        user because the database contains blown up coils and unphysical
        configurations as well! later Minerva/V3Fit/StelOpt...
        reconstructions will be used found under
        "ArchiveDB/raw/W7XAnalysis/Equilibrium/..."
    Args:
        t1 (int): timestamp indicating starting point of the search window
        t5 (int): timestamp indicating ending point of the search window
    Returns:
        vmec_ids = np.array2d, first dimension is timestamps,
            second is corresponding vmec_id
    """
    try:
        parbox = archivedb.get_parameters_box(
            "ArchiveDB/raw/W7XAnalysis/Equilibrium/RefEq_PARLOG" +
            "/V1/parms/equilibriumID", t1, t5, useCache=True)

        vmec_ids = (np.array(parbox['dimensions']),
                    np.array(parbox['values']))

    except Exception:
        print(
            'Vmec_id not yet determined. Please contact Jonathan Schilling \
            (jonathan.schilling@ipp.mpg.de). He will update the database.')

    return (vmec_ids)


def read_active_scatter_vol_params(
        shotno='20181018.0412',
        shot_year=2018,
        TS_version='V10',
        t1=0,
        t5=0):
    """ Setup the scattering volume parameters. Retrieved
        from archive after activity check.
    Args:
        shotno (str): program number in archivedb formatting
        shot_year (int): year in which the shot was taken
        TS_version (str): the version of the Thomson Scattering
            data to refer to
        t1 (int): timestamp indicating starting point of time window
        t5 (int): timestamp indicating ending point of time window
    Returns:
        volumes: data structure containing geometry
            information about the volumes (position...)
        t (ndarray): contains all timestamps for
            which Thomson data is available
        params: data structure containing the parameters
            pertaining to the volumes object (activity...)
    """
    volumes_html = str(urllib2.urlopen(
        "http://archive-webapi.ipp-hgw.mpg.de/Test/" +
        "raw/W7X/QTB_Profile").read())

    volumes_mark = 'volume_(.*?)_'
    # find all possibly available scattering volumes
    volumes = re.findall(volumes_mark, volumes_html)
    volumes = np.unique(volumes)
    volumes = np.sort(np.array(volumes, dtype=int))
    if shot_year == 2017:
        volumes = volumes[volumes <= 1000]
    # read scattering volume informations
    # from archive (positions, size, active...)
    # and exclude inactive volumes
    params = {}

    for i in volumes:
        parameter_url = "Test/raw/W7X/QTB_Profile/volume_%s_PARLOG/%s" % (
            i, TS_version)

        if shot_year == 2017:
            active = archivedb.get_parameters_box(
                "Test/raw/W7X/QTB_Test_V/volume_%s_PARLOG/%s/parms/active" % (
                    i, TS_version), t1, t5, useCache=True)['values'][0]
            # , use_last_version=True)['values'][0]

        elif shot_year >= 2018:
            active = archivedb.get_parameters_box(
                "Test/raw/W7X/QTB_Profile/volume_%s_PARLOG/%s/parms/active"
                % (i, TS_version), t1, t5, useCache=True)['values'][0]
            #     use_last_version=False)['values'][0]

        if active == 0:
            volumes = volumes[volumes != i]
        elif active == 1:
            params[i] = archivedb.get_parameters_box(
                parameter_url, t1, t5, useCache=True)
            # , use_last_version=False)

    # read TS timestaps from archive for first active volume
    TS_timestamp_first_active_volume = \
        "Test/raw/W7X/QTB_Profile/volume_{}_DATASTREAM/{}/1/ne_map".format(
            volumes[0], TS_version)

    try:
        t = archivedb.get_signal(
            TS_timestamp_first_active_volume, t1, t5, useCache=True)[0]
        #     use_last_version=False)[0]
    except Exception:
        print('TS data of the requested version not yet evaluated.\
              Please contact Golo Fuchert (golo.fuchert@ipp.mpg.de).\
              He will update the database.\
              IF YOU KNOW WHAT YOU ARE DOING you MIGHT\
              use automatic TS data version')
    return (volumes, t, params)


def build_volume_data(
        shotno,
        shot_year,
        vmec_ids,
        t,
        t1,
        t5,
        volumes,
        scatter_vol_coordinates,
        TS_version,
        scaling):
    """ gathers all acquired information and organises it in a
        coherent human readable data structure,
        ordered by increasing timestamp, and increasing r_eff.
        Certain quantities are, at this stage, yet to be determined,
        empty columns are setup.
    Args:
        shotno (str): program number in archivedb formatting
        shot_year (int): the year in which the shot was taken
        vmec_ids (ndarray): contains timestamps and
            relative vmec configurations
        t (ndarray): contains all timestamps for which
            Thomson data is available
        t1 (int): timestamp indicating starting point of time window
        t5 (int): timestamp indicating ending point of time window
        volumes: data structure containing geometry
            information about the volumes (position...)
        sorted_geometry: data structure, contains all calculated
            geometric coordinates sorted by r_eff
        TS_version (str): the version of the
            Thomson Scattering data to refer to
        scaling (str): method that will be used to scale the
            Thomson data with respect to the line
            integrated interferometry signal
    Returns:
        data (ndarray): shape(t.size*volumes.size, 37),
            data structure containing (almost) all information relative
            to the shot. Columns are callable by tag.
    """
    trigger1 = int(archivedb.get_program_from_to(shotno, useCache=True)[0])

    # build the full data array: time, vmec_id, volume number and r_eff
    data = np.array(
        np.zeros([t.size * volumes.size]),
        dtype=[('t', np.int64),
               ('shot_time', str),
               ('shotno', '<U16'),
               ('laserno', int),
               ('TS_version', '<U16'),
               ('vmec_id', '<U16'),
               ('factor_real_{}'.format(scaling), float),
               ('factor_mapped_{}'.format(scaling), float),
               ('scattering_volume_number', int),
               ('x_real', float),
               ('y_real', float),
               ('z_real', float),
               ('r_real', float),
               ('r_eff', float),
               ('r_cyl', float),
               ('phi_cyl', float),
               ('s_vmec', float),
               ('theta_vmec', float),
               ('phi_vmec', float),
               ('Te_map', float),
               ('Te_low', float),
               ('Te_high', float),
               ('ne_map', float),
               ('ne_low', float),
               ('ne_high', float),
               ('pe_map', float),
               ('pe_low', float),
               ('pe_high', float),
               ('ne_map_scaled_{}'.format(scaling), float),
               ('ne_low_scaled_{}'.format(scaling), float),
               ('ne_high_scaled_{}'.format(scaling), float),
               ('pe_map_scaled_{}'.format(scaling), float),
               ('pe_low_scaled_{}'.format(scaling), float),
               ('pe_high_scaled_{}'.format(scaling), float),
               ('We_map', float),
               ('We_low', float),
               ('We_high', float)])

    # insert TS time stamps
    data['t'] = np.repeat(t, volumes.size, axis=0)
    # insert time in seconds after t1
    data['shot_time'] = np.repeat(np.array(timestamp_to_time_after_t1(
        *t, shotno=shotno, t1=trigger1)), volumes.size, axis=0)

    # insert shotno
    data['shotno'] = np.repeat(shotno, data['shotno'].size, axis=0)
    # insert matching vmec id per timestamp
    for j in range(vmec_ids[1].size - 1):
        data['vmec_id'][
            (data['t'] >= vmec_ids[0][j]) &
            (data['t'] <= vmec_ids[0][j + 1])] = vmec_ids[1][j]

    # insert TS laserno
    if shot_year == 2017:
        laserno = np.ones(t.size)
        laserno_version = 'None'

    elif shot_year == 2018:
        laser_url = \
            "Test/raw/W7X/QTB_Profile_LaserStream/" + \
            "laser_number_DATASTREAM/{}/0/number".format(TS_version)

        try:
            laserno = archivedb.get_signal(
                laser_url, t1, t5, useCache=True)[1]
            # , use_last_version=False)[1]
            laserno_version = TS_version

        except Exception:
            laserno = archivedb.get_signal(
                laser_url, t1, t5, useCache=True)[1]
            # , use_last_version=True)[1]
            laserno_version = archivedb.get_last_version(
                laser_url, t1, t5, useCache=True)

    data['laserno'] = np.repeat(laserno, volumes.size, axis=0)
    # insert TS data version
    data['TS_version'] = np.repeat(
        TS_version, data['TS_version'].size, axis=0)

    # insert real space scaling factor
    data['factor_real_{}'.format(scaling)] = \
        np.nan * np.ones(data['factor_real_{}'.format(scaling)].size)
    # insert mapped space scaling factor
    data['factor_mapped_{}'.format(scaling)] = \
        np.nan * np.ones(data['factor_real_{}'.format(scaling)].size)

    # insert scatter_vol_coordinates
    for volume_quantity in [
            'scattering_volume_number', 'x_real', 'y_real', 'z_real',
            'r_real', 'r_eff', 'r_cyl', 'phi_cyl', 's_vmec', 'theta_vmec',
            'phi_vmec']:
        data[volume_quantity] = \
            np.tile(scatter_vol_coordinates[volume_quantity],
                    np.unique(t).size)

    # insert TS data from archive
    TS_data = pull_ts_data(
        TS_version, scatter_vol_coordinates, t, t1, t5)
    # retrieve TS data

    TS_data_2D = TS_data.reshape(
        scatter_vol_coordinates['scattering_volume_number'].size * t.size,
        TS_data.shape[2])

    i = 0
    for TS_quantity in [
            'Te_map', 'Te_low', 'Te_high', 'ne_map', 'ne_low', 'ne_high',
            'pe_map', 'pe_low', 'pe_high']:
        data[TS_quantity] = TS_data_2D[:, i]
        i += 1

    if global_debug:
        shot_day = int(shotno[:8])
        np.savetxt(
            '../results/THOMSON/' + str(shot_day) + '/' + shotno[-3:] + '/' +
            'unscaled_TS_data_{shotno}_{TS_version}.txt'.format(
                **locals()),  # data['shotno'][0]),
            data, delimiter=';', fmt='%s')  # header=data.dtype.names,
        # fmt=['%i', '%f', '%s', '%i', '%s', '%s', '%f', '%f', '%i', '%f',
        #      '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f',
        #      '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f',
        #      '%f', '%f', '%f', '%f', '%f', '%f', '%f'],

    return data, laserno_version


def coordinate_transformation(
        vmec_ids,
        volumes,
        params,
        vmec_client,
        shot_year):
    """ calculates r_eff, cylinder and vmec coordinates
        from the x,y,z position from the parlog of the scattering volume for
        all different mag. configs using vmec and extrapolates quadratic or
        linear (see code below) for volumes out of the vmec region. The
        extrapolation variable = distance to magnetic axis (at toroidal
        position of the most central volume).
    Args:
        vmec_ids (ndarray): contains timestamps and
            relative vmec configurations
        volumes: data structure containing geometry information about
            the volumes (position...)
        params: data structure containing the parameters pertaining
            to the volumes object (activity...)
        vmec_client (osa.Client): links to webservice allowing for
            mapping of real space coordinates to vmec
        shot_year (int): the year in which the shot was taken
    Returns:
        scatter_vol_coordinates: data structure, contains all calculated
            geometric coordinates sorted by r_real
    """

    p1 = vmec_client.types.Points3D()
    p2 = vmec_client.types.Points3D()

    for j in range(np.unique(vmec_ids[1]).size):

        scatter_vol_coordinates = np.array(
            np.empty([len(volumes)]),
            dtype=[('scattering_volume_number', int),
                   ('r_real', float),
                   ('r_eff', float),
                   ('x_real', float),
                   ('y_real', float),
                   ('z_real', float),
                   ('r_cyl', float),
                   ('phi_cyl', float),
                   ('s_vmec', float),
                   ('theta_vmec', float),
                   ('phi_vmec', float)])
        p1.x1 = []
        p1.x2 = []
        p1.x3 = []

        for i in range(len(volumes)):
            # read x, y, z Torushall coordinate for each scatter vol
            p1.x1.append(params[volumes[i]]["values"][0]["position"]['x_m'])
            p1.x2.append(params[volumes[i]]["values"][0]["position"]['y_m'])
            p1.x3.append(params[volumes[i]]["values"][0]["position"]['z_m'])
            # scatterining_volume_no =
            # scatter_vol_coordinates['scattering_volume_number']
            scatter_vol_coordinates['scattering_volume_number'][i] = \
                volumes[i]

        # r_eff
        scatter_vol_coordinates['r_eff'] = np.array(
            vmec_client.service.getReff('%s' % vmec_ids[1][j], p1))
        # x_real
        scatter_vol_coordinates['x_real'] = p1.x1
        # y_real
        scatter_vol_coordinates['y_real'] = p1.x2
        # z_real
        scatter_vol_coordinates['z_real'] = p1.x3
        # r_cyl
        scatter_vol_coordinates['r_cyl'] = np.sqrt(
            np.add(np.multiply(p1.x1, p1.x1), np.multiply(p1.x2, p1.x2)))
        p2.x1 = scatter_vol_coordinates['r_cyl']
        # phi_cyl
        scatter_vol_coordinates['phi_cyl'] = np.arctan2(p1.x2, p1.x1)
        p2.x2 = scatter_vol_coordinates['phi_cyl']
        # z_cyl
        p2.x3 = p1.x3
        # cyl_coords to vmec_coords
        vmec_coords = vmec_client.service.toVMECCoordinates(
            '{}'.format(vmec_ids[1][j]), p2, 0.003)
        # s_vmec
        scatter_vol_coordinates['s_vmec'] = vmec_coords.x1
        # theta_vmec
        scatter_vol_coordinates['theta_vmec'] = vmec_coords.x2
        # phi_vmec
        scatter_vol_coordinates['phi_vmec'] = vmec_coords.x3

        # determine position of magnetic axis
        phi_axis = np.arctan2(
            params[int(
                scatter_vol_coordinates['scattering_volume_number'][
                    np.argmin(scatter_vol_coordinates['r_eff'])])][
                "values"][0]["position"]['y_m'],
            params[int(
                scatter_vol_coordinates['scattering_volume_number'][
                    np.argmin(scatter_vol_coordinates['r_eff'])])][
                "values"][0]["position"]['x_m'])

        axis = vmec_client.service.getMagneticAxis(
            '%s' % vmec_ids[1][j], phi_axis)
        axis_cart = np.array([
            axis.x1 * np.cos(axis.x2), axis.x1 * np.sin(axis.x2), axis.x3])

        # determine r_real: the real space
        # distance of all volumes the from magnetic
        # axis at the angle of the innermost volume
        # r_real = scatter_vol_coordinates['r_real']
        for i in scatter_vol_coordinates['scattering_volume_number']:
            scatter_vol_coordinates['r_real'][
                scatter_vol_coordinates['scattering_volume_number'] == i] = \
                ((params[int(i)]["values"][0]["position"]['x_m'] -
                  axis_cart[0])**2 +
                 (params[int(i)]["values"][0]["position"]['y_m'] -
                  axis_cart[1])**2 +
                 (params[int(i)]["values"][0]["position"]['z_m'] -
                  axis_cart[2])**2) ** .5

        # quadratic fit r_eff(r_real) and s_vmec(r_real)
        # seperately for M and N optics
        M_inds_inside_LCFS = np.array(scatter_vol_coordinates[
            'scattering_volume_number'] < 1000) * np.array(
            scatter_vol_coordinates['r_eff'] != np.inf)
        N_inds_inside_LCFS = np.array(scatter_vol_coordinates[
            'scattering_volume_number'] > 1000) * np.array(
            scatter_vol_coordinates['r_eff'] != np.inf)

        if any(i < 1000 for i in volumes):
            M_fit = np.poly1d(
                np.polyfit(scatter_vol_coordinates['r_real'][
                    M_inds_inside_LCFS], scatter_vol_coordinates['r_eff'][
                    M_inds_inside_LCFS], 2))

            M_fit_s = np.poly1d(
                np.polyfit(scatter_vol_coordinates['r_real'][
                    M_inds_inside_LCFS], scatter_vol_coordinates['s_vmec'][
                    M_inds_inside_LCFS], 2))

        if any(i >= 1000 for i in volumes):
            N_fit = np.poly1d(
                np.polyfit(scatter_vol_coordinates['r_real'][
                    N_inds_inside_LCFS], scatter_vol_coordinates['r_eff'][
                    N_inds_inside_LCFS], 2))
            N_fit_s = np.poly1d(
                np.polyfit(scatter_vol_coordinates['r_real'][
                    N_inds_inside_LCFS], scatter_vol_coordinates['s_vmec'][
                    N_inds_inside_LCFS], 2))

        # determine r_eff and s_vmec for volumes out of
        # the LCFS via the fit and multiply inboard side volumes r_eff,
        # s_vmec and r_real by -1 to gain full profiles
        central_M_volume = scatter_vol_coordinates[
            'scattering_volume_number'][scatter_vol_coordinates[
                'scattering_volume_number'] < 1000][np.argmin(
                    scatter_vol_coordinates['r_eff'][scatter_vol_coordinates[
                        'scattering_volume_number'] < 1000])]
        central_N_volume = scatter_vol_coordinates[
            'scattering_volume_number'][scatter_vol_coordinates[
                'scattering_volume_number'] > 1000][np.argmin(
                    scatter_vol_coordinates['r_eff'][scatter_vol_coordinates[
                        'scattering_volume_number'] > 1000])]

        for i in volumes:
            if i < 1000:
                if scatter_vol_coordinates['r_eff'][
                        scatter_vol_coordinates[
                            'scattering_volume_number'] == i] == np.inf:

                    scatter_vol_coordinates['r_eff'][scatter_vol_coordinates[
                        'scattering_volume_number'] == i] = M_fit(
                            scatter_vol_coordinates['r_real'][
                                scatter_vol_coordinates[
                                    'scattering_volume_number'] == i])[0]

                    scatter_vol_coordinates['s_vmec'][scatter_vol_coordinates[
                        'scattering_volume_number'] == i] = M_fit_s(
                            scatter_vol_coordinates['r_real'][
                                scatter_vol_coordinates[
                                    'scattering_volume_number'] == i])[0]

                if i < central_M_volume:
                    scatter_vol_coordinates['r_eff'][scatter_vol_coordinates[
                        'scattering_volume_number'] == i] = -1 * \
                        scatter_vol_coordinates['r_eff'][
                            scatter_vol_coordinates[
                                'scattering_volume_number'] == i]

                    scatter_vol_coordinates['s_vmec'][
                        scatter_vol_coordinates[
                            'scattering_volume_number'] == i] = -1 * \
                        scatter_vol_coordinates['s_vmec'][
                            scatter_vol_coordinates[
                                'scattering_volume_number'] == i]

                    scatter_vol_coordinates['r_real'][
                        scatter_vol_coordinates[
                            'scattering_volume_number'] == i] = -1 * \
                        scatter_vol_coordinates['r_real'][
                            scatter_vol_coordinates[
                                'scattering_volume_number'] == i]

            elif i > 1000:
                if scatter_vol_coordinates['r_eff'][scatter_vol_coordinates[
                        'scattering_volume_number'] == i] == np.inf:

                    scatter_vol_coordinates['r_eff'][
                        scatter_vol_coordinates[
                            'scattering_volume_number'] == i] = N_fit(
                                scatter_vol_coordinates['r_real'][
                                    scatter_vol_coordinates[
                                        'scattering_volume_number'] == i])[0]

                    scatter_vol_coordinates['s_vmec'][
                        scatter_vol_coordinates[
                            'scattering_volume_number'] == i] = N_fit_s(
                                scatter_vol_coordinates['r_real'][
                                    scatter_vol_coordinates[
                                        'scattering_volume_number'] == i])[0]

                if i > central_N_volume:
                    scatter_vol_coordinates['r_eff'][
                        scatter_vol_coordinates[
                            'scattering_volume_number'] == i] = -1 * \
                        scatter_vol_coordinates['r_eff'][
                            scatter_vol_coordinates[
                                'scattering_volume_number'] == i]

                    scatter_vol_coordinates['s_vmec'][
                        scatter_vol_coordinates[
                            'scattering_volume_number'] == i] = -1 * \
                        scatter_vol_coordinates['s_vmec'][
                            scatter_vol_coordinates[
                                'scattering_volume_number'] == i]

                    scatter_vol_coordinates['r_real'][
                        scatter_vol_coordinates[
                            'scattering_volume_number'] == i] = -1 * \
                        scatter_vol_coordinates['r_real'][
                            scatter_vol_coordinates[
                                'scattering_volume_number'] == i]

        # sort volume-quantities increasing by real
        # space distance from magnetic axis
        scatter_vol_coordinates.sort(order='r_real')

        # linearely extrapolate theta values out of LCFS
        extrap_theta_outboard = np.poly1d(np.polyfit(
            scatter_vol_coordinates['r_real'][(
                scatter_vol_coordinates['r_real'] >= 0.3) &
                (scatter_vol_coordinates['r_real'] <= 0.6)],
            scatter_vol_coordinates['theta_vmec'][(
                scatter_vol_coordinates['r_real'] >= 0.3) &
                (scatter_vol_coordinates['r_real'] <= 0.6)], 1))

        scatter_vol_coordinates['theta_vmec'][
            scatter_vol_coordinates['r_real'] >= 0.6] = \
            extrap_theta_outboard(
                scatter_vol_coordinates['r_real'][
                    scatter_vol_coordinates['r_real'] >= 0.6])

        if shot_year >= 2018:
            extrap_theta_inboard = np.poly1d(np.polyfit(
                scatter_vol_coordinates['r_real'][(
                    scatter_vol_coordinates['r_real'] <= -0.3) &
                    (scatter_vol_coordinates['r_real'] >= -0.6)],
                scatter_vol_coordinates['theta_vmec'][(
                    scatter_vol_coordinates['r_real'] <= -0.3) &
                    (scatter_vol_coordinates['r_real'] >= -0.6)], 1))

            scatter_vol_coordinates['theta_vmec'][
                scatter_vol_coordinates['r_real'] <= -0.6] = \
                extrap_theta_inboard(scatter_vol_coordinates['r_real'][
                    scatter_vol_coordinates['r_real'] <= -0.6])

        # linearely extrapolate phi values out of LCFS
        extrap_phi_outboard = np.poly1d(np.polyfit(
            scatter_vol_coordinates['r_real'][(
                scatter_vol_coordinates['r_real'] <= 0.6) &
                (scatter_vol_coordinates['r_real'] >= 0.3)],
            scatter_vol_coordinates['phi_vmec'][(
                scatter_vol_coordinates['r_real'] <= 0.6) &
                (scatter_vol_coordinates['r_real'] >= 0.3)], 1))

        scatter_vol_coordinates['phi_vmec'][
            scatter_vol_coordinates['r_real'] >= 0.6] = \
            extrap_phi_outboard(scatter_vol_coordinates['r_real'][
                scatter_vol_coordinates['r_real'] >= 0.6])

        if shot_year >= 2018:
            extrap_phi_inboard = np.poly1d(np.polyfit(
                scatter_vol_coordinates['r_real'][(
                    scatter_vol_coordinates['r_real'] <= -0.3) &
                    (scatter_vol_coordinates['r_real'] >= -0.6)],
                scatter_vol_coordinates['phi_vmec'][(
                    scatter_vol_coordinates['r_real'] <= -0.3) &
                    (scatter_vol_coordinates['r_real'] >= -0.6)], 1))

            scatter_vol_coordinates['phi_vmec'][
                scatter_vol_coordinates['r_real'] <= -0.6] = \
                extrap_phi_inboard(scatter_vol_coordinates['r_real'][
                    scatter_vol_coordinates['r_real'] <= -0.6])

    return (scatter_vol_coordinates)


def pull_ts_data(
        TS_version='V10',
        scatter_vol_coordinates={},
        t=[],
        t1=0,
        t5=0):
    """ pull TS data (Te, ne and pe + errors) from the archive
    Args:
        TS_version (str, optional):
        scatter_vol_coordinates (dict, optional):
        t (list, optional):
        t1 (int, optional):
        t5 (int, optional):
    Returns:
        TS_data (dict):
    """
    TS_data = np.empty((9, t.size, 0))
    # signal = np.empty((9, t.size))

    for v in scatter_vol_coordinates['scattering_volume_number']:
        TS_data = np.dstack((
            TS_data,
            archivedb.get_signal_box(
                "Test/raw/W7X/QTB_Profile/volume_%s_DATASTREAM/%s" % (
                    int(v), TS_version),
                t1, t5, channels=[0, 6, 7, 1, 8, 9, 2, 10, 11],
                useCache=True)[1]))
    # use_last_version=False,
    # debug=False)

    TS_data = TS_data.transpose(1, 2, 0)
    # Note: TS_data = f[time index, scattering volume index, observable index]
    # such that TS_data[:,0,0] is the temporal evolution of Te of the
    # 0st volume,
    # TS_data[:,0,1] = ne(t), ...

    return TS_data


def calculate_w_e(
        t=[],
        data={},
        volumes=[]):
    """ calculate W_e scaled
    Notes:
        pe = data['pe_map'] in kPa
        k_B = 8.6173303*10**-5 [eV/K] allready included in Te!
    Args:
        data (dict, optional):
        t (list, optional):
        volumes (list, optional):
    Returns:
        None
    """

    We_map = np.empty(t.shape)
    We_high = np.empty(t.shape)
    We_low = np.empty(t.shape)
    for time in np.unique(data['t']):

        # sort data by r_eff
        # mask_low = data['s_vmec']<=1
        sorted_geometry_r_eff_array = np.argsort(
            data['r_eff'][(data['t'] == time) & (data['s_vmec'] <= 1) &
                          (data['s_vmec'] >= -1)]**2)

        pos_pe_map = data['pe_map'][
            (data['t'] == time) & (data['s_vmec'] <= 1) &
            (data['s_vmec'] >= -1)][sorted_geometry_r_eff_array]

        finite_pe_map = np.isfinite(pos_pe_map)

        pos_pe_low = data['pe_low'][
            (data['t'] == time) & (data['s_vmec'] <= 1) &
            (data['s_vmec'] >= -1)][sorted_geometry_r_eff_array]

        finite_pe_low = np.isfinite(pos_pe_low)

        pos_pe_high = data['pe_high'][
            (data['t'] == time) & (data['s_vmec'] <= 1) &
            (data['s_vmec'] >= -1)][sorted_geometry_r_eff_array]

        finite_pe_high = np.isfinite(pos_pe_high)

        pos_Vol = (2 * np.pi**2 * 5.5 * data['r_eff'][
            (data['t'] == time) & (data['s_vmec'] <= 1) &
            (data['s_vmec'] >= -1)]**2)[sorted_geometry_r_eff_array]

        # LSQ univariate splines seem to be sufficent
        # for this case because the data has no large noise
        spline_knots = [0.1 * pos_Vol[finite_pe_map][-1],
                        0.5 * pos_Vol[finite_pe_map][-1]]
        LSQuni_map = \
            interpolate.LSQUnivariateSpline(
                pos_Vol[finite_pe_map], pos_pe_map[finite_pe_map],
                spline_knots)
        LSQuni_low = \
            interpolate.LSQUnivariateSpline(
                pos_Vol[finite_pe_low], pos_pe_low[finite_pe_low],
                spline_knots)
        LSQuni_high = \
            interpolate.LSQUnivariateSpline(
                pos_Vol[finite_pe_high], pos_pe_high[finite_pe_high],
                spline_knots)

        # fit TS density profile mapped to the interferometry LOSs
        pe_map_fit = LSQuni_map(np.arange(0, pos_Vol[-1], 0.1))
        pe_low_fit = LSQuni_low(np.arange(0, pos_Vol[-1], 0.1))
        pe_high_fit = LSQuni_high(np.arange(0, pos_Vol[-1], 0.1))

        # 1.5 from E_kin = 3/2 m*v**2 for mono-particles with f=3
        We_map[np.unique(data['t']) == time] = \
            1.5 * np.trapz(pe_map_fit, np.arange(0, pos_Vol[-1], 0.1))
        We_low[np.unique(data['t']) == time] = \
            1.5 * np.trapz(pe_low_fit, np.arange(0, pos_Vol[-1], 0.1))
        We_high[np.unique(data['t']) == time] = \
            1.5 * np.trapz(pe_high_fit, np.arange(0, pos_Vol[-1], 0.1))

    data['We_map'] = np.repeat(We_map, volumes.size, axis=0)
    data['We_low'] = np.repeat(We_low, volumes.size, axis=0)
    data['We_high'] = np.repeat(We_high, volumes.size, axis=0)

    return


def gauss_fitting(
        r_eff_inter_los_in=[],
        r_eff_TS_in_mir=[],
        finite_data_in_mir=[],
        data_in_mir=[]):
    """ way to gauss fit
    Args:
        r_eff_inter_los_in (list, optional):
        r_eff_TS_in_mir (list, optional):
        finite_data_in_mir (list, optional):
        data_in_mir (list, optional):
    Returns:
        ne_inter_profile_los (list):
        ne_inter_profile_los_cov (list):
    """
    # Gaussian process regression
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + \
        WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0).fit(
        r_eff_TS_in_mir[finite_data_in_mir][:, np.newaxis],
        data_in_mir[finite_data_in_mir])

    # interpolate mirrored TS data in vmec
    # coordinates using the Guassian process fit
    ne_inter_profile_los, ne_inter_profile_los_cov = \
        gp.predict(r_eff_inter_los_in[:, np.newaxis], return_cov=True)

    return ne_inter_profile_los, ne_inter_profile_los_cov


def lsq_fitting(
        r_eff_inter_los1_in=[],
        r_eff_inter_los2_in=[],
        r_eff_TS_in_mir=[],
        finite_data_in_mir=[],
        data_in_mir=[],
        minor_radius=0.0):
    """ least sqares fitting
    Args:
        r_eff_inter_los1_in (list, optional):
        r_eff_inter_los2_in (list, optional):
        r_eff_TS_in_mir (list, optional):
        finite_data_in_mir (list, optional):
        data_in_mir (list, optional):
        minor_radius (float, optional):
    Returns:
        ne_inter_profile_los1 (list):
        ne_inter_profile_los2 (list):
    """
    # 'old' LSQ univariate spline regression if sklearn package is unavailable
    LSQuni = interpolate.LSQUnivariateSpline(
        r_eff_TS_in_mir[finite_data_in_mir], data_in_mir[finite_data_in_mir],
        [-minor_radius, -0.8 * minor_radius, -0.5 * minor_radius, 0,
         0.5 * minor_radius, 0.8 * minor_radius, minor_radius])

    # fit TS density profile mapped to the interferometry LOSs
    ne_inter_profile_los1 = LSQuni(r_eff_inter_los1_in)
    ne_inter_profile_los2 = LSQuni(r_eff_inter_los2_in)

    return ne_inter_profile_los1, ne_inter_profile_los2


def plot_sanity_check(
        r_real_TS,
        data_in_mir,
        finite_data,
        r_real_fit,
        r_eff_TS_in_mir,
        ne,
        ne_low,
        ne_up,
        r_eff_inter_los1_in,
        r_eff_in_mir,
        secs):

    # plot fitted profiles for sanity check
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("mapped fits @ {} s" .format(secs))
    # ax.plot(data['r_eff'][data['t']==i], data['ne_map'][data['t']==i], 'o')
    ax.plot(r_real_TS, data_in_mir, 'o', label=r'data')
    ax.plot(r_real_fit, ne, 'k', label=r'gauss')
    ax.set_xlabel(r'$r_{real}$')
    ax.set_ylabel(r'$n_{e} (m^{-3})$')
    ax.fill_between(r_real_fit, ne_low, ne_up,
                    alpha=0.5, color='k')

    # Comparison of different fitting routines
    # interpolate mirrored TS data in vmec coordinates
    # using LSQ Univariate splines
    # ax.plot(r_real_fit, LSQuni(r_eff_inter_los1_in), 'r', label='LSQuni')
    # interpolate mirrored TS data in vmec
    # coordinates using robust Univariate splines
    # uni = interpolate.UnivariateSpline(r_eff_TS_in_mir[finite_data],
    # data_in_mir[finite_data], s=r_eff_in_mir.size/8)
    # ax.plot(r_real_fit, uni(r_eff_inter_los1_in), 'g', label='uni')
    # # interpolate mirrored TS data in vmec coordinates using Cubic splines
    # cs = interpolate.CubicSpline(
    #     r_eff_TS_in_mir[finite_data], data_in[finite_data]),
    #     s=r_eff_in_mir.size/8)
    # ax.plot(r_real_fit, cs(r_eff_inter_los1_in), 'b', label='cubic')

    ax.legend()
    plt.show()
    return


def burst_mode(
        t1=0,
        t5=0,
        burst=False):
    """ burst mode
    Args:
        t1 (int, optional):
        t5 (int, optional):
        burst (bool, optional):
    Returns:
        t_inter (int):
        ne_inter (int):
    """

    # print("Calling Burst_Mode")
    if burst:
        # TODO if shotno in ... genfromtxt... else slow interferometry
        # TODO define burst mode/avalable fast interferometry shot file
        # fast interferometry, available by request
        # from k.j.brunner - currently under:
        t_inter, ne_inter = archivedb.get_signal(
            "Test/raw/W7XAnalysis/QMJ_IEDDI/IED_DDR/V1/1/density",
            t1, t5, useCache=True)

    else:
        # slow but always available ethernet signal (jitter up to 10 ms!!!)
        # this signal is sufficent for most of the cases,
        # e.g. if no fast (<10ms) physics is studied
        t_inter, ne_inter = archivedb.get_signal(
            "ArchiveDB/codac/W7X/CoDaStationDesc.16339/" +
            "DataModuleDesc.16341_DATASTREAM/0/" +
            "Line integrated density/scaled", t1, t5, useCache=True)
        t_inter = t_inter - 10e6

    return t_inter, ne_inter


def calculate_upper_lower_integrals(
        data_x=(),
        data=(),
        fit_x=(),
        fit='none'):
    """ integral limits
    Args:
        data_x (tuple, optional):
        data (tuple, optional):
        fit_x (tuple, optional):
        fit (str, optional):
    Returns:
        upper (float):
        lower (float):
    """
    std = np.std((data - np.interp(data_x, fit_x, fit)))
    upper = np.trapz(fit + std, fit_x)
    lower = np.trapz(fit - std, fit_x)
    return upper, lower


def scale_ts_data(
        t=[],
        t1=0,
        t5=0,
        data={},
        vmec_ids=[],
        shot_year=2018,
        vmec_client=None,
        vmec_base='none',
        volumes=[],
        burst=False,
        sanity_checks=False,
        scaling='gauss',
        debug=False):
    """ scale line-int. ne (and pe) + errors from TS to
        interferometry int_ne_dl integrated in real space
        "radius" along the two CO2 laser beam LOSs
    Notes:
        The script provides scaling factors for the mapped TS data
        (one of) the scaling factors has to be applied manually by the user!
        two factors are given in OP1.b:
        a) from scaling after integration in real
            space assuming exactly equal LOSs
        for TS and Interferometry
        b) from scaling TS data after mapping to the Interferometry
            LOSs using vmec real space scaling is not available
            in OP1.2a because full profiles are required!
        (the real space factor will be nan)
    Args:
        t (int, optional):
        t1 (int, optional):
        data (dict, optional):
        vmec_ids (list, optional):
        shot_year (int, optional):
        vmec_client (None, optional):
        vmec_base (str, optional):
        volumes (list, optional):
        burst (bool, optional):
        sanity_checks (bool, optional):
        scaling (str, optional):
        debug (bool, optional):
    """
    factor = np.nan * np.ones(t.size)
    vmec_id_old = "not_yet_set"
    ne_TS_integrated = np.array([])
    ne_inter_integrated = np.array([])  # Estimate for ne by interferometry
    factor_error_up = np.array([])
    factor_error_low = np.array([])
    t_inter, ne_inter = burst_mode(t1, t5, burst)
    inter_error = np.sqrt((1e18)**2 + (4e18)**2)
    # ERROR on interferometry is hardcoded only temporarily

    # entrance and exit points of the interferometer CO2 beam
    los1 = np.array(((-4.349577, 0.604697, 0.503093),
                     (-6.284106, 1.141353, -0.188108)))  # neg. z = outboard
    los2 = np.array(((-4.340268, 0.593509, 0.474497),
                     (-6.291524, 1.134799, -0.222691)))

    # set up 1000 LOS points and their respective distance in 3D/real space
    volumes_los1 = np.empty((1001, 3))
    volumes_los2 = np.empty((1001, 3))
    r_real_inter_los1 = np.empty((1001, 1))
    r_real_inter_los2 = np.empty((1001, 1))

    for i in range(1001):
        volumes_los1[i] = los1[0] - 0.001 * i * (los1[0] - los1[1])
        # from inboard side to outboard side

        volumes_los2[i] = los2[0] - 0.001 * i * (los2[0] - los2[1])
        r_real_inter_los1[i] = np.linalg.norm(volumes_los1[i] - los1[0])
        r_real_inter_los2[i] = np.linalg.norm(volumes_los2[i] - los2[0])

    # initiates 2 vmec points/vectors
    p1 = vmec_client.types.Points3D()
    p2 = vmec_client.types.Points3D()

    # fit_rout = input("Data fitting ('gauss' employs gaussian kernel,
    # 'LSQ' a univariate spline, kwarg "fit_rout = 'n'" skips this step): ")

    L = len(np.unique(data['t']))
    # for i in np.unique(data['t']):
    for j, i in enumerate(np.unique(data['t'])):
        # print(i, j)
        if (j / L) % 0.02 == 0.0 and debug:
            print(str(int((j / L) * 100.)) + '%', end=' ')

        if sanity_checks:
            sanity = 0

        vmec_id = data['vmec_id'][data['t'] == i][0]
        # get the r_eff of the 1000 LOS points
        # only call vmec points multiple times if vmec
        # equilibrium changes troughout the shot
        if vmec_id != vmec_id_old:
            vmec_id_old = vmec_id
            p1.x1 = volumes_los1[:, 0]
            p1.x2 = volumes_los1[:, 1]
            p1.x3 = volumes_los1[:, 2]
            p2.x1 = volumes_los2[:, 0]
            p2.x2 = volumes_los2[:, 1]
            p2.x3 = volumes_los2[:, 2]

            for j in range(np.unique(vmec_ids[1]).size):
                r_eff_inter_los1 = np.array(
                    vmec_client.service.getReff('%s' % vmec_ids[1][j], p1))

                r_eff_inter_los2 = np.array(
                    vmec_client.service.getReff('%s' % vmec_ids[1][j], p2))

            # remove points out of LCFS
            r_real_inter_los1_in = r_real_inter_los1[
                r_eff_inter_los1 != np.inf]
            r_real_inter_los2_in = r_real_inter_los2[
                r_eff_inter_los2 != np.inf]

            r_eff_inter_los1_in = r_eff_inter_los1[r_eff_inter_los1 != np.inf]
            r_eff_inter_los2_in = r_eff_inter_los2[r_eff_inter_los2 != np.inf]

            # differentiate between inboard (negative)
            # and outboard (positive) r_eff
            r_eff_inter_los1_in[
                0:np.argwhere(r_eff_inter_los1_in == np.nanmin(
                    r_eff_inter_los1_in))[0][0]:1] = -1 * \
                r_eff_inter_los1_in[
                    0:np.argwhere(r_eff_inter_los1_in == np.nanmin(
                        r_eff_inter_los1_in))[0][0]:1]

            r_eff_inter_los2_in[
                0:np.argwhere(r_eff_inter_los2_in == np.nanmin(
                    r_eff_inter_los2_in))[0][0]:1] = -1 * \
                r_eff_inter_los2_in[
                    0:np.argwhere(r_eff_inter_los2_in == np.nanmin(
                        r_eff_inter_los2_in))[0][0]:1]

        # call radii for TS
        minor_radius = float(json.loads(urllib2.urlopen(
            vmec_base + vmec_id +
            '/minorradius.json').read().decode('utf-8'))['minorRadius'])

        r_eff_TS_in = data['r_eff'][data['t'] == i]
        r_real_TS_in = data['r_real'][data['t'] == i]
        data_in = data['ne_map'][data['t'] == i]

        # TODO res = getattr(globals(),
        # '_function_for_year_{shot_year}')
        # think about this
        if shot_year == 2017:
            # mirror data in vmec coordinates to gain symetric profile fit
            data_in = data_in[np.argsort(r_eff_TS_in)]
            r_real_TS_in = r_real_TS_in[np.argsort(r_eff_TS_in)]
            r_eff_TS_in = r_eff_TS_in[np.argsort(r_eff_TS_in)]
            r_eff_TS_in = abs(r_eff_TS_in)
            data_in_mir = np.append(np.flipud(data_in), data_in)
            finite_data_in_mir = np.isfinite(data_in_mir)
            r_eff_TS_in_mir = np.append(
                -1 * np.flipud(r_eff_TS_in), r_eff_TS_in)
            r_real_TS_in_mir = np.append(
                -1 * np.flipud(r_real_TS_in), r_real_TS_in)

            # if fit_rout=='gauss':
            r_real_TS = r_real_TS_in_mir
            r_real_fit = r_real_inter_los1_in[:, 0] - .5 * (
                r_real_inter_los1_in[:, 0][0] +
                r_real_inter_los1_in[:, 0][-1])
            data_in = data_in_mir
            finite_data = finite_data_in_mir

            if scaling == 'gauss':
                ne_inter_profile_los1, ne_inter_profile_los1_cov = \
                    gauss_fitting(
                        r_eff_inter_los1_in, r_eff_TS_in_mir,
                        finite_data_in_mir, data_in_mir)
                ne_inter_profile_los2, ne_inter_profile_los2_cov = \
                    gauss_fitting(
                        r_eff_inter_los2_in, r_eff_TS_in_mir,
                        finite_data_in_mir, data_in_mir)

                r_real_TS_fit = r_eff_inter_los1_in
                ne_low = ne_inter_profile_los1 - np.sqrt(np.diag(
                    ne_inter_profile_los1_cov))
                # Only los1 is used, maybe add average of both?

                ne_up = ne_inter_profile_los1 + np.sqrt(np.diag(
                    ne_inter_profile_los1_cov))
                ne_fit = ne_inter_profile_los1
                ne_inter_low = ne_low
                ne_inter_up = ne_up
                # To avoid issues with the code since only
                # one of the 2 factors can be calculated

                tmp1, tmp2 = calculate_upper_lower_integrals(
                    r_eff_TS_in_mir[finite_data_in_mir],
                    data_in_mir[finite_data_in_mir], r_eff_inter_los1_in,
                    ne_inter_profile_los1)
                integral1 = np.trapz(ne_inter_profile_los1,
                                     r_real_inter_los1_in[:, 0])
                integral2 = np.trapz(
                    ne_inter_profile_los2, r_real_inter_los2_in[:, 0])

            elif scaling == 'raw':
                r_real_TS_fit = data['r_eff'][data['t'] == i]
                # Not actually r_real, but needed for integration loop end

                ne_fit, ne_up, ne_low = \
                    data[data['t'] == i]['ne_map'], \
                    data[data['t'] == i]['ne_high'], \
                    data[data['t'] == i]['ne_low']
                tmp1, tmp2 = np.trapz(
                    data[data['t'] == i]['ne_high'],
                    data['r_eff'][data['t'] == i]), \
                    np.trapz(data[data['t'] == i]['ne_low'],
                             data['r_eff'][data['t'] == i])

                ne_inter_low = ne_low
                ne_inter_up = ne_up
                ne_inter_profile_los1 = ne_fit
                integral1 = np.trapz(
                    data_in_mir[finite_data_in_mir],
                    r_eff_TS_in_mir[finite_data_in_mir])
                integral2 = np.trapz(
                    data_in_mir[finite_data_in_mir],
                    r_eff_TS_in_mir[finite_data_in_mir])

            if sanity_checks:
                if scaling == 'raw':
                    if global_debug:
                        print("No sanity check to perform as data is scaled" +
                              "by raw data integration.")

                elif sanity % 10 == 0:
                    # print(data[data['t'] == i]['shotno'][0])
                    secs = timestamp_to_time_after_t1(
                        i, shotno=data[data['t'] == i]['shotno'][0], t1=t1)[0]
                    plot_sanity_check(
                        r_real_TS_in_mir, data_in_mir, finite_data,
                        r_real_fit, r_eff_TS_in_mir,
                        ne_inter_profile_los1, ne_inter_low, ne_inter_up,
                        r_eff_inter_los1_in, r_eff_TS_in_mir, secs)

            # Plot_Fitted_Profiles(
            #     data, r_real_TS, data_in, finite_data,
            #     r_real_fit, r_eff_TS_in_mir,
            #     ne, ne_low, ne_up, r_eff_inter_los1_in, r_eff_in_mir, t1, i)

            # elif fit_rout=='LSQ':
            #     ne_inter_profile_los1, ne_inter_profile_los2 = LSQ_Fitting(
            #         r_eff_inter_los1_in,
            #         r_eff_inter_los2_in, r_eff_TS_in_mir,
            #         finite_data_in_mir, data_in_mir, minor_radius)
            #     ne_low = ne_inter_profile_los1
            #     # Dummy argument, not used but need for
            #     # Plot_Fitted_Profiles
            #     ne_up = ne_inter_profile_los1
            #     # Dummy argument, not used but need for
            #     # Plot_Fitted_Profiles

        elif shot_year == 2018:
            # remove inboard side scattering volumes out of the LCFS
            data_in = data_in[(r_eff_TS_in >= -1 * minor_radius)]
            r_real_TS_in = r_real_TS_in[(r_eff_TS_in >= -1 * minor_radius)]
            r_eff_TS_in = r_eff_TS_in[(r_eff_TS_in >= -1 * minor_radius)]

            # mirror data in vmec coordinates to gain symetric profile fit
            data_in = data_in[np.argsort(r_eff_TS_in)]
            r_real_TS_in = r_real_TS_in[np.argsort(r_eff_TS_in)]
            r_eff_TS_in = r_eff_TS_in[np.argsort(r_eff_TS_in)]
            r_eff_TS_in = abs(r_eff_TS_in)

            data_in_mir = np.append(np.flipud(data_in), data_in)

            finite_data_in_mir = np.isfinite(data_in_mir)

            r_eff_TS_in_mir = np.append(
                -1 * np.flipud(r_eff_TS_in), r_eff_TS_in)
            r_real_TS_in_mir = np.append(
                -1 * np.flipud(r_real_TS_in), r_real_TS_in)

            # print(r_eff_TS_in_mir)
            # if fit_rout == 'gauss':
            if scaling == 'gauss':
                ne_inter_profile_los1, ne_inter_profile_los1_cov = \
                    gauss_fitting(
                        r_eff_inter_los1_in, r_eff_TS_in_mir,
                        finite_data_in_mir, data_in_mir)

                ne_inter_profile_los2, ne_inter_profile_los2_cov = \
                    gauss_fitting(
                        r_eff_inter_los2_in, r_eff_TS_in_mir,
                        finite_data_in_mir, data_in_mir)

                ne_inter_low = ne_inter_profile_los1 - \
                    np.sqrt(np.diag(ne_inter_profile_los1_cov))
                ne_inter_up = ne_inter_profile_los1 + \
                    np.sqrt(np.diag(ne_inter_profile_los1_cov))

                integral1 = np.trapz(
                    ne_inter_profile_los1, r_real_inter_los1_in[:, 0])
                integral2 = np.trapz(
                    ne_inter_profile_los2, r_real_inter_los2_in[:, 0])

            elif scaling == 'raw':
                ne_inter_profile_los1, ne_inter_profile_los2, \
                    ne_inter_low, ne_inter_up = \
                    data[data['t'] == i]['ne_map'], \
                    data[data['t'] == i]['ne_map'], \
                    data[data['t'] == i]['ne_high'],\
                    data[data['t'] == i]['ne_low']

                integral1 = np.trapz(
                    data_in_mir[finite_data_in_mir],
                    r_eff_TS_in_mir[finite_data_in_mir])
                integral2 = np.trapz(
                    data_in_mir[finite_data_in_mir],
                    r_eff_TS_in_mir[finite_data_in_mir])

            # elif fit_rout == 'LSQ':
            #    ne_inter_profile_los1, ne_inter_profile_los2 = LSQ_Fitting(
            #        r_eff_inter_los1_in, r_eff_inter_los2_in,
            #        r_eff_TS_in_mir, finite_data_in_mir, data_in_mir,
            #        minor_radius)
            #    ne_low = ne_inter_profile_los1
            #    # Dummy argument, not used but need for
            #    # Plot_Fitted_Profiles
            #    ne_up = ne_inter_profile_los1
            #    # Dummy argument, not used but need for Plot_Fitted_Profiles

            r_real_TS = r_real_TS_in_mir
            finite_data = np.isfinite(data['ne_map'][data['t'] == i])
            r_real_TS_fit = np.arange(r_real_TS[0], r_real_TS[-1], 0.001)
            r_real_fit = r_real_inter_los1_in[:, 0] - .5 * (
                r_real_inter_los1_in[:, 0][0] +
                r_real_inter_los1_in[:, 0][-1])

            # LSQuni = interpolate.LSQUnivariateSpline(
            #     r_eff_TS_in_mir[finite_data],
            #     data['ne_map'][data['t']==i][finite_data],
            #     [-minor_radius, -0.8 * minor_radius, -0.5 * minor_radius, 0,
            #     0.5 * minor_radius,\
            #    0.8*minor_radius, minor_radius])

            if sanity_checks:
                if scaling == 'raw':
                    if global_debug:
                        print("No sanity check to perform as data is" +
                              " scaled by raw data integration.")
                elif sanity % 10 == 0:
                    # print(data[data['t'] == i]['shotno'][0])
                    secs = timestamp_to_time_after_t1(
                        i, shotno=data[data['t'] == i]['shotno'][0], t1=t1)[0]
                    plot_sanity_check(
                        r_real_TS_in_mir, data_in_mir, finite_data,
                        r_real_fit, r_eff_TS_in_mir, ne_inter_profile_los1,
                        ne_inter_low, ne_inter_up, r_eff_inter_los1_in,
                        r_eff_TS_in_mir, secs)

            r_real_TS = data['r_real'][data['t'] == i]
            finite_data = np.isfinite(data['ne_map'][data['t'] == i])
            if scaling == 'gauss':
                r_real_TS_fit = np.arange(r_real_TS[0], r_real_TS[-1], 0.001)
                ne_fit, ne_err = \
                    gauss_fitting(r_real_TS_fit, r_real_TS,
                                  finite_data, data['ne_map'][data['t'] == i])
                # Calculate upper and lower integrals for
                # statistical uncertainty
                tmp1, tmp2 = calculate_upper_lower_integrals(
                    r_real_TS[finite_data], data['ne_map'][data['t'] == i][
                        finite_data], r_real_TS_fit, ne_fit)

                ne_up = ne_fit + np.sqrt(np.diag(ne_err))
                ne_low = ne_fit - np.sqrt(np.diag(ne_err))

            elif scaling == 'raw':
                r_real_TS_fit = r_real_TS
                ne_fit, ne_up, ne_low = data[data['t'] == i]['ne_map'], \
                    data[data['t'] == i]['ne_high'], \
                    data[data['t'] == i]['ne_low']

                tmp1, tmp2 = np.trapz(
                    data[data['t'] == i]['ne_high'],
                    data['r_real'][data['t'] == i]), \
                    np.trapz(data[data['t'] == i]['ne_low'],
                             data['r_real'][data['t'] == i])

            if sanity_checks:
                if scaling == 'raw':
                    if global_debug:
                        print("No sanity check to perform as data is " +
                              "scaled by raw data integration.")
                elif sanity % 10 == 0:
                    plot_sanity_check(
                        r_real_TS, data['ne_map'][data['t'] == i],
                        finite_data, r_real_TS_fit, r_eff_TS_in_mir,
                        ne_fit, ne_low, ne_up, r_eff_inter_los1_in,
                        r_eff_TS_in_mir, secs)

        # integerate TS density profile in real space along TS Laser LOSs
        real_integral = np.trapz(ne_fit, r_real_TS_fit)
        ne_TS_integrated = np.append(
            ne_TS_integrated, real_integral)

        # Compute relative error on scaling factor
        tmp1 = np.sqrt(
            (inter_error / np.interp(i, t_inter, ne_inter))**2 +
            ((tmp1 - real_integral) / real_integral)**2)
        tmp2 = np.sqrt(
            (inter_error / np.interp(i, t_inter, ne_inter))**2 +
            ((tmp2 - real_integral) / real_integral)**2)

        factor_error_up = np.append(factor_error_up, tmp1)
        factor_error_low = np.append(factor_error_low, tmp2)

        # integerate TS density profile in real space
        # on both interferometry LOSs
        # and devide by 2 for camparison with
        # Interferometry single path signal
        ne_inter_integrated = np.append(
            ne_inter_integrated, 0.5 * (integral1 + integral2))

        if sanity_checks:
            sanity += 1

    # calculation of the cross-calibarion factor
    factorLOS = np.interp(
        np.int64(np.unique(data['t'])), t_inter, ne_inter) / \
        (1e19 * ne_TS_integrated)

    # put real space factor into the data array
    data['factor_real_{}'.format(scaling)] = np.repeat(
        factorLOS, volumes.size, axis=0)

    ##########################################################################
    # Golos correction factor OP1.2a
    # factor = archivedb.get_signal("Test/raw/W7X/QTB_correction/
    #     interf_DATASTREAM/0/corrVac", t1, t5_2s)[1]
    ##########################################################################
    # calculation of the cross-calibration factor
    factor = np.interp(
        np.int64(np.unique(data['t'])), t_inter, ne_inter) / \
        (1e19 * ne_inter_integrated)

    # compute absolute error on scaling factor
    factor_error_up = factorLOS * factor_error_up
    factor_error_low = factorLOS * factor_error_low

    # put mapped factor into the data array
    data['factor_mapped_{}'.format(scaling)] = np.repeat(
        factor, volumes.size, axis=0)
    factor_error_up = np.repeat(factor_error_up, volumes.size, axis=0)
    factor_error_low = np.repeat(factor_error_low, volumes.size, axis=0)

    for word in ['ne', 'pe']:
        s_map = '_'.join([word, 'map', 'scaled', scaling])
        # e.g.  ne_map_scaled_raw
        s_high = '_'.join([word, 'high', 'scaled', scaling])
        s_low = '_'.join([word, 'low', 'scaled', scaling])
        loop_map = '_'.join([word, 'map'])
        high = '_'.join([word, 'high'])
        low = '_'.join([word, 'low'])
        data[s_map] = data[loop_map] * data['factor_real_{}'.format(scaling)]
        # Propagate errors from scaling factor
        data[s_high] = (data[s_map] + np.sqrt(
            (data['factor_real_{}'.format(scaling)] *
             (data[high] - data[loop_map])**2 +
             (factor_error_up * data[loop_map])**2)))
        data[s_low] = (data[s_map] - np.sqrt(
            (data['factor_real_{}'.format(scaling)] *
             (data[low] - data[loop_map])**2 +
             (factor_error_low * data[loop_map])**2)))

    return
