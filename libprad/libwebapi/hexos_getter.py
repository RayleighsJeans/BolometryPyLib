""" ********************************************************************** """
# Created on Mon Mar 19 14:17:55 2018
# @author: twegner
# @imported: phacker
# TODO: normalisieren als flag

import sys
import os

import numpy as np
import scipy as sp
import MDSplus
import json
import glob

from HEXOScamera import HEXOSCamera
import logbook_api as logbook
from mClass import int_to_Roman as i2r
from mClass import Roman_to_int as r2i
import plot_funcs as pf
import webapi_access as api

Z = np.zeros
ones = np.ones

stdwrite = sys.stdout.write
stdflush = sys.stdout.flush

# timing factor
timeprolong = 1

""" ********************************************************************** """


def load_LineData_HEXOS():

    # header and 0th row
    # 0  :    --
    # 1  :  lambda_lit  --  2.478
    # 2  :  E_lit  --  500.340
    # 3  :  obs  --  -1.000
    # 4  :  ovplot  --  0
    # 5  :  group  --  -1
    # 6  :  std line  --  0
    # 7  :  calib  --  0
    # 8  :  code  --  7.06
    # 9  :  el  --  N
    # 10 :  ion  --  VII
    # 11 :  comment  --  R; mult (2.478+2.478)
    # 12 :  transition  --  1s - 2p
    # 13 :  E_up  --  500.300
    # 14 :  source  --  CAMDB
    # 15 :  reference  --
    # 16 :  (a)dded / (m)odified  --  m: B. Buttenschoen

    file = r'../../LineData/VUVlines_W7X.linedata.csv'
    lambda_lit = np.loadtxt(file, usecols=[1], delimiter='|', dtype=np.float)
    E_lit = np.loadtxt(file, usecols=[2], delimiter='|', dtype=np.float)
    el = np.loadtxt(file, usecols=[9], delimiter='|', dtype=str)
    ion = [np.loadtxt(file, usecols=[9], delimiter='|', dtype=str)[i] +
           ' ' + x for i, x in enumerate(np.loadtxt(
               file, usecols=[10], delimiter='|', dtype=str))]
    transition = np.loadtxt(file, usecols=[12], delimiter='|', dtype=str)
    E_up = [np.float(x) if x != '' else None for
            i, x in enumerate(np.loadtxt(
                file, usecols=[13], delimiter='|', dtype=str))]

    return (lambda_lit, E_lit,
            el, ion, transition, E_up)


def get_hexos_xics(
        date=20181010,
        shot=32,
        mat='Fe',
        debug=False,
        saving=True,
        hexos=True,
        xics=False,
        indent_level='\t'):
    """ get HEXOS and XICS data
    Args:
        date (int, optional): XP day
        shot (int, optional): XP ID
        mat (str, optional): metrial/element
        debug (bool, optional): debugging
        saving (bool, optional): saving
        hexos (bool, optional): should do hexos
        xics (bool, optional): should do xics
    Returns:
       hexos_dict (0, dict): HEXOS results
       xics_dict (1, dict): XICS results
    Notes:
        None
    """
    try:
        # get timestamps for shot:
        a, d = logbook.db_request_lobgook(
            date=str(date), shot=shot)

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print('\t\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + '\t\t\\\  db_req failed')
        return

    try:
        # look for XP ID in logbook
        time_start = d['_source']['from'] + 61000000000
        if debug:
            stdwrite('\n' + indent_level + '>> get data: XP ID: ' +
                     str(d['_id'][3:]) + '\n')
            stdflush()

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print('\t\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + '\t\t\\\  shot not found')

    # ECRH duration
    tags = d['_source']['tags']
    for i in range(len(tags)):
        if tags[i].get('name') == 'ECRH duration':
            powertime = np.float(tags[i]['value'])

    for n in ['HEXOS', 'XICS']:
        path = '../results/' + n + '/' + str(date) + '/'

        # check if dir exist:
        if saving == 1:
            if not os.path.exists(path):
                os.makedirs(path)
                if debug:
                    stdwrite('\n' + indent_level + '\\\ get data: path ' +
                             path + ' is created')
                    stdflush()

    hexos_dict, xics_dict = None, None
    if hexos:
        hexos_dict = get_hexos(
            mat=mat,
            start=time_start - 0.5e9,
            end=(powertime + timeprolong) * 1e9 + time_start,
            path='../results/HEXOS/' + str(date) + '/',
            date=date,
            shot=shot,
            debug=debug,
            saving=saving,
            indent_level=indent_level)
    if xics:
        xics_dict = get_xics(
            date=date,
            shot=shot,
            path='../results/XICS/' + str(date) + '/',
            saving=saving,
            extended=True,
            indent_level=indent_level)

    return (hexos_dict, xics_dict)


def smooth(
        y=np.ones((10000)),
        box_pts=100):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def get_hexos(
        mat='C',
        start=5.,
        end=10.,
        path='../results/HEXOS/20181010',
        date=20181010,
        shot=32,
        debug=False,
        saving=True,
        indent_level='\t'):
    """ downloading HEXOS data and returning
    Args:
        mat (0, str, optional): meterial/element to look for
        start (1, float, optional): time of start POSIX
        end (2, float, optional): time of end POSIX
        path (3, str, optional): writing path
        date (4, int, optional): XP date
        shot (5, int, optional): XP ID
        debug (6, bool, optional): debug printing
        saving (7, bool, optional): saving bool
    Returns:
        tmp (0, dict): results dictionary with all lines defined for that
            specific material/element above
    Notes:
        None
    """
    file = path + 'hexos_' + str(date) + '.' + \
        str(shot).zfill(3) + '_' + mat + '.json'

    if not glob.glob(file) == []:
        if debug:
            print(indent_level + '\t\\\ HEXOS data found, loading...')
        try:
            with open(file, 'r') as infile:
                tmp = json.load(infile)
            infile.close()
            return(tmp)

        except Exception:
            print(indent_level + '\t\\\ failed loading HEXOS data')
            return(None)

    # line info for hexos
    # TODO: extend line infos on C, O etc.
    hexmat, calib_poly = hexos_getInfo(material=mat)
    tbackground = [60.6, 60.8]  # [3, 23.415,'Ni XXVI'],
    flag, offset = 0, .0
    pixels = sp.linspace(1, 1024, 1024)

    # saving dict
    tmp = {'label': 'HEXOS_data',
           'xpid': str(date) + '.' + str(shot).zfill(3),
           'E_lit': {},
           'lambda_lit': {},
           'values': {}}

    strr = []
    for i, c in enumerate(hexmat[:, 0]):
        camera = int(c)
        wav = np.float(hexmat[i, 1])
        strr.append(hexmat[i, 2])

        if flag != camera:
            if debug:
                stdwrite('\n' + indent_level +
                         '>> HEXOS: load cam ' + str(camera))
                stdflush()
            h = HEXOSCamera(camera)
            flag = camera

            try:
                # download
                h.retrieveDataFromDB(
                    start, end, showProgressBar=False, adjustFirstPixel=False)

            except Exception:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                stdwrite('\n\t\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
                         '\n\t\t\\\ no HEXOS ' + str(camera) + ' data')
                stdflush()

                return (False, 0, 0)

            if i == 0:
                try:
                    # respective HEXOS time
                    time = h.getTimeVector()
                    hexos_time = (time - start) / 1.e9 + 60.5  # now in ms

                except Exception:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(
                        exc_tb.tb_frame.f_code.co_filename)[1]
                    stdwrite('\n\t\t\\\ ', exc_type, fname,
                             exc_tb.tb_lineno, '\n\t\t\\\ time vec failed')
                    stdflush()

            try:
                # get array with the spectra(time, pixels)
                spectra = h.get2DSpec()

            except Exception:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                stdwrite('\n\t\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
                         '\n\t\t\\\ get 2D spectra failed')
                stdflush()

            # pixels mapping
            wavel = sp.polyval(calib_poly[camera - 1], pixels) + offset

            # remove background from data
            tindex = np.where(
                (hexos_time > tbackground[0]) & (hexos_time < tbackground[1]))
            tindex = tindex[0]
            background = spectra[tindex, :].mean(0)
            spectr = spectra - background
            if debug:
                stdwrite("\n" + indent_level +
                         ">> HEXOS: loaded cam " + str(camera))
                stdflush()

        minindex = np.argmin(abs(wavel - wav))
        yplot = smooth((spectr[:, minindex]), 1)

        yplot = (yplot)

        if i == 0:
            hexint = np.empty(
                shape=[np.shape(yplot)[0], np.shape(hexmat)[0] + 1])

            for j in range(np.shape(hexos_time)[0]):
                hexint[j, i] = (hexos_time[j] - 61) * 1e3

        for j in range(np.shape(yplot)[0]):
            hexint[j, i + 1] = yplot[j]

    reducer = 1
    hexos_length = np.shape(yplot)[0]
    tmp['dimensions'] = time.tolist()
    for i, tag in enumerate(hexmat[:, 2]):
        tmp['values'][tag.replace(' ', '_')] = \
            [hexint[0:hexos_length:reducer, 0].tolist(),
             hexint[0:hexos_length:reducer, i + 1].tolist()]
        tmp['E_lit'][tag.replace(' ', '_')] = np.float(hexmat[i, 3])
        tmp['lambda_lit'][tag.replace(' ', '_')] = np.float(hexmat[i, 1])

    if saving:
        with open(file, 'w') as f:
            json.dump(tmp, f, indent=4, sort_keys=False)
        f.close()

        stdwrite("\n" + indent_level + ">> HEXOS: saved")
        stdflush()

    return(tmp)


def hexos_ratios(
        mat='C',
        program='20181010.032',
        keylist=['C_II', 'C_III', 'C_IV', 'C_V'],
        debug=False,
        saving=True):
    """ downloading HEXOS data and returning
    Args:
        mat (0, str, optional): meterial/element to look for
        program (1, int, optional): XP ID
        debug (2, bool, optional): debug printing
        saving (3, bool, optional): saving bool
    Returns:
        ...
    Notes:
        None
    """

    hexos_dat, _ = get_hexos_xics(
        date=int(program[0:8]), shot=int(program[9:]),
        mat=mat, saving=True, hexos=True, xics=False)

    if True:  # try:
        info, req = api.xpid_info(program=program)
        off = (info["programs"][0]["trigger"]["4"][0] -
               info['programs'][0]['trigger']['1'][0]) / (1e9) - 0.3

        split_list, int_list, labels = [], [], []
        if keylist is None:
            keylist = sorted(list(hexos_dat['values'].keys()))

        ratios = Z((
            3, len(keylist) - 1,
            np.shape(hexos_dat['values'][
                list(hexos_dat['values'].keys())[0]])[1]))

        for i, key in enumerate(keylist):
            split_list.append(key.replace('_', ' '). split())
            int_list.append(r2i(split_list[i][1]))

            if len(split_list[i]) == 3:
                int_list[i] += int(split_list[i][2][1]) * 2.  # 0.1

        keylist = [x for _, x in sorted(zip(int_list, keylist))]
        for i, key in enumerate(keylist[:-1]):

            max_i, max_ipo = np.max(hexos_dat['values'][key][1]), \
                np.max(hexos_dat['values'][keylist[i + 1]][1])
            smth = pf.smoothing(hexos_dat['values'][keylist[i + 1]][1], 100)

            ratios[0, i, :] = \
                [x * 1e-3 for x in hexos_dat['values'][keylist[i + 1]][0]]
            ratios[1, i, :] = \
                [(smth[j] / max_ipo) / (v / max_i)
                 if 0.2 < ratios[0, i, j] < off else 0.0 for j, v in enumerate(
                    pf.smoothing(hexos_dat['values'][key][1], 100))]
            ratios[2, i, :] = \
                [(hexos_dat['values'][keylist[i + 1]][1][j] * 1e-4) / (v * 1e-4)
                 if 0.2 < ratios[0, i, j] < off else 0.0 for j, v in enumerate(
                    hexos_dat['values'][key][1])]

            labels.append(keylist[i + 1].replace('_', ' ') +
                          '/' + key.replace('_', ' '))

    # except Exception:
    #     print('\n\t\t\\\ HEXOS ratios failed')

    return (ratios, labels, hexos_dat)


def get_xics(
        date=20181010,
        shot=32,
        path='../results/XICS/',
        saving=True,
        extended=True,
        indent_level='\t'):
    """ getting XICS data and returning
    Args:
        date (0, int, optional): XP date
        shot (1, int, optional): XP ID
        path (2, str, optional): writing path
        saving (3, bool, optional): saving bool
        extended (4, bool, optional): also get Te info
    Returns:
       tmp (0, dict): results with ion temp and stuff
    Notes:
        None.
    """
    expIDMDS = int(str(date - 20000000) + str(shot).zfill(3))

    try:
        MDSplus.setenv('qsw_eval_path', 'mds-data-1.ipp-hgw.mpg.de::')
        tree = MDSplus.Tree('qsw_eval', expIDMDS)
        ti = tree.getNode('XICS:TI').data()
        tdl = tree.getNode('XICS_LINE:TI').data()
        sigma_tdl = tree.getNode('XICS_LINE:TI:SIGMA').data()
        ti0 = tree.getNode('XICS_LINE:TI').data()
        sigma_ti0 = tree.getNode('XICS_LINE:TI0:SIGMA').data()
        er = tree.getNode('XICS:ER').data()
        sigma_er = tree.getNode('XICS:ER:SIGMA').data()
        sigma_ti = tree.getNode('XICS.TI:SIGMA').data()
        time_ti = tree.getNode('XICS.TI:TIME').data()
        reff_ti = tree.getNode('XICS.TI.REFF').data()
        rho_ti = tree.getNode('XICS.TI.RHO').data()

        if extended:
            te = tree.getNode('XICS:TE').data()
            sigma_te = tree.getNode('XICS.TE:SIGMA').data()
            time_te = tree.getNode('XICS.TE:TIME').data()
            reff_te = tree.getNode('XICS.TE.REFF').data()
            rho_te = tree.getNode('XICS.TE.RHO').data()

        stdwrite('\n' + indent_level + '>> XICS: loaded data')
        stdflush()

        tmp = {'label': 'XICS_data', 'values': {'ti': {
            'values': [{'ti': ti.tolist()},
                       {'ti_sigma': sigma_ti.tolist()},
                       {'ti_rho': rho_ti.tolist()},
                       {'ti_reff': reff_ti.tolist()},
                       {'ti_line': tdl.tolist()},
                       {'ti_line_sigma': sigma_tdl.tolist()},
                       {'ti0': ti0.tolist()},
                       {'ti0_sigma': sigma_ti0.tolist()},
                       {'e_r': er.tolist()},
                       {'e_r_sigma': sigma_er.tolist()}],
            'dimensions': time_ti.tolist()}},
            'xpid': str(date) + '.' + str(shot).zfill(3)}

        if extended:
            tmp['values']['te'] = \
                {'values': [{'te': te.tolist()},
                            {'te_sigma': sigma_te.tolist()},
                            {'te_reff': reff_te.tolist()},
                            {'te_rho': rho_te.tolist()}],
                 'dimensions': time_te.tolist()}

        if saving:
            with open(path + 'xics_' + str(date) + '.' +
                      str(shot).zfill(3) + '.json', 'w') as f:
                json.dump(tmp, f, indent=4, sort_keys=False)
            f.close()

            stdwrite('\n' + indent_level + '>> XICS: saved')
            stdflush()

        return(tmp)

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        stdwrite('\n\t\t\\\ ' + str(exc_type) + ' ' +
                 str(fname) + ' ' + str(exc_tb.tb_lineno) +
                 '\n\t\t\\\ XICS: no data')
        return(None)


def hexos_getInfo(
        material='C'):
    """ HEXOS line info for materials/elements
    Args:
        material (str, optional): element PSE string
    Returns:
        hexmat (ndarray): line and camera infor for ionisation stages
        calib_poly (ndarray): calibration polynome for power series
    Notes:
        None
    """
    # get info from line data file
    lambda_lit, E_lit, el, ion, transition, E_up = load_LineData_HEXOS()

    # set up finds for material
    hitList = np.where(el == material)[0]
    hexmat = [0] * len(hitList)

    for i, x in enumerate(hitList):
        L = lambda_lit[x]
        # set camera given by lambda
        if .0 < L < 11.:
            c = 1
        if 11. < L < 20.:
            c = 2
        if 20. < L < 60.:
            c = 3
        if 60 < L:
            c = 4
        # ionisation level and Energy
        hexmat[i] = [c, L, ion[x], E_lit[x]]

    # Wavelength calibration
    calib_poly = [
        [-5.26408526e-10,
         3.00904877e-06,
         5.68674427e-03,
         2.56606993e+00],
        [7.3342033460e-10,
         5.5963113791e-07,
         -1.6271674542e-02,
         2.4477668611e+01],
        [-1.7557197479e-09,
         6.8180868259e-06,
         -5.0960088911e-02,
         6.6711693376e+01],
        [-3.0264408949e-09,
         6.8935650893e-06,
         9.4944027852e-02,
         5.9872881940e+01]]

    return np.array(hexmat), np.array(calib_poly)


def return_mean_profile(
        input_data=Z((40, 1000)),
        threshUp=5.,
        threshLow=.1,
        indent_level='\t'):

    try:
        k, mean_profile = 0, Z((np.shape(input_data)[0]))

        for i, vec in enumerate(np.transpose(input_data)):
            if ((np.max(vec) <= threshUp) and (np.min(vec) >= threshLow)):
                k, mean_profile = k + 1, mean_profile + vec
        mean_profile = mean_profile / k

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  mean profile failed')

    return mean_profile
