""" **************************************************************************
    so HEADER """

import os
import sys
import numpy as np
import scipy as sc
import csv
import plot_funcs as plot_funcs
import logbook_api as logbook_api
from pandas import *
import warnings

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")

ndarray = np.zeros((1)).__class__
stdwrite = sys.stdout.write
stdflush = sys.stdout.flush

""" eo header
************************************************************************** """


def dens_diag(
        priority_object,
        data_object,
        radpower_object,
        logbook_data,
        date,
        shotno,
        program,
        program_info,
        base,
        indent_level):

    print(indent_level + '>> looking for crit properties')
    try:
        location = base + r'/../results/COMPARISON/POWER_DIAGNOSTIC/' + date
        if not os.path.exists(location):
            os.makedirs(location)

        t1 = priority_object[36]  # T1 of CODAC where ECRH on
        prad_vbc = [v * 1e-6 for v in radpower_object[2]]  # vbc in MW
        prad_hbc = [h * 1e-6 for h in radpower_object[1]]  # hbc in MW

        pecrh = [e * 1e-3 for e in data_object[7]['values']]  # ECRH in MW
        wdia = [w * 1e-3 for w in data_object[11]['values']]  # wdia in MJ
        nel = [n * 1e-19 for n in data_object[10]['values']]  # ne in 1e19 m^-2
        Te_out = [T for T in data_object[9]['values']]  # in keV
        Te_in = [T for T in data_object[8]['values']]  # in keV

        time_prad = radpower_object[0]  # normalized time vector of diags
        time_ecrh = [(t - t1) * 1e-9 for t in data_object[7]['dimensions']]
        time_wdia = [(t - t1) * 1e-9 for t in data_object[11]['dimensions']]
        time_nel = [(t - t1) * 1e-9 for t in data_object[10]['dimensions']]
        time_Te = [(t - t1) * 1e-9 for t in data_object[8]['dimensions']]

        datas = [prad_hbc, prad_vbc, pecrh, wdia,
                 nel, Te_out, Te_in]
        times = [time_prad, time_prad, time_ecrh,
                 time_wdia, time_nel, time_Te, time_Te]

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  dens diag setup failed')
        return

    try:
        [L, nBIG] = [0, 0]
        for n in range(0, len(times)):
            if (L <= len(times[n])):
                [L, nBIG] = [len(times[n]), n]
        lin = np.linspace(0, L - 1, L)

        for n in range(0, len(times)):
            if (n is not nBIG):
                default = np.linspace(0, len(datas[n]) - 1, len(datas[n]))
                lin = np.linspace(0, len(datas[n]) - 1, L)
                foo = np.interp(lin, default, datas[n])
                bar = np.interp(lin, default, times[n])
                [datas[n], times[n]] = [foo, bar]

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  dens diag interpolation failed')
        return

    try:
        # max Vp_rad
        mx_Vprad = max(datas[1][100:-100])
        if isinstance(datas[1], ndarray):
            pos_Vprad = np.where(datas[1] == mx_Vprad)[0]
        elif isinstance(datas[1], list):
            pos_Vprad = datas[1].index(mx_Vprad)

        # max Hp_rad
        mx_Hprad = max(datas[0][100:-100])
        if isinstance(datas[0], ndarray):
            pos_Hprad = np.where(datas[0] == mx_Hprad)[0]
        elif isinstance(datas[0], list):
            pos_Hprad = datas[0].index(mx_Hprad)

        # max pecrh
        mx_ecrh = max(datas[2])
        if isinstance(datas[2], ndarray):
            pos_ecrh = np.where(datas[2] == mx_ecrh)[0]
        elif isinstance(datas[2], list):
            pos_ecrh = datas[2].index(mx_ecrh)

        # max wdia
        mx_wdia = max(datas[3])
        if isinstance(datas[3], ndarray):
            pos_wdia = np.where(datas[3] == mx_wdia)[0]
        elif isinstance(datas[3], list):
            pos_wdia = datas[3].index(mx_wdia)

        # max nel
        mx_nel = max(datas[4])
        if isinstance(datas[4], ndarray):
            pos_nel = np.where(datas[4] == mx_nel)[0]
        elif isinstance(datas[4], list):
            pos_nel = datas[4].index(mx_nel)

        # max Te outside
        mx_Te_out = max(datas[5])
        if isinstance(datas[5], ndarray):
            pos_Te_out = np.where(datas[5] == mx_Te_out)[0]
        elif isinstance(datas[5], list):
            pos_Te_out = datas[5].index(mx_Te_out)

        # max Te core
        mx_Te_in = max(datas[6])
        if isinstance(datas[6], ndarray):
            pos_Te_in = np.where(datas[6] == mx_Te_in)[0]
        elif isinstance(datas[6], list):
            pos_Te_in = datas[6].index(mx_Te_in)

        mx_vals = [mx_Hprad, mx_Vprad, mx_ecrh,
                   mx_wdia, mx_nel, mx_Te_out, mx_Te_in]
        pos_mx = [pos_Hprad, pos_Vprad, pos_ecrh,
                  pos_wdia, pos_nel, pos_Te_out, pos_Te_in]
        mx_times = [.0] * len(mx_vals)

        # make list of
        for n in range(0, len(mx_vals)):
            try:
                if (((len(pos_mx[n]) > 1) and
                     (len(pos_mx[n]) > len(times[n]) / 2))):
                    pos_mx[n] = pos_mx[n - 1] + 200

                elif ((len(pos_mx[n]) > 1) and
                      (len(pos_mx[n]) < len(times[n]) / 2)):
                    pos_mx[n] = pos_mx[n][round(len(pos_mx[n] / 2))]

                elif (len(pos_mx[n]) < 1):
                    foo = [t - max(mx_times) for t in times[n]]
                    pos_mx[n] = foo.index(min(foo)) + 200
                mx_times[n] = times[n][pos_mx[n]]

            except Exception:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                if False:
                    print(indent_level + '\t\\\ ', exc_type, fname,
                          exc_tb.tb_lineno)

        groupMXs = np.zeros((len(mx_vals), len(mx_vals)))
        # values at max of different properties
        for l in range(0, len(datas)):
            for n in range(0, len(datas)):
                if (n is not l):

                    # array'ize and find min
                    [times[n], datas[n]] = \
                        [np.array(times[n]), np.array(datas[n])]
                    newpos = np.where(
                        np.abs(times[n] - mx_times[l]) ==
                        min(np.abs(times[n] - mx_times[l])))[0]

                    try:  # sometimes returns multiple values
                        if (len(newpos) >= 1):
                            newpos = newpos[0]
                    except Exception:
                        pass

                    # get val
                    groupMXs[l, n] = datas[n][newpos]
                else:
                    # already cached that
                    groupMXs[l, n] = mx_vals[l]

        if True:
            cols = ['at Prad HBC', 'at Prad VBC', 'at ECRH',
                    'at Wdia', 'at ne', 'at Te out', 'at Te in']
            idx = ['val Prad HBC', 'val Prad VBC', 'val ECRH',
                   'val Wdia', 'val ne', 'val Te out', 'val Te in']

            print(
                indent_level +
                '\t>> DataFrame returned searching for crit diags',
                '\n' + '* ' * 56 + '\n', DataFrame(
                    groupMXs, columns=cols, index=idx), '\n' + '* ' * 56)

        pre = '\t'
        print("* * * * Searching diagnostic: " +
              "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" +
              " * * * * * * * * * * * \n" +
              pre + "       max P_rad(VBC) =",
              mx_vals[1], 'MW        @ t =', mx_times[1], 's\n' +
              pre + "       max P_ECRH     =",
              mx_vals[2], 'MW        @ t =', mx_times[2], 's\n' +
              pre + "       max W_dia      =",
              mx_vals[3], 'MJ        @ t =', mx_times[3], 's\n' +
              pre + "       max n_e        =",
              mx_vals[4], '1e19 m^-3 @ t =', mx_times[4], 's\n' +
              pre + "       max T_e out    =",
              mx_vals[5], 'keV       @ t =', mx_times[5], 's\n' +
              pre + "       max T_e in     =",
              mx_vals[6], 'keV       @ t =', mx_times[6], 's\n' +
              '* ' * 56)

        dens_diag_dump(datas, times, mx_vals, pos_mx, groupMXs,
                       priority_object, data_object, radpower_object,
                       logbook_data, date, shotno, program, program_info,
                       base, indent_level)

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  dens diag interpolation failed')

    return


def power_balance(
        priority_object,
        ecrh_off,
        datas,
        times,
        groupMXs,
        pos_mx,
        base,
        date,
        shotno,
        program,
        program_info,
        indent_level):

    print(indent_level + '>> power balance calculation')
    try:
        [datas, times] = [np.array(datas), np.array(times)]
        [lengthMx, it] = [0, 0]
        rngT = np.zeros((len(times), len(times)))

        # find common spots at 0.0s and T4 trigger
        T4 = (priority_object[37] - priority_object[36]) / 1e9

        for n in range(0, len(times)):
            rngT[n, 0] = np.where(
                np.abs(times[n]) == min(np.abs(times[n])))[0] - 2000
            rngT[n, 1] = np.where(
                np.abs(times[n] - T4) == min(np.abs(times[n] - T4)))[0]

            if (rngT[n, 1] - rngT[n, 0] >= lengthMx):
                lengthMx = rngT[n, 1] - rngT[n, 0]
                it = n

        # create time-matching data via interpolation and time
        BSTime = times[it, rngT[it, 0]:rngT[it, 1]]
        interpDatas = np.zeros((len(times), lengthMx))

        for n in range(0, len(times)):
            if n is not it:
                default = np.linspace(0, rngT[n, 1] - rngT[n, 0] - 1,
                                      rngT[n, 1] - rngT[n, 0])
                lin = np.linspace(0, lengthMx - 1, lengthMx)
                interpDatas[n] = np.interp(
                    lin, default, datas[n, rngT[n, 0]:rngT[n, 1]])
            else:
                interpDatas[n] = datas[n][rngT[n, 0]:rngT[n, 1]]

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  power balance interpol. failed')
        return

    # calculate stuff
    try:
        pbalance = np.zeros((lengthMx))
        pbalance = interpDatas[2] - interpDatas[1] - \
            sc.signal.savgol_filter(np.gradient(
                interpDatas[3], BSTime[1] - BSTime[0]), 51, 2)

        plot_funcs.balance_plot(interpDatas, pbalance, BSTime, ecrh_off,
                                date, shotno, base, indent_level)

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level +
              '\t\\\  power balance calculation. failed')

    return


def dens_diag_dump(
        datas,
        times,
        mx_vals,
        pos_mx,
        groupMxs,
        priority_object,
        data_object,
        radpower_object,
        logbook_data,
        date,
        shotno,
        program,
        program_info,
        base,
        indent_level):

    print(indent_level + ">> Writing power/crit dens info to file ...")
    try:
        # get info shot and put through for
        # power and sudo scaling diag
        infoo_shot = logbook_api.get_infoo_shot(int(shotno), logbook_data,
                                                indent_level)

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  logbook info failed')
        return

    location = base + "/../results/COMPARISON/POWER_DIAGNOSTIC/" + date
    if not os.path.exists(location):
            os.makedirs(location)
    try:
        data_list = \
            [date,                              # 0 date
             int(shotno),                       # 1 XP ID
             infoo_shot[3],                     # 2 magconf
             infoo_shot[2],                     # 3 ecrh_duration
             infoo_shot[0],                     # 4 ecrh_power
             infoo_shot[1],                     # 5 energy
             infoo_shot[4][0],                  # 7 gas1 type
             infoo_shot[4][1],                  # 8 gas1 approach
             infoo_shot[5][0],                  # 9 gas2 type
             infoo_shot[5][1],                  # 10 gas2 approach
             infoo_shot[6][0],                  # 11 gas3 type
             infoo_shot[6][1],                  # 12 gas3 approach
             infoo_shot[7][1],                  # 13 pellet desc
             mx_vals[0],                        # 14 HBC max Prad
             mx_vals[1],                        # 15 VBC max Prad
             mx_vals[2],                        # 16 max ECRH
             mx_vals[3],                        # 17 max Wdia
             mx_vals[4],                        # 18 max Eldens
             mx_vals[5],                        # 19 max Te out
             mx_vals[6],                        # 20 max Te core
             groupMxs[0, 2],                    # 21 ECRH at max Prad
             groupMxs[0, 3],                    # Wdia at max Prad
             groupMxs[0, 4],                    # ne at max Prad
             groupMxs[0, 5],                    # Te out at max Prad
             groupMxs[0, 6]]                    # Te core at max Prad

        data_file = base + r'/../results/COMPARISON/' + \
            'POWER_DIAGNOSTIC/power_diagnostic.dat'
        with open(data_file, 'a') as file:
                writer = csv.writer(file, delimiter='\t', lineterminator='\n')
                writer.writerow(data_list)
        file.close()
        print(indent_level + ">> Done writing...")

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(indent_level + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n' + indent_level + '\t\\\  diag write failed')

    return
