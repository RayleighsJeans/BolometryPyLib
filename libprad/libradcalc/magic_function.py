""" ***************************************************************************
    header """

import os
import warnings
import sys

import uncertainties as unc
import uncertainties.umath as umath

import pickle
import numpy as np
import scipy as sc


warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")

Z = np.zeros
ones = np.ones

""" eo header
************************************************************************** """


def magic_function(
        voltage=Z(10000),  # in V
        dt=1.6e-3,  # in s
        U0=5.0,  # in V
        kappa_ch=8e-4,  # in A^2
        r_ch=1e3,  # in Ohm
        tau_ch=1e-1,  # in s
        rc=1e3,  # in Ohm
        f_bridge=2500.,
        c_cab=5e-9,
        filter_method='raw',
        spike_method='derivative'):
    """ Calculates the power of a given channel signal with the electrical
        properties at hand. Also looks for digital caching fixes
    Args:
        priority_object (0, list): list of predefined data needed for calc
        channel_volt (1, list): Signal vector of given channel
        time (2, list): Time vector
        U0 (3, npfloat): Excitation voltage of bridge
        kappa_ch (4, npfloat): Channels heat capacity
        r_ch (5, npfloat): Channels electrical resistance
        tau_ch (6, npfloat): ...
    Returns:
        power_chn (0, list): Power vector of given channel signal.
    NOTES: None. """
    try:  # catch fails
        # quick digital caching fix with threshold and width
        [spikes, thresh, WW, mL] = [0, 2.e-1, 10, len(voltage) - 1]

        if spike_method is not None:
            # on scnd deriv find max, look for voltage
            secnd_deriv = np.abs(np.gradient(np.gradient(voltage))).tolist()
            hit_ud_val = np.abs(voltage[secnd_deriv.index(max(secnd_deriv))])
            pos = secnd_deriv.index(max(secnd_deriv))

            if spike_method == 'linear':
                if hit_ud_val >= thresh:  # is there hit?
                    n = 0
                    while (n <= mL):  # inside length
                        if (voltage[n] >= thresh):
                            L, R = n, n
                            while (voltage[R] >= thresh) and (R < mL):
                                R += 1
                            filler = np.linspace(voltage[L],
                                                 voltage[R], R - L + 1)
                            for k in range(L, R):
                                voltage[k] = filler[k - L]
                            n = R - 1
                            spikes += R - L
                        n += 1

            elif spike_method == 'derivative':
                # if spike bigger than thresh, shift points over
                while (hit_ud_val >= thresh):  # and (spikes <= 10):
                    # as long as spikes
                    spikes += 1
                    if (pos in range(mL - WW, mL)):  # on right edge
                        voltage[mL - WW:mL] = \
                            voltage[mL - (2 * WW + 1):mL - (WW + 1)]
                    elif (pos in range(0, WW)):  # on left edge
                        voltage[0:WW] = voltage[WW + 1:2 * WW + 1]
                    else:  # somewhere in between
                        voltage[pos - WW:pos] = \
                            voltage[pos - (2 * WW + 1):pos - (WW + 1)]
                        voltage[pos:pos + WW] = \
                            voltage[pos + WW:pos + (2 * WW)]

                    # re-evaluate the data for next iteration
                    secnd_deriv =  \
                        np.abs(np.gradient(np.gradient(voltage))).tolist()
                    hit_ud_val = np.abs(voltage[secnd_deriv.index(
                        max(secnd_deriv))])
                    pos = secnd_deriv.index(max(secnd_deriv))

        N = 51
        # savitzky golay filtering of second order with halfwidth N
        if (filter_method == 'savgol'):
            Ud_filt = sc.signal.savgol_filter(voltage, N, 2)  # in V
            # Ud_filt = channel_volt
            dUdt_filt = np.gradient(Ud_filt, dt)  # in V/s

        # moving average of width N and gradient
        elif (filter_method == 'mean'):
            Ud_filt = np.convolve(  # in V
                voltage, np.ones((N,)) / N, mode='same')
            dUdt_filt = np.gradient(Ud_filt, dt)  # in V/s

        # moving average of width N (convolution with 1)
        elif (filter_method == 'convolve'):
            Ud_filt = np.convolve(  # in V
                voltage, np.ones((N,)) / N, mode='same')
            dUdt_filt = sc.signal.savgol_filter(Ud_filt, N, 2)  # in V/s
            dUdt_filt = np.convolve(dUdt_filt, np.ones((N,)) / N,  # in V/s
                                    mode='same')

        # nothing, raw
        else:  # elif (filter_method == 'raw'):
            Ud_filt = voltage  # in V
            dUdt_filt = np.gradient(Ud_filt, dt)  # in V/s

        # subparts of the equations calculated seperately
        U_eff = U0 * r_ch / (r_ch + 2.0 * rc)  # in V
        omega = 2.0 * np.pi * f_bridge * c_cab  # F * Hz (Siemens)
        rt_gc = np.sqrt(1.0 + (omega * (r_ch + rc))**2)  # unitless
        beta = (1.0 - (omega * r_ch)**2 + (omega * rc)**2) / \
            (1.0 + (omega * (r_ch + rc))**2)  # unitless
        tau_fact = 1.0 - U_eff**2 * beta / 4.0 / kappa_ch / \
            (r_ch + rc)**2  # unitless

        # finally power
        power = \
            2.0 / U_eff * (r_ch + 2.0 * rc) * kappa_ch * \
            rt_gc * (tau_ch * dUdt_filt + tau_fact * Ud_filt)
        # 1/ volt * (ohm * ampere^2 * (second * volt / second + volt) -> Watts

    except Exception:  # cath fails and debug
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print('\t\t\t\t\\\ ', exc_type, fname, exc_tb.tb_lineno,
              '\n\t\t\t\t\\\  Error in magic function')

    # return results and spike numbers found
    return power, spikes  # watts


def gaussian_error_propagation(
        Ud=0.4e-3,  # V
        dUd=0.5e-6,  # V, stddiv from exp
        dUdt=0.4e-3,  # V/s, from 0.0mV to 0.2mV in 0.5s
        ddUdt=1e-6,  # V/s, stddiv over 0.5s
        dt=0.0015999,  # s, ~1.6ms
        ddt=0.015e-3,  # s, timing error
        U0=5.0,  # V
        dU0=0.001,  # V
        rc=41.0,  # Ohm
        drc=0.001,  # Ohm
        f_bridge=625.,  # Hertz
        df_bridge=1.0,  # Hertz
        c_cab=2.e-9,  # F
        dc_cab=5.e-10,  # F
        Imax=4.7e-3,  # A
        dImax=1.e-6,  # A
        I01=4.21e-5,  # A
        dI01=1.e-6,  # A
        I02=4.61e-3,  # A
        dI02=1.e-6,  # A
        T=-9.0e0,  # s
        dT=0.0004,  # s
        Rload=10.,  # Ohm
        dRload=0.001,  # Ohm
        Rcable=40.,  # Ohm
        dRcable=0.001):  # Ohm

    U0 = unc.ufloat(U0, dU0)
    rc = unc.ufloat(rc, drc)
    f_bridge = unc.ufloat(f_bridge, df_bridge)
    c_cab = unc.ufloat(c_cab, dc_cab)
    Imax = unc.ufloat(Imax, dImax)
    I01 = unc.ufloat(I01, dI01)
    I02 = unc.ufloat(I02, dI02)
    T = unc.ufloat(T, dT)
    Rload = unc.ufloat(Rload, dRload)
    Rcable = unc.ufloat(Rcable, dRcable)

    dt = unc.ufloat(dt, ddt)
    Ud = unc.ufloat(Ud, dUd)
    dUdt = unc.ufloat(dUdt, ddUdt)

    RB1 = 2. * (((2.5) / (Imax - I01)) - (Rload + Rcable))
    tau_ch = 1. / (T) * (-1.)
    kappa_ch = RB1 * ((((I02 - I01)**4) / 4.) / 2.5) / (Imax - I02)
    r_ch = (RB1 + umath.sqrt(RB1**2 + 5.**2 / kappa_ch)) / 2.

    U_eff = U0 * r_ch / (r_ch + 2.0 * rc)
    omega = 2.0 * np.pi * f_bridge * c_cab
    rt_gc = umath.sqrt(1.0 + (omega * (r_ch + rc))**2)
    beta = (1.0 - (omega * r_ch)**2 + (omega * rc)**2) / \
        (1.0 + (omega * (r_ch + rc))**2)
    tau_fact = 1.0 - U_eff**2 * beta / 4.0 / kappa_ch / \
        (r_ch + rc)**2

    power = \
        2.0 / U_eff * (r_ch + 2.0 * rc) * kappa_ch * \
        rt_gc * (tau_ch * dUdt + tau_fact * Ud)

    error = []
    error.append(power.derivatives[U0] * dU0)
    error.append(power.derivatives[rc] * drc)
    error.append(power.derivatives[f_bridge] * df_bridge)
    error.append(power.derivatives[c_cab] * dc_cab)
    error.append(power.derivatives[Imax] * dImax)
    error.append(power.derivatives[I01] * dI01)
    error.append(power.derivatives[I02] * dI02)
    error.append(power.derivatives[T] * dT)
    error.append(power.derivatives[Rload] * dRload)
    error.append(power.derivatives[Rcable] * dRcable)
    dpower = np.sum(error)

    D = {'label': 'systematic error calculation with uncertainties',
         'values': {
             'variables': {},
             'error_propagation': {},
             'power': power,
             'dpower': dpower}
         }

    D['values']['variables']['U0'] = U0
    D['values']['variables']['rc'] = rc
    D['values']['variables']['f_bridge'] = f_bridge
    D['values']['variables']['c_cab'] = c_cab
    D['values']['variables']['Imax'] = Imax
    D['values']['variables']['I01'] = I01
    D['values']['variables']['I02'] = I02
    D['values']['variables']['T'] = T
    D['values']['variables']['Rload'] = Rload
    D['values']['variables']['Rcable'] = Rcable
    D['values']['variables']['dt'] = dt
    D['values']['variables']['Ud'] = Ud
    D['values']['variables']['dUdt'] = dUdt
    D['values']['variables']['RB1'] = RB1
    D['values']['variables']['tau_ch'] = tau_ch
    D['values']['variables']['kappa_ch'] = kappa_ch
    D['values']['variables']['r_ch'] = r_ch
    D['values']['variables']['U_eff'] = U_eff
    D['values']['variables']['omega'] = omega
    D['values']['variables']['rt_gc'] = rt_gc
    D['values']['variables']['beta'] = beta
    D['values']['variables']['tau_fact'] = tau_fact

    D['values']['error_propagation']['U0'] = error[0]
    D['values']['error_propagation']['rc'] = error[1]
    D['values']['error_propagation']['f_bridge'] = error[2]
    D['values']['error_propagation']['c_cab'] = error[3]
    D['values']['error_propagation']['Imax'] = error[4]
    D['values']['error_propagation']['I01'] = error[5]
    D['values']['error_propagation']['I02'] = error[6]
    D['values']['error_propagation']['T'] = error[7]
    D['values']['error_propagation']['Rload'] = error[8]
    D['values']['error_propagation']['Rcable'] = error[9]

    dump = '../files/gaussian_error_propagation.pkl'
    with open(dump, 'wb') as outfile:
        pickle.dump(D, outfile, pickle.HIGHEST_PROTOCOL)
    outfile.close()

    print('>> At U_d=', format(Ud.nominal_value, '.3e'), 'V',
          'the power P=', format(power.nominal_value, '.3e'), 'W',
          '; s=', format(power.s, '.3e'), 'W',
          ' dpower=', format(dpower, '.3e'), 'W')

    return (D, error, power, dpower)
