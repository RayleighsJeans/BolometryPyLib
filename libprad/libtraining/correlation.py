""" ********************************************************************** """

import sys
import math
import os
import ast
import numpy as np
import scipy.signal as sig

import mClass
import prad_calculation as prad_calc
import training_plot as train_plot

Z = np.zeros
stdwrite = sys.stdout.write
stdflush = sys.stdout.flush

""" ********************************************************************** """


def correlate(
        cam='HBC',
        method='weighted_deviation',
        program='20181010.032',
        prad={'none': None},
        channels={'none': None},
        prio={'none': None},
        sel=[[0, 15, 30], [1, 14, 29], [2, 16, 28]],
        saving=True,
        plot=False):
    """ Do correlation calculations for the full camera prad
        and the newly approximated radiated power loss
    Args:
        program (str, optional):
        prad (array, optional): Radiation power loss from archive
        channels (array, optional): All channels power time traces
        priority_object (list, optional): Predefined list of constants
        sel (list, optional):
        camera (str, optional):
        method (str, optional):
        plot (bool, optional):
        redo_check (bool, optional):
        example_mode (bool, optional):
    Returns:
        None
    Notes:
        None
    """
    broken_channels = prio['geometry']['channels']['droplist']
    # combinations
    sel_id = len(sel)
    results = Z((sel_id,))
    freqs = Z((500))

    # need stop
    t_1 = (prio['t0'] + 1e8)  # - priority_object[36]) / (1e9) + 1.0
    t_2 = (prio['t4'] - 1e8)  # - 0.1
    ix_s = mClass.find_nearest(np.array(channels['dimensions']), t_1)[0]
    ix_t = mClass.find_nearest(np.array(channels['dimensions']), t_2)[0]

    loc = loc = '../results/COMBINATIONS/' + program + '/' + method + '/'
    try:
        loc = loc + str(len(sel[0])) + '_' + cam + '/'
        str_sel = str(len(sel[0]))
    except TypeError:
        loc = loc + '1_' + cam + '/'
        str_sel = '1'

    if not os.path.exists(loc):
        os.makedirs(loc)

    if saving:
        if (os.path.isfile(loc + method + '.dat')):
            print('\t\t>> ' + method + ' ' + cam + ' ' + str_sel + ' exists')
            # load file
            results = [None, None]
            # specified path for this sensitivity analysis, loading
            path = loc + method + '.dat'
            results[0] = [ast.literal_eval(x) for x in np.genfromtxt(
                path, delimiter='\t', dtype=str, usecols=[0])]  # combinations
            results[1] = np.loadtxt(  # each combinations quality
                path, delimiter='\t', dtype=np.float, usecols=[1])
        return (results)  # already done

    alpha, phi, beta = default_correlation(
        arr1=prad['values'], arr2=prad['values'],
        id_1=ix_s, id_2=ix_t, time_raw=channels['dimensions'])

    N_combination, first_combination = len(sel), int(str_sel)

    sys.stdout.write('\n')
    sys.stdout.flush()
    for j, ch in enumerate(sel):  # combination in list
        sys.stdout.write(
            '\r\t\tC:' + str(first_combination) + ' ' + cam + ' ' +
            '[%-25s] %d%%' % ('=' * math.ceil((j / N_combination) * 25),
                              (j / N_combination) * 100))
        sys.stdout.flush()

        if isinstance(ch, int):
            if ch not in broken_channels:  # skip broken channels
                selection = [ch]
            else:
                continue
        else:
            selection = ch

        calculation, volume_sum = prad_calc.calculate_prad(
            power=np.array(channels['values'], dtype=np.float),
            volume=prio['geometry']['geometry']['vbolo'],
            k_bolo=prio['geometry']['geometry']['kbolo'],
            volume_torus=prio['volume_torus'],
            channels=selection, camera_list=selection,
            date=int(program[0:8]), shotno=int(program[-3:]),
            brk_chan=broken_channels, debug=False)

        e_val, trace, freqs, pref, lab, ylab = evaluation(
            method=method, arr1=np.array(prad['values']), arr2=calculation,
            id_1=ix_s, id_2=ix_t, time_raw=channels['dimensions'],
            freqs=freqs, plot=plot)

        # calculate for all the methods used above
        results[j] = e_val / alpha if method is not \
            'mean_deviation' else e_val

    if saving:
        path = loc + method + '.dat'
        with open(path, 'w') as file:
            for k, curr_s in enumerate(sel):
                if not isinstance(sel[k], (int, list)):
                    file.write('{}\t{}\n'.format(curr_s.tolist(), results[k]))
                else:
                    file.write('{}\t{}\n'.format(curr_s, results[k]))
        file.close()

    if plot:
        train_plot.correlation_comb_plot(
            method=method, cell=ch,
            arr1=np.array(prad['values'], dtype=np.float),
            arr2=calculation, trace=trace,
            phi=phi, freqs=freqs, beta=beta, alpha=alpha,
            theta=results[j], time=np.array(prio['time']),
            id_1=ix_s, id_2=ix_t, pref=pref, lab=lab,
            ylab=ylab, loc=loc)

        train_plot.correlation_results_plot(
            ylab=ylab, lab=lab, str_sel=str_sel,
            results=results, loc=loc, pref=pref)
    return ([sel, results])


def evaluation(
        method='weighted_deviation',
        arr1=Z((10000)),
        arr2=Z((10000)),
        id_1=1000,
        id_2=9000,
        time_raw=Z((10000)),
        freqs=Z((500)),
        plot=False):

    # METHODS
    # standard deviation likeness
    if method == 'weighted_deviation':
        pref, lab = 'wghtd_dev', 'weighted deviation'
        values, trace = weighted_deviation_func(
            arr1=arr1, arr2=arr2, id_1=id_1, id_2=id_2)
        ylab = 'weighted deviation'

    # mean deviation from Daihong
    elif method == 'mean_deviation':
        pref, lab = 'mean_dev', 'mean deviation'
        values, trace = daz_mean_deviation(
            arr1=arr1, arr2=arr2, id_1=id_1, id_2=id_2)
        ylab = 'weighted deviation'

    # cross correlation
    elif method == 'correlation':
        pref, lab = 'cross_corr', 'cross correlation'
        values, trace = correlation_function(
            arr1=arr1, arr2=arr2, id_1=id_1, id_2=id_2,
            time_trace=time_raw)
        ylab = 'cross corr. [a.u.]'

    # own take at cross correlation, mathmatical way
    elif method == 'self_correlation':
        pref, lab = 'self_cross_corr', 'cross correlation'
        values, trace = self_correlation_function(
            arr1=arr1, arr2=arr2, id_1=id_1, id_2=id_2,
            time_trace=time_raw)
        ylab = 'cross corr. [a.u.]'

    # ALL THE FFT METHODS
    elif 'fft' in method:
        ylab = 'none'

        if method == 'fft_and_convolve':
            pref, lab = 'fft_and_convolve', 'FFT and convol.'
            values, trace, freqs = fft_and_convolve(
                arr1=arr1, arr2=arr2, id_1=id_1, id_2=id_2,
                time_trace=time_raw)[0:3]

        elif method == 'fft_solo':
            pref, lab = 'fft_solo', 'int. FFT'
            values, trace, freqs = fft_solo(
                arr1=arr1, arr2=arr2, id_1=id_1, id_2=id_2,
                time_trace=time_raw)[0:3]

        elif method == 'fftconvolve_integrated':
            pref, lab = 'fftconvolve', 'FFT and convol.'
            values, trace, freqs = fftconvolve_integrated(
                arr1=arr1, arr2=arr2, id_1=id_1, id_2=id_2,
                time_trace=time_raw)[0:3]

        elif method == 'coherence_fft':
            pref, lab = 'coherence', 'coherence'
            values, trace, freqs = coherence_spectrum_func(
                arr1=arr1, arr2=arr2, id_1=id_1, id_2=id_2,
                nperseg=5000, frequency=1250)

    return (values, trace, freqs, pref, lab, ylab)


def default_correlation(
        method='weighted_deviation',
        arr1=Z((10000)),
        arr2=Z((10000)),
        id_1=1000,
        id_2=9000,
        time_raw=Z((10000))):
    """ return default values for correlation
    Args:
        method (str, optional): analisation method
        arr1 (TYPE, optional): Array 1
        arr2 (TYPE, optional): Array 2 =? Array 1
        id_1 (int, optional): start index
        id_2 (int, optional): end index
        time_raw (TYPE, optional): if for frequency analysis time is needed
    Returns:
        alpha ()
        phi ()
        beta ()
    """
    # Master norm values
    if method == 'coherence_fft':
        beta = 1.
        alpha, phi, freqs = coherence_spectrum_func(
            arr1=arr1, arr2=arr2, id_1=id_1, id_2=id_2,
            nperseg=5000, frequency=1250)

    elif method == 'weighted_deviation':
        beta = 1.
        alpha, phi = weighted_deviation_func(
            arr1=arr1, arr2=arr2, id_1=id_1, id_2=id_2)

    elif method == 'mean_deviation':
        beta = 1.
        alpha, phi = daz_mean_deviation(
            arr1=arr1, arr2=arr2, id_1=id_1, id_2=id_2)

    elif method == 'correlation':
        beta = 1.
        alpha, phi = correlation_function(
            arr1=arr1, arr2=arr2, id_1=id_1, id_2=id_2, time_trace=time_raw)

    elif method == 'self_correlation':
        beta = 1.
        alpha, phi = self_correlation_function(
            arr1=arr1, arr2=arr2, id_1=id_1, id_2=id_2, time_trace=time_raw)

    elif method == 'fft_and_convolve':
        beta = 1.e13
        alpha, phi = fft_and_convolve(
            arr1=arr1, arr2=arr2, id_1=id_1, id_2=id_2,
            time_trace=time_raw)[0:2]

    elif method == 'fft_solo':
        beta = 1.e20
        alpha, phi = fft_solo(
            arr1=arr1, arr2=arr2, id_1=id_1, id_2=id_2,
            time_trace=time_raw)[0:2]

    elif method == 'fftconvolve_integrated':
        beta = 1.e20
        alpha, phi, = fftconvolve_integrated(
            arr1=arr1, arr2=arr2, id_1=id_1, id_2=id_2,
            time_trace=time_raw)[0:2]

    return (alpha, phi, beta)


def weighted_deviation_func(
        arr1=Z((10000)),
        arr2=Z((10000)),
        id_1=1000,
        id_2=9000):
    """ Function defining the weighted deviation application
    Args:
        arr1 (ndarray, optional): Array like input
        arr2 (ndarray, optional): Same size array like input
        id_1 (int, optional): Starting index of eval
        id_2 (int, optional): End index of eval
    Returns:
        eval_val (0, np.float): Result of weighted deviation function
        eval_list (1, list): Weighted deviation between arr1 & arr2
    ToDos:
        None
    Notes:
        None
    """
    eval_list = [
        1 - c / d if (d > c) else 0.0 for d, c in
        zip(arr1[id_1:id_2],
            np.abs([u - v for u, v in zip(arr1, arr2)])[id_1:id_2])]
    eval_val = np.mean(eval_list)
    return eval_val, np.array(eval_list)


def correlation_function(
        arr1=Z((10000)),
        arr2=Z((10000)),
        id_1=1000,
        id_2=9000,
        time_trace=np.linspace(1.e4, 2.e4, 10000)):
    """ Cross-correlation function with norm
    Args:
        arr1 (array, optional): Array like data 1
        arr2 (array, optional): Array like data 2
        id_1 (int, optional): Eval start index
        id_2 (int, optional): Eval end index
        time_trace (array, optional): Time vector regarding data
    Returns:
        eval_val (0, np.float): Cross correlation built-in function
    Notes:
        None
    """
    eval_val = 1 / ((time_trace[id_2] - time_trace[id_1]) ** 2) * \
        np.correlate(arr1[id_1:id_2], arr2[id_1:id_2], mode='valid')[0]
    return eval_val, np.ones(id_2 - id_1) * eval_val


def self_correlation_function(
        arr1=Z((10000)),
        arr2=Z((10000)),
        id_1=1000,
        id_2=9000,
        time_trace=np.linspace(.0, 16., 10000)):
    """ Own take at cross-correlation function with norm
    Args:
        arr1 (ndarray, optional): Array like data 1
        arr2 (ndarray, optional): Array like data 2
        id_1 (int, optional): Eval start index
        id_2 (int, optional): Eval end index
        time_trace (array, optional): Time vector regarding data
    Returns:
        eval_val (0, np.float): Evaluation with cross corr. built-in function
        eval_list (1, ndarray): Colvolution normalized
    Notes:
        None
    """
    id_diff = id_2 - id_1
    ret_1 = np.convolve(arr1[id_1:id_2], arr2[id_1:id_2], mode='same')
    eval_list = ret_1 / (1e6 * (time_trace[id_2] - time_trace[id_1]))
    ret_2 = np.sum(ret_1)
    eval_val = ret_2 / (1e6 * (time_trace[id_2] - time_trace[id_1]) * id_diff)
    return eval_val, np.array(eval_list)


def daz_mean_deviation(
        arr1=Z((10000)),
        arr2=Z((10000)),
        id_1=1000,
        id_2=9000):
    """ Daihong Zhangs take on the weighted_deviation method
    Args:
        arr1 (ndarray, optional): Array like data 1
        arr2 (ndarray, optional): Array like data 2
        id_1 (int, optional): Eval start index
        id_2 (int, optional): Eval end index
    Returns:
        eval_val (0, np.float): 1/sqrt(sum(square(arr1-arr2))/N)
        eval_list (1, ndarray): square(arr1=arr2)/N
    ToDos:
        None
    Notes:
        None
    """
    id_diff = id_2 - id_1
    eval_list = np.square(
        np.array(arr1[id_1:id_2]) - np.array(arr2[id_1:id_2])) / id_diff
    eval_val = 1 / np.sqrt(np.sum(eval_list))
    return eval_val, np.array(eval_list)


def fft_and_convolve(
        arr1=Z((10000)),
        arr2=Z((10000)),
        id_1=1000,
        id_2=9000,
        time_trace=np.linspace(.0, 16., 10000)):
    """ Successive built-in convolution and FFT functions
    Args:
        arr1 (TYPE, optional): Array like data 1
        arr2 (TYPE, optional): Array like data 2
        id_1 (int, optional): Eval start index
        id_2 (int, optional): Eval end index
    Returns:
        spec_dens (0, np.float): mean(convol)
        convol (1, ndarray): abs(g_ij) ** 2 / (g_i * g_j)
        g_ij (2, ndarray): Convoluted and FFT'd arr1 with arr2
        freqs (3, ndarray): Ordered frequency range of FFT
        g_i (4, ndarray): Convoluted (1's) and FFT'd arr1
        g_j (5, ndarray): Convoluted (1's) and FFT'd arr2
    Todos:
        None
    Notes:
        None
    """
    id_diff = id_2 - id_1
    g_i = np.fft.fft(np.convolve(
        arr1[id_1:id_2], np.ones((id_diff,)), mode='same') / id_diff)
    g_j = np.fft.fft(np.convolve(
        arr2[id_1:id_2], np.ones((id_diff,)), mode='same') / id_diff)
    g_ij = np.fft.fft(np.convolve(
        arr1[id_1:id_2], arr2[id_1:id_2], mode='same') / id_diff)
    # calculations
    convol = np.abs((g_ij) ** 2) / np.abs(g_i * g_j)
    spec_dens = np.mean(convol)
    freqs = np.fft.fftfreq(
        convol.shape[-1], d=(time_trace[1] - time_trace[0]) * 1e-9)
    # sorting
    convol, freqs = (
        np.array(list(x)) for x in
        zip(*sorted(zip(convol, freqs), key=lambda pair: pair[1])))
    return spec_dens, convol, freqs, g_ij, g_i, g_j


def fft_solo(
        arr1=Z((10000)),
        arr2=Z((10000)),
        id_1=1000,
        id_2=9000,
        time_trace=np.linspace(.0, 16., 10000)):
    """ FFT routine for spectral density
    Args:
        arr1 (ndarray, optional): Array like data 1
        arr2 (ndarray, optional): Array like data 2
        id_1 (int, optional): Eval start index
        id_2 (int, optional): Eval end index
    Returns:
        convol (0, ndarray): abs(g_ij) ** 2 / (g_i * g_j)
        g_ij (1, ndarray): g_i* x g_j
        g_i (2, ndarray): fft(arr1)
        g_j (3, ndarray): fft(arr2)
    Notes:
        None
    ToDos:
        None
    """
    g_i = np.fft.fft(arr1[id_1:id_2])
    g_j = np.fft.fft(arr2[id_1:id_2])
    g_ij = np.conj(g_i) * g_j
    convol = np.abs((g_ij) ** 2) / np.abs(g_i * g_j)
    spec_dens = np.mean(convol)
    freqs = np.fft.fftfreq(
        convol.shape[-1], d=(time_trace[1] - time_trace[0]) * 1e-9)
    # sorting
    convol, freqs = (
        np.array(list(x)) for x in
        zip(*sorted(zip(convol, freqs), key=lambda pair: pair[1])))
    return spec_dens, convol, freqs, g_ij, g_i, g_j


def fftconvolve_integrated(
        arr1=Z((10000)),
        arr2=Z((10000)),
        id_1=1000,
        id_2=9000,
        time_trace=np.linspace(.0, 16., 10000)):
    """ Integrated FFT and convolution routine for spectral power density
    Args:
        arr1 (ndarray, optional): Array like data 1
        arr2 (ndarray, optional): Array like data 2
        id_1 (ndarray, optional): Eval start index
        id_2 (ndarray, optional): Eval end index
    Returns:
        convol (0, ndarray): abs(g_ij) ** 2 / (g_i * g_j)
        g_ij (1, ndarray): g_i* x g_j
        g_i (2, ndarray): fftconvolve(arr1)
        g_j (3, ndarray): fftconvolve(arr2)
    Notes:
        None
    """
    id_diff = id_2 - id_1
    g_i = sig.fftconvolve(
        arr1[id_1:id_2], np.ones((id_diff,)), mode='same')
    g_j = sig.fftconvolve(
        arr2[id_1:id_2], np.ones((id_diff,)), mode='same')
    g_ij = np.conj(g_i) * g_j
    convol = np.abs((g_ij) ** 2) / np.abs(g_i * g_j)
    spec_dens = np.mean(convol)
    freqs = np.fft.fftfreq(
        convol.shape[-1], d=(time_trace[1] - time_trace[0]) * 1e-9)
    # sorting
    convol, freqs = (
        np.array(list(x)) for x in
        zip(*sorted(zip(convol, freqs), key=lambda pair: pair[1])))
    return spec_dens, convol, freqs, g_ij, g_i, g_j


def coherence_spectrum_func(
        arr1=Z((10000)),
        arr2=Z((10000)),
        id_1=3000,
        id_2=8000,
        nperseg=5000,
        frequency=1250):
    """ Coherence analysis and spectral power density
    Args:
        arr1 (ndarray, optional): Array like input
        arr2 (ndarray, optional): Same size array like input
        id_1 (int, optional): Starting index of eval
        id_2 (int, optional): End index of eval
        nperseg (int, optional): Anlysis window length
        frequency (int, optional): Sample frequency of origin
    Returns:
        nm_coher (0, np.float): Mean coherence
        coherence (1, ndarray): Spectral coherence
        frequencies (2, ndarray): Frequency range of coherence
    ToDos:
        None
    Notes:
        None
    """
    frequencies, coherence = sig.coherence(
        arr1[id_1:id_2], arr2[id_1:id_2],
        frequency, nperseg=nperseg)
    mn_coher = np.mean(coherence)
    return mn_coher, np.array(coherence), np.array(frequencies)
