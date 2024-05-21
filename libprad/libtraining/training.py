""" so header ************************************************************ """

import sys
import os
import combination as comb
import correlation as corr
import sensitivity as sensitivity
import webapi_access as api
import mClass

base = os.getcwd()
stdwrite = sys.stdout.write
stdflush = sys.stdout.flush

""" eo header ************************************************************ """


def training_master(
        program='20181010.032',
        URI_vers='V4',
        mode=None,
        nCH=None,
        saving=True,
        plot=False):
    """ taking combinations and calculating correlation
        training for channel selection
    Args:
        program (str, optional): XPID. Defaults to '20181010.032'.
        URI_vers (str, optional): Data entry. Defaults to 'V4'.
        mode ([type], optional): Evaluation mode. Defaults to None.
        nCH ([type], optional): Channels per comb. Defaults to None.
        saving (bool, optional): Save?. Defaults to True.
        plot (bool, optional): Plot?. Defaults to False.
    """
    print('>> start training on ' + program + '\n')
    base_URI = 'http://archive-webapi.ipp-hgw.mpg.de/'
    dat_URI = base_URI + 'Test/raw/W7XAnalysis/QSB_Bolometry/'

    program_info = api.xpid_info(program=program)[0]

    dat = {}
    dat['BoloSignal'] = api.download_single(
        data_req=dat_URI + 'BoloAdjusted_DATASTREAM/' + URI_vers + '/',
        program_info=program_info, debug=False)
    dat['P_rad_hbc'] = api.download_single(
        data_req=dat_URI + 'PradHBCm_DATASTREAM/' + URI_vers + '/0/PradHBCm/',
        program_info=program_info, debug=False)
    dat['power'] = api.download_single(
        data_req=dat_URI + 'PradChannels_DATASTREAM/' + URI_vers + '/',
        program_info=program_info, debug=False)

    # necessary info for computation
    prio, status = api.do_before_running(
        program_info=program_info, program=program,
        date=program[0:8], data_object=dat)
    if status:
        print('\t\t\\\ failed at priority data, return')
        return

    # set up combination space
    combinations, cams = comb.indexing_combi(channels=nCH)

    if mode is None:
        modes = ['weighted_deviation', 'self_correlation', 'mean_deviation',
                 'fftconvolve_integrated', 'fft_solo', 'fft_and_convolve',
                 'correlation', 'coherence_fft']
    else:
        modes = [mode]

    for l, m in enumerate(modes):
        print('\t\\\ mode: ' + m)
        for n, c in enumerate(combinations):
            training_results = corr.correlate(
                plot=plot, program=program, saving=True,
                prad=dat['P_rad_hbc'], channels=dat['power'],
                prio=prio, method=m, sel=c, cam=cams[n])

        sensitivity_results, combination_results = \
            sensitivity.sensitivity_evaluation(
                program=program, method=m, saving=saving,
                prad=dat['P_rad_hbc'], cameras=cams, prio=prio,
                amounts=[len(x[0]) for i, x in enumerate(combinations)],
                channels=dat['power'], plot=plot)
    return (training_results, sensitivity_results)
