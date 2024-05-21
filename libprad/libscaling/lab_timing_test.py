""" ********************************************************************** """

import sys
import warnings
import numpy as np
import json
import matplotlib.pyplot as p
import pandas as pd

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")
warnings.filterwarnings("ignore", "Reloaded modules")

stdwrite = sys.stdout.write
stdflush = sys.stdout.flush
ones = np.ones
Z = np.zeros

""" ********************************************************************** """


def lab_data_grab():
    """ loads and calculates the data from Dasen's file
    Returns:
        data (0, pandas.dataframe): results from file, no calc
    """

    # loc = '../files/analysis_time_measurements w-o_laser.xlsx'
    loc = '../files/time_measurements_old.xlsx'
    file = pd.ExcelFile(loc)

    df = {sheet_name: file.parse(sheet_name)
          for sheet_name in file.sheet_names}
    data = df['Tabelle1'][:551]

    foo = {'label': 'lab_time_deviation',
           'measurements': {}}
    values = foo['measurements']

    DF_keys = [
        'DAQ Time (Time of Acquisition) [s]',   # 0
        'Sample Time [ms]',                     # 1
        'Frequency (f_DAQ) [kHz]',              # 2
        'No. of Samples',                       # 3
        'Repetition',                           # 4
        'Rept',                                 # 5
        'Measured Time for DAQ (Tav) [ms]',     # 6
        'Loop DAQ Timing [ms]',                 # 7
        'Predicted Time [ms] to Finalize DAQ',  # 8
        'Comments',                             # 9
        'Means of Measured Time for DAQ [ms]',  # 10
        'Standard deviation  1',                # 11
        'Means of Loop DAQ Timing [ms]',        # 12
        'Standard deviation  2']                # 13

    values['DAQ times'] = data[DF_keys[0]][::50].tolist()
    values['Sample times'] = data[DF_keys[1]][:50:10].tolist()
    values['Sample frequencies'] = data[DF_keys[2]][:50:10].tolist()
    values['no. of samples'] = data[DF_keys[3]][:301:10].tolist()

    M, L = len(values['DAQ times']), len(values['Sample times'])
    bar = Z((M, L, 3, 10))
    ex = Z((M, L, 4))
    for i, T in enumerate(values['DAQ times']):
        for j, t in enumerate(values['Sample times']):
            cm = (i * L * 10)
            c = (i * L * 10) + (j * 10)
            # predicted DAQ time [ms]
            bar[i, j, 0, 0] = data[DF_keys[8]][cm]
            # measured DAQ time [ms]
            bar[i, j, 1, :] = data[DF_keys[6]][c:c + 10]
            # loop sample time [ms]
            bar[i, j, 2, :] = data[DF_keys[7]][c:c + 10]
            # mean and standard deviation of measured DAQ time [ms]
            ex[i, j, 0] = np.mean(bar[i, j, 1, :])
            ex[i, j, 1] = np.std(bar[i, j, 1, :])
            # mean and standard deviation of loop sample time [ms]
            ex[i, j, 2] = np.mean(bar[i, j, 2, :])
            ex[i, j, 3] = np.std(bar[i, j, 2, :])
    foo['results'] = ex.tolist()
    foo['dataset'] = bar.tolist()

    # calc deviations and stuff
    epsilon = Z((M, L, 6))
    for i in range(6):
        for j in range(L):
            # 0: measured DAQ time - predicted DAQ time
            epsilon[i, j, 0] = ex[i, j, 0] - bar[i, 0, 0, 0]
            # 1: measured DAQ time / no of samples -> t real
            epsilon[i, j, 1] = ex[i, j, 0] / \
                values['no. of samples'][(i * L) + (j)]
            # 2: [1] - set sample time -> t error
            epsilon[i, j, 2] = epsilon[i, j, 1] - \
                values['Sample times'][j]
            # 3: [2] / set sample time -> % error
            epsilon[i, j, 3] = epsilon[i, j, 2] / \
                values['Sample times'][j]
            # 4: loop timing - set sample time -> t error
            epsilon[i, j, 4] = ex[i, j, 2] - \
                values['Sample times'][j]
            # 5: [4] / set sample time -> % error
            epsilon[i, j, 5] = epsilon[i, j, 4] / \
                values['Sample times'][j]
    foo['errors'] = epsilon.tolist()

    if False:
        plot_1(L=L, bar=bar, values=values)
        plot_2(L=L, values=values, ex=ex, epsilon=epsilon)

    # saving
    with open('../files/timing_lab_DAQ.json', 'w') as f:
        json.dump(foo, f, sort_keys=False, indent=4)
    f.close()

    return  # data


def plot_1(
        L=5,
        bar=Z((10, 10)),
        values={}):

    fig, axes = p.subplots(6, 2)
    for i in range(6):
        for j in range(L):

            # measurement times
            axes[i, 0].plot(
                bar[i, j, 1, :], '-.', alpha=0.7, lw=2.,
                label=str(values['Sample times'][j]))
            axes[i, 0].set_title(
                'set time: ' + str(bar[i, 0, 0, 0]) + ' ms')
            axes[i, 0].set_ylabel('measured DAQ time [ms]')

            # sample times
            axes[i, 1].plot(bar[i, j, 2, :], '-.', alpha=0.7, lw=2.0,
                            label=str(values['Sample times'][j]))
            axes[i, 1].set_title('set time: ' + str(bar[i, 0, 0, 0]))
            axes[i, 1].set_ylabel('measured loop timing [ms]')

        for k in [0, 1]:
            axes[i, k].legend()
            axes[i, k].grid(b=True, which='major', linestyle='-.')

    for k in [0, 1]:
        axes[6 - 1, k].set_xlabel('no. of try')

    fig.set_size_inches(15., 6 * 7.)
    fig.savefig('../results/laboratory_DAQ_times.pdf',
                bbox_inches='tight', dpi=169.0)
    p.close('all')

    return


def plot_2(
        L,
        values={},
        ex=Z((10, 10)),
        epsilon=Z((10, 10))):

    fig, axes = p.subplots(6)
    for j in range(L):
        # 0: measured DAQ time - predicted DAQ time
        axes[0].errorbar(
            values['DAQ times'], epsilon[:, j, 0],
            yerr=ex[:, j, 1], uplims=True, lolims=True,
            alpha=0.8, lw=2., ls='-.',
            label=values['Sample times'][j])
        # 1: measured DAQ time / no of samples -> t real
        axes[1].errorbar(
            values['DAQ times'], epsilon[:, j, 1],
            yerr=ex[:, j, 1] / values['no. of samples'][j],
            uplims=True, lolims=True,
            ls='-.', alpha=0.7, lw=2.,
            label=values['Sample times'][j])
        # 2: [1] - set sample time -> t error
        axes[2].errorbar(
            values['DAQ times'], epsilon[:, j, 2],
            yerr=ex[:, j, 1] / values['no. of samples'][j],
            uplims=True, lolims=True,
            ls='-.', alpha=0.7, lw=2.,
            label=values['Sample times'][j])
        # 3: [2] / set sample time -> % error
        axes[3].errorbar(
            values['DAQ times'], epsilon[:, j, 3] * 100,
            yerr=(ex[:, j, 1] / values['no. of samples'][j]) /
            values['Sample times'][j] * 100, uplims=True, lolims=True,
            ls='-.', alpha=0.7, lw=2.,
            label=values['Sample times'][j])
        # 4: loop timing - set sample time -> t error
        axes[4].errorbar(
            values['DAQ times'], epsilon[:, j, 4],
            yerr=ex[:, j, 3], uplims=True, lolims=True,
            ls='-.', alpha=0.7, lw=2.,
            label=values['Sample times'][j])
        # 5: [4] / set sample time -> % error
        axes[5].errorbar(
            values['DAQ times'], epsilon[:, j, 5] * 100,
            yerr=ex[:, j, 3] / values['Sample times'][j] * 100,
            uplims=True, lolims=True,
            ls='-.', alpha=0.7, lw=2.,
            label=values['Sample times'][j])

    for k in [0, 1, 2, 3, 4, 5]:
        axes[k].set_xlim(4., 51.)
        axes[k].set_xlabel('DAQ time [s]')

    axes[0].set_ylabel('DAQ time deviation [s]')
    axes[1].set_ylabel('actual sample time [ms]')
    axes[2].set_ylabel('timing error [ms]')
    axes[3].set_ylabel('timing error [%]')
    axes[4].set_ylabel('loop timing error [ms]')
    axes[5].set_ylabel('loop timing error [%]')

    for m in range(6):
        axes[m].legend()
        axes[m].grid(b=True, which='major', linestyle='-.')

    fig.set_size_inches(15., 6 * 7.)
    fig.savefig('../results/timing_errors_DAQ.pdf',
                bbox_inches='tight', dpi=169.0)
    p.close('all')

    return
