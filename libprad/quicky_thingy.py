""" ********************************************************************** """

import sys
import warnings
import numpy as np
import json
import matplotlib.pyplot as p

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


def resistances_check():
    file_path = '../files/resistances_20181129.json'
    with open(file_path, 'r') as dat:
        res = json.load(dat)
    dat.close()

    tmp = res['values']['bridges']
    R12 = tmp['R_12']
    R34 = tmp['R_34']
    R13 = tmp['R_13']
    R24 = tmp['R_24']
    R23 = tmp['R_23']
    R14 = tmp['R_14']
    R = list(zip(R14, R12, R23, R34))

    N = len(res['channels'])
    rBar = np.zeros(N)
    R1 = np.zeros(N)
    R2 = np.zeros(N)
    R3 = np.zeros(N)
    R4 = np.zeros(N)
    dR = np.zeros(N)

    for ch in range(0, len(res['channels'])):
        rBar[ch] = 1 / 17 * (
            3 * R12[ch] + 3 * R23[ch] + 3 * R34[ch] +
            3 * R14[ch] + 4 * R13[ch] + 4 * R24[ch])
        R1[ch] = np.sum(np.array([3 / 2, -1 / 2, -1 / 2, -1 / 2]) *
                        R[ch]) + rBar[ch]
        R2[ch] = np.sum(np.array([-1 / 2, 3 / 2, -1 / 2, -1 / 2]) *
                        R[ch]) + rBar[ch]
        R3[ch] = np.sum(np.array([-1 / 2, -1 / 2, 3 / 2, -1 / 2]) *
                        R[ch]) + rBar[ch]
        R4[ch] = np.sum(np.array([-1 / 2, -1 / 2, -1 / 2, 3 / 2]) *
                        R[ch]) + rBar[ch]

        L = [R13[ch], R24[ch], R23[ch], R14[ch]]
        dR[ch] = (np.max(L) - np.min(L)) / np.mean(L)

    fig, (bar, rps, err) = p.subplots(nrows=3)
    for ax in [bar, rps, err]:
        ax.set_xlim(-.5, 17.5)
        ax.set_xticks(np.linspace(0, 17, 18))
        ax.set_xticklabels(res['channels'])
        ax.set_ylabel('R [k$\Omega$]')
        ax.axvspan(-1, 1.5, facecolor='red', alpha=0.3)
        ax.axvspan(1.5, 18, facecolor='blue', alpha=0.3)

    bar.set_title('$\overline{R}$')
    bar.plot([x / 1e3 for x in rBar], ':', marker='o',
             color='k', label='mean bridge')
    bar.text(1.0, 1.03, 'HBC', color='red', rotation=90, size=11)
    bar.text(9, 1.03, 'VBC', color='blue', rotation=90, size=11)

    rps.set_title('R$_{1}$, R$_{2}$, R$_{3}$, R$_{4}$')
    rps.plot([x / 1e3 for x in R1], ':', marker='<',
             color='green', label='R$_{1}$')
    rps.plot([x / 1e3 for x in R2], ':', marker='*',
             color='orange', label='R$_{2}$')
    rps.plot([x / 1e3 for x in R3], ':', marker='p',
             color='cyan', label='R$_{3}$')
    rps.plot([x / 1e3 for x in R4], ':', marker='^',
             color='purple', label='R$_{4}$')
    rps.text(1.0, 1.0, 'HBC', color='red', rotation=90, size=11)
    rps.text(9, 1.0, 'VBC', color='blue', rotation=90, size=11)

    err.set_title('normalized deviation across' +
                  ' [R$_{13}$, R$_{24}$, R$_{23}$, R$_{14}$')
    err.set_ylabel('$\Delta$R in %')
    err.plot([x * 100 for x in dR], ':', marker='h',
             color='red', label='deviation')
    err.text(1.0, 3.0, 'HBC', color='red', rotation=90, size=11)
    err.text(9, 3.0, 'VBC', color='blue', rotation=90, size=11)

    bar.legend()
    rps.legend()
    err.legend()

    # p.show()
    fig.savefig('../results/resistances_check_20181129.pdf')
    p.close('all')

    return


def time_delay_feedback():
    filecols = \
        [[
         '../files/datafiles/peak_XPIDs.dat',
         '../files/datafiles/peak_before_times.dat',
         '../files/datafiles/peak_before_values.dat',
         '../files/datafiles/peak_after_times.dat',
         '../files/datafiles/peak_after_values.dat'],
         [
         '../files/datafiles/steady_XPIDs.dat',
         '../files/datafiles/steady_before_times.dat',
         '../files/datafiles/steady_before_values.dat',
         '../files/datafiles/steady_after_times.dat',
         '../files/datafiles/steady_after_values.dat'],
         [
         [0, 1, 2],
         [0, 1, 2, 3, 4, 5, 6, 7],
         [0, 1, 2, 3, 4, 5, 6, 7],
         [0, 1, 2, 3, 4, 5, 6, 7],
         [0, 1, 2, 3, 4, 5, 6, 7]
         ]]

    # peak values
    res = [None] * len(filecols[0])
    for i in range(0, len(filecols[0])):
        cols = filecols[2][i]
        res[i] = np.loadtxt(
            filecols[0][i], delimiter='\t',
            dtype=np.float, usecols=cols)
    vbc_peak_tdiff = \
        [a - res[3][i, 6] for i, a in enumerate(res[3][:, 0])]
    hbc_peak_tdiff = \
        [a - res[3][i, 6] for i, a in enumerate(res[3][:, 1])]

    vbc_peak_frad = \
        [a / res[4][i, 4] for i, a in enumerate(res[4][:, 0])]
    hbc_peak_frad = \
        [a / res[4][i, 4] for i, a in enumerate(res[4][:, 1])]

    vmin = min(min(vbc_peak_frad), min(hbc_peak_frad))
    vmax = max(max(vbc_peak_frad), max(hbc_peak_frad))

    pmin = min(min(res[4][:, 0]), min(res[4][:, 1]))
    pmax = max(max(res[4][:, 0]), max(res[4][:, 1]))

    fig = p.figure()
    host = fig.add_axes([0.1, 1., 0.8, 0.4], xticklabels=[])
    guest = host.twinx()
    host.grid(b=True, which='major', linestyle='-.')
    cm = p.cm.get_cmap('spectral')

    partner = fig.add_axes([0.1, 0.6, 0.8, 0.4])
    child = partner.twinx()
    partner.grid(b=True, which='major', linestyle='-.')

    sibling = fig.add_axes([0.1, .1, 0.8, 0.4])
    cousin = sibling.twinx()
    sibling.grid(b=True, which='major', linestyle='-.')

    sibling.axhspan(np.mean(vbc_peak_frad) * 0.52,
                    np.mean(vbc_peak_frad) * 1.48,
                    facecolor='gray', alpha=0.3)
    sibling.axhspan(np.mean(vbc_peak_frad) * 0.66,
                    np.mean(vbc_peak_frad) * 1.34,
                    facecolor='gray', alpha=0.3)

    vbc = host.scatter(
        vbc_peak_tdiff, res[4][:, 0], c=vbc_peak_frad,
        vmin=vmin, vmax=vmax, s=80.,
        cmap=cm, label='peak $\\Delta$t$_{VBC}$',
        marker='^')
    hbc = host.scatter(
        hbc_peak_tdiff, res[4][:, 1], c=hbc_peak_frad,
        vmin=vmin, vmax=vmax, s=80.,
        cmap=cm, label='peak $\\Delta$t$_{HBC}$',
        marker='s')
    guest.scatter(
        vbc_peak_tdiff, res[4][:, 4],
        c='k', marker='x', alpha=.0)

    cbar = p.colorbar(vbc, ax=host)
    cbar.set_label('f$_{rad}$')

    vbcL = sibling.scatter(
        res[4][:, 4], vbc_peak_frad[:], c=res[4][:, 0],
        vmin=pmin, vmax=pmax, s=80.,
        cmap=cm, label='f$_{rad, VBC}$', marker='^')
    hbcL = sibling.scatter(
        res[4][:, 4], hbc_peak_frad[:], c=res[4][:, 1],
        vmin=pmin, vmax=pmax, s=80.,
        cmap=cm, label='f$_{rad, HBC}$', marker='s')
    cousin.scatter(
        res[4][:, 4], res[4][:, 2], s=80.,
        c='k', marker='x', alpha=.0)

    cbar = p.colorbar(vbcL, ax=sibling)
    cbar.set_label('P$_{rad}$ [MW]')

    L = len(vbc_peak_frad)
    for i, txt in enumerate(np.linspace(0, L - 1, L)):
        host.annotate(int(txt), (vbc_peak_tdiff[i], res[4][i, 0]))
        host.annotate(int(txt), (hbc_peak_tdiff[i], res[4][i, 1]))

        sibling.annotate(int(txt), (res[4][i, 4], vbc_peak_frad[i]))
        sibling.annotate(int(txt), (res[4][i, 4], hbc_peak_frad[i]))

    # steady values
    res = [None] * len(filecols[1])
    for i in range(0, len(filecols[1])):
        cols = filecols[2][i]
        res[i] = np.loadtxt(
            filecols[1][i], delimiter='\t',
            dtype=np.float, usecols=cols)
    vbc_steady_tdiff = \
        [a - res[3][i, 6] for i, a in enumerate(res[3][:, 0])]
    hbc_steady_tdiff = \
        [a - res[3][i, 6] for i, a in enumerate(res[3][:, 1])]

    vbc_steady_frad = \
        [a / res[4][i, 4] for i, a in enumerate(res[4][:, 0])]
    hbc_steady_frad = \
        [a / res[4][i, 4] for i, a in enumerate(res[4][:, 1])]

    vmin = min(min(vbc_steady_frad), min(hbc_steady_frad))
    vmax = max(max(vbc_steady_frad), max(hbc_steady_frad))

    vbcS = partner.scatter(
        vbc_steady_tdiff, res[4][:, 0], c=vbc_steady_frad,
        vmin=vmin, vmax=vmax, s=80.,
        cmap=cm, label='eq. $\\Delta$t$_{VBC}$', marker='^')
    hbcS = partner.scatter(
        hbc_steady_tdiff, res[4][:, 1], c=hbc_steady_frad,
        vmin=vmin, vmax=vmax, s=80.,
        cmap=cm, label='eq. $\\Delta$t$_{HBC}$', marker='s')
    child.scatter(
        vbc_steady_tdiff, res[4][:, 4],
        c='k', marker='x', alpha=.0)

    L = len(vbc_steady_frad)
    for i, txt in enumerate(np.linspace(0, L - 1, L)):
        partner.annotate(int(txt), (vbc_steady_tdiff[i], res[4][i, 0]))
        partner.annotate(int(txt), (hbc_steady_tdiff[i], res[4][i, 1]))

    cbar = p.colorbar(vbcS, ax=partner, ticks=[.4, .8, 1.2, 1.6])
    cbar.set_label('f$_{rad}$')

    host.yaxis.label.set_color('k')
    host.tick_params(axis='y', colors='k')
    host.legend([vbc, hbc], [l.get_label() for l in [vbc, hbc]])
    host.set_ylabel('P$_{rad}$ [MW]')

    partner.yaxis.label.set_color('k')
    partner.tick_params(axis='y', colors='k')
    partner.legend([vbcS, hbcS], [l.get_label() for l in [vbcS, hbcS]])
    partner.set_ylabel('P$_{rad}$ [MW]')
    partner.set_xlabel('$\\Delta$t')

    sibling.set_ylabel('f$_{rad}$')
    child.set_ylabel('ECRH [MW]')
    guest.set_ylabel('ECRH [MW]')

    sibling.legend([vbcL, hbcL], [l.get_label() for l in [vbcL, hbcL]])
    sibling.set_xlabel('ECRH [MW]')
    cousin.set_ylabel('n$_{e}$ [10$^{19}$ m$^{-3}$]')

    xmin, xmax = \
        min(min(vbc_peak_tdiff), min(hbc_peak_tdiff),
            min(vbc_steady_tdiff), min(vbc_steady_tdiff)) - .05, \
        max(max(vbc_peak_tdiff), max(hbc_peak_tdiff),
            max(vbc_steady_tdiff), max(hbc_steady_tdiff)) + .05
    host.set_xlim(xmin, xmax)
    partner.set_xlim(xmin, xmax)

    fig.set_size_inches(10, 10)
    fig.savefig("../results/CURRENT/quicky_thing.pdf",
                bbox_inches='tight', dpi=169.0)
    p.close('all')

    return res
