""" **************************************************************************
    so header """

import os
import sys
import json
import warnings
import numpy as np
import matplotlib.pyplot as p
import mClass

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")

Z = np.zeros
ones = np.ones

""" eo header
************************************************************************** """


def pSetup(
        time=Z(10000),
        V=Z((128, 10000)),
        ch=0,
        indent_level='\t'):
    """ Sets up the plot for the clicky_thingy to be exported and looked at.
    Args:
        time: Time vector.
        V: Multidim voltage array.
        ch: Channel currently under investigation.
        priority_object: Constants and vars previously set up about XP.
        indent_level: Printing indentation.
    Returns:
        f, ax: Figure and axis objects to plot to/plotted.
    """

    if False:  # silent
        print(indent_level + '\t\t\\\ plot setup')
    # basic setup
    f = p.figure(figsize=(16, 9))
    ax = f.add_subplot(111)
    # plot lines of neighboring channels
    ax.plot(time, V[ch], color='r', label='Ch#' + str(ch))
    ax.plot(time, V[ch - 1], ':', color='b', label='Ch#' + str(ch - 1))
    ax.plot(time, V[ch + 1], ':', color='g', label='Ch#' + str(ch + 1))
    # make legend
    ax.legend()
    return f, ax


def if_found_shift(
        time=[],
        coords=[[], []],
        frstLVL=0.0,
        voltage=[]):
    """ For the given coordinates from the clicks the value shift
        is done according to the matching level in a small range around
        the FIRST click.
    Args:
        time: Time vector.
        coords: Click coordinates, 2x2 for x,y.
        frstLVL: Because second flips are possible, and the shift is then
            different we need do consider the previous level. 0 at start.
        voltage: Onedimensional voltage array.
    Returns:
        voltage: Shifted onedimensional voltage array.
        val: Value found at shift positions to move around.
        idx1: First click position (index).
        idx2: Second click position.
        idx3: Position of closest value around idx2 to idx1 level.
    """

    # looking for sensible positions in clicky
    clkID1, Tclk1 = mClass.find_nearest(np.array(time), coords[0][0])
    clkID2, Tclk2 = mClass.find_nearest(np.array(time), coords[1][0])
    # maximum values around pos
    Mclk1 = np.max(voltage[clkID1 - 50:clkID1 + 50])
    Mclk2 = np.max(voltage[clkID2 - 50:clkID2 + 50])
    # get the IDs of max vals
    idx1 = voltage[clkID1 - 50:clkID1 + 50].index(Mclk1) + (clkID1 - 50)
    idx2 = voltage[clkID2 - 50:clkID2 + 50].index(Mclk2) + (clkID2 - 50)
    # get the value related to first click
    idx3, val = mClass.find_nearest(
        np.array(voltage[idx2 - 50:idx2 + 50]),
        voltage[idx1])

    # shift level as dLVL so for second correction is adequate
    idx3 = idx3 + (idx2 - 50)
    dLVL = abs(frstLVL - val)

    # getting shifty in here
    voltage[idx1:idx3] = \
        [(val + abs(val - V)) if V < val * 1.05
         else (V + dLVL) for V in
         voltage[idx1:idx3]]
    # the boys are back in town
    return voltage, val, idx1, idx2, idx3


def prev_flip(
        t=Z(10000),
        voltage=Z(10000),
        FFDB={'none': None},
        indent_level='\t'):
    """ Applies flip positions and levels from previously stored
        in data loaded.
    Args:
        t: Time vector.
        voltage: Multichannel voltage array.
        FFDB: Flip fix database as dictionary, shortened.
        priority_object: Constants and vars to be used.
        indent_level: Printing indentation level.
    Returns:
        voltage: Multichannel voltage array.
    """
    print(indent_level + '\t\t\\\ adjusting voltage according to found data')
    try:
        flipd_chan = [0, 1, 2, 3, 4, 5, 6, 7, 10, 13, 14, 15,
                      16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28,
                      51, 52, 53, 54, 55, 56, 57,
                      70, 71, 72, 73, 74, 75, 84, 85]

        for ch in flipd_chan:  # flippers
            if (ch in FFDB['channels']):
                frstLVL = 0.0  # not found yet
                i = FFDB['channels'].index(ch)  # where the data is and set
                [val, idx1, idx3] = \
                    [FFDB['level'][i],
                     FFDB['first_click_ID'][i],
                     FFDB['matchup_ID'][i]]

                if (val is not None) and (idx1 is not None):
                    # re-adjust levels
                    dLVL = abs(frstLVL - val)
                    voltage[ch][idx1:idx3] = \
                        [(val + abs(val - V)) if V < val * 1.05
                         else (V + dLVL) for V in voltage[ch][idx1:idx3]]

                    # if there is a true stored here
                    if FFDB['second_flip'][i]:
                        frstLVL = val  # load from dict
                        [val, idx1, idx3] = \
                            [FFDB['second level'][i],
                             FFDB['second first_click_ID'][i],
                             FFDB['second matchup_ID'][i]]

                        # calculate level diff and adjust values
                        dLVL = abs(frstLVL - val)
                        voltage[ch][idx1:idx3] = \
                            [(val + abs(val - V)) if V < val * 1.05
                             else (V + dLVL) for V in voltage[ch][idx1:idx3]]

                    # show the precious so that one can check
                    [f, ax] = \
                        pSetup(t, voltage, ch)
                    ax.plot(
                        t, voltage[ch], color='orange',
                        label='CH#' + str(ch) + '$^{\prime}$')
                    ax.legend()
                    f.savefig('../results/test.pdf')
                    p.close(f)

            else:
                pass

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print('\n' + '\t\\\ ', exc_type, fname, exc_tb.tb_lineno)

    return voltage


def reset_ffdb(
        FlipDat={}):
    """ Resets the current flip fix database handed over to the function.
    Args:
        FlipDat: Shortened FFDB.
    Returns:
        FlipDat: Shortened FFDB.
    """

    # just a huge dict setup
    [FlipDat['channels'], FlipDat['level'], FlipDat['first_click_ID'],
     FlipDat['second_click_ID'], FlipDat['matchup_ID'],
     FlipDat['second_flip'], FlipDat['second level'],
     FlipDat['second first_click_ID'],
     FlipDat['second second_click_ID'],
     FlipDat['second matchup_ID']] = \
        [[], [], [], [], [], [], [], [], [], []]

    return FlipDat  # pew pew pew


def clicky(
        t=Z(10000),
        voltage=Z((128, 10000)),
        ch=0,
        coords=[0.0, 0.0],
        frstLVL=1.0,
        indent_level='\t'):
    """ Sets up plot, collects clicks and and shifts the levels of what found.
    Args:
        t: Time vector.
        V: Multichannel voltage array.
        coords: Coordinates of clicks.
        priority_object: Constants and vars loaded before.
        indent_level: Printing indentation.
    Returns:
        V: Shifted voltage multichannel array.
        val: Value of points where voltage is shifted to.
        idx1-3: Index for clicks and same levels around.
    """

    # setting up figure and showing to user for possible fix
    [f, ax] = \
        pSetup(t, voltage, ch)

    # IMPORTANT: Event ridden click for plot which returns location
    def onclick(event):
        coords.append((event.xdata, event.ydata))
        if (len(coords) == 2):
            f.canvas.mpl_disconnect(cid)
            p.close()
        return

    # show and hold until two clicks, then return
    cid = f.canvas.mpl_connect('button_press_event', onclick)
    p.show()
    # as long
    while (len(coords) < 2):
        p.pause(.2)

    # double click on same spot means void
    if (coords[1][0] - 25 <= coords[0][0] <= coords[1][0] + 25):
        return [voltage, None, None, None, None]

    # quick dirty magic function that flips back
    voltage[ch], val, idx1, idx2, idx3 = \
        if_found_shift(t, coords, frstLVL, voltage[ch])

    # plotting and saving, then close
    ax.plot(t, voltage[ch], color='orange',
            label='CH#' + str(ch) + '$^{\prime}$')
    ax.legend()
    f.savefig('../results/test.pdf')
    p.close(f)

    return [voltage, val, idx1, idx2, idx3]  # return data


def clicky_thingy(
        voltage=Z((128, 10000)),
        date='20181010',
        shotno='032',
        indent_level='\t'):
    """ Returns for two clicks in plot the position and value. Done for a
        multitude of channels specified in flipd_chn. Repeat saved data
        in a json file that has the shot, channels, location and levels.
    Args:
        voltage: Preloaded data from the archive of voltages from Bolo.
        priority_object: Constants, times and variables needed once.
        date: Date.
        shotno: XP ID.
        indent_level: Printing indentation level.
    Returns:
        voltage: Voltage signals fixed.
    """

    print(indent_level + '\t>> Channel fix selection mode',
          'with user input of clicky mightyness...')

    # vars
    indent_level = indent_level + '\t'
    dat_file = 'libradcalc/flipfixDB.json'
    flipd_chan = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 14, 15,
                  16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28,
                  51, 52, 53, 54, 55, 56, 57,
                  70, 71, 72, 73, 74, 75, 84, 85]

    # voltages and time
    V = voltage['values']  # in V
    t = voltage['dimensions']  # in ns

    # looking for file where flip data is saved
    if not os.path.isfile(dat_file):
        print(indent_level + '\t>> Couldn\'t find any file at:', dat_file)
        flipfixDB = {'dates': {date: {'XPID': {shotno: {}}}}}  # default

    elif os.path.isfile(dat_file):
        print(indent_level + '\t>> Found file at:', dat_file)

        try:
            with open(dat_file, 'r') as infile:
                flipfixDB = json.load(infile)  # loading ...
            infile.close()

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print('\n' + indent_level + '\t\\\ ', exc_type,
                  fname, exc_tb.tb_lineno)

            flipfixDB = {'dates': {}}  # if not successfull, reset

    if not (date in flipfixDB['dates'].keys()):
        flipfixDB['dates'][date] = {'XPID': {shotno: {}}}  # set shot

    if not (shotno in flipfixDB['dates'][date]['XPID'].keys()):
        # if not shot in data, also reset entirely with templates
        flipfixDB['dates'][date]['XPID'][shotno] = \
            {'channels': [], 'level': [], 'first_click_ID': [],
             'second_click_ID': [], 'matchup_ID': [],
             'second_flip': [], 'second level': [],
             'second first_click_ID': [], 'second second_click_ID': [],
             'second matchup_ID': []}

    else:
        try:
            FlipDat = flipfixDB['dates'][date]['XPID'][shotno]  # shorten
            if ((np.sort(FlipDat['channels']).tolist() ==
                 np.sort(flipd_chan).tolist())):  # only repeat if self
                print(indent_level + '\t>> Repeat found from', dat_file,
                      'and channels', FlipDat['channels'])

                # same flips as file, hence previous
                V = prev_flip(t, V, FlipDat)
                return V  # done therefore

            else:
                # dont reset, trusting previous selections
                # FlipDat = reset_ffdb(FlipDat)
                pass

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print('\n' + indent_level + '\t\\\ ', exc_type,
                  fname, exc_tb.tb_lineno)
            FlipDat = reset_ffdb(FlipDat)  # reset

    FFDB = flipfixDB['dates'][date]['XPID'][shotno]  # shorten
    # over all flippers
    for ch in flipd_chan:
        if ch not in FFDB['channels']:

            stat = True  # messed up?
            bflp = V[:]
            while stat:  # do until satisfied with results
                [coords, frstLVL] = \
                    [[], 0.0]  # reset

                # clicky function that sets up plots, gets values
                V, val, idx1, idx2, idx3 = clicky(
                    t, bflp[:], ch, coords, frstLVL)
                stat = mClass.query_yes_no(
                    indent_level + '\t\t>> Selection OK for ' +
                    str(ch) + ':\n' + indent_level +
                    '\t\t   Try again?', 'no')

            # if clicked double and didnt find stuff
            if (val is None) and (idx1 is None) and \
               (idx2 is None) and (idx3 is None):
                FFDB['channels'].append(int(ch))
                FFDB['level'].append(None)
                FFDB['first_click_ID'].append(None)
                FFDB['second_click_ID'].append(None)
                FFDB['matchup_ID'].append(None)
                FFDB['second_flip'].append(False)
                FFDB['second level'].append(None)
                FFDB['second first_click_ID'].append(None)
                FFDB['second second_click_ID'].append(None)
                FFDB['second matchup_ID'].append(None)

            elif (val is not None) and (idx1 is not None) and \
                 (idx2 is not None) and (idx3 is not None):
                # DATA
                FFDB['channels'].append(int(ch))
                FFDB['level'].append(val)
                FFDB['first_click_ID'].append(int(idx1))
                FFDB['second_click_ID'].append(int(idx2))
                FFDB['matchup_ID'].append(int(idx3))

                # if second flip found, try again
                scndFlp = mClass.query_yes_no(
                    indent_level + '\t\t>> Second flip found in ' +
                    str(ch) + '?\n' + indent_level +
                    '\t\t   Try second?', 'yes')

                if scndFlp:  # second flip seen?
                    bflp = V[:]

                    stat = True  # messed up?
                    while stat:  # do until satisfied with results
                        [frstLVL, coords] = [val, []]

                        [V, val, idx1, idx2, idx3] = clicky(
                            t, bflp, ch, coords, frstLVL)
                        stat = mClass.query_yes_no(
                            indent_level + '\t\t>> Second OK for ' +
                            str(ch) + ':\n' + indent_level +
                            '\t\t   Try again?', 'no')

                    if (val is not None) and (idx1 is not None) and \
                       (idx2 is not None) and (idx3 is not None):
                        # DATA
                        FFDB['second_flip'].append(True)
                        FFDB['second level'].append(val)
                        FFDB['second first_click_ID'].append(int(idx1))
                        FFDB['second second_click_ID'].append(int(idx2))
                        FFDB['second matchup_ID'].append(int(idx3))
                    else:
                        # nothign in second, that's it
                        FFDB['second_flip'].append(False)
                        FFDB['second level'].append(None)
                        FFDB['second first_click_ID'].append(None)
                        FFDB['second second_click_ID'].append(None)
                        FFDB['second matchup_ID'].append(None)

                else:
                    # nothign in second, that's it
                    FFDB['second_flip'].append(False)
                    FFDB['second level'].append(None)
                    FFDB['second first_click_ID'].append(None)
                    FFDB['second second_click_ID'].append(None)
                    FFDB['second matchup_ID'].append(None)

                p.close('all')
        else:
            print(indent_level + '\t >> Channel', ch, 'already in database')
            # use prev found flip data
            index = FFDB['channels'].index(ch)

            # only reuse if data, else none
            if (FFDB['level'][index] is not None) and \
               (FFDB['first_click_ID'][index] is not None):
                coords = \
                    [[t[FFDB['first_click_ID'][index]], 0.0],
                     [t[FFDB['second_click_ID'][index]], 0.0]]

                V[ch], val1, idx1, idx2, idx3 = \
                    if_found_shift(t, coords, 0.0, V[ch])

                # if second flip stored
                if FFDB['second_flip'][index]:
                    coords = \
                        [[t[FFDB['second first_click_ID'][index]], 0.0],
                         [t[FFDB['second second_click_ID'][index]], 0.0]]
                    V[ch], val2, idx1, idx2, idx3 = \
                        if_found_shift(t, coords, val1, V[ch])

    # saving everything that is stored in FFDB back to file
    with open(dat_file, 'w') as outfile:
        json.dump(flipfixDB, outfile, indent=4, sort_keys=False)
    outfile.close()

    # put back and return correction
    return V
