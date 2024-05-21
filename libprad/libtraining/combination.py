""" so header ************************************************************ """

import numpy as np

import dat_lists as dat_lists

""" eo header ************************************************************ """


def indexing_combi(
        channels=None):
    """ Sets up index combinations for training routine.
    Args:
        channels (int, opt): number of channels to use combinations from
    Returns:
        combinations (list of list): List of meshgrids with different channels
    Notes:
        VBC
        ECHAN 64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80
        GCHAN  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
        ECHAN 81,82,83,84,85,86,87,54,55,56,57,58,59,60,61,62,63
        GCHAN 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33
    """

    if channels is not None:
        if (channels == 1):
            geom = dat_lists.geom_dat_to_json()
            combinations = [
                np.array([[ch] for ch in geom['channels'][
                    'eChannels']['HBCm']]),
                np.array([[ch] for ch in geom['channels'][
                    'eChannels']['VBC']])]
            cams = ['HBCm', 'VBC']
        if (channels == 3):
            combinations = [
                np.array(np.meshgrid(  # 0 (HBC, 3)
                    range(0, 11), range(12, 22),
                    range(23, 31))).T.reshape(-1, 3),
                np.array(np.meshgrid(  # 1 (VBC, 3)
                    range(64, 74), range(75, 84),
                    range(56, 63))).T.reshape(-1, 3)]
            cams = ['HBCm', 'VBC']
        elif (channels == 4):
            combinations = [
                np.array(np.meshgrid(  # 2 (HBC, 4)
                    range(0, 8), range(9, 16), range(17, 24),
                    range(25, 31))).T.reshape(-1, 4),
                np.array(np.meshgrid(  # 3 (VBC, 4)
                    range(64, 72), range(73, 80), range(81, 87),
                    range(56, 63))).T.reshape(-1, 4)]
            cams = ['HBCm', 'VBC']
        elif (channels == 5):
            combinations = [
                np.array(np.meshgrid(  # 4 (HBC, 5)
                    range(0, 5), range(6, 11), range(13, 18),
                    range(20, 25), range(26, 31))).T.reshape(-1, 5),
                np.array(np.meshgrid(  # 5 (VBC, 5)
                    range(63, 69), range(70, 76), range(77, 82),
                    range(83, 87), range(56, 62))).T.reshape(-1, 5)]
            cams = ['HBCm', 'VBC']
        elif (channels == 6):
            combinations = [
                np.array(np.meshgrid(  # 6 (HBC, 6)
                    range(0, 4), range(6, 10), range(12, 16),
                    range(18, 22), range(22, 26),
                    range(27, 31))).T.reshape(-1, 6),
                np.array(np.meshgrid(  # 7 (VBC, 6)
                    range(64, 68), range(70, 74), range(76, 80),
                    range(82, 86), range(54, 58),
                    range(59, 63))).T.reshape(-1, 6)]
            cams = ['HBCm', 'VBC']
        elif (channels == 7):
            combinations = [
                np.array(np.meshgrid(  # 8 (HBC, 7)
                    range(0, 3), range(5, 8), range(10, 13), range(14, 16),
                    range(17, 20), range(22, 25),
                    range(28, 31))).T.reshape(-1, 7),
                np.array(np.meshgrid(  # 9 (VBC, 7)
                    range(64, 67), range(69, 72), range(74, 77),
                    range(78, 80), range(81, 84), range(54, 57),
                    range(60, 63))).T.reshape(-1, 7)]
            cams = ['HBCm', 'VBC']
        else:
            return (None, None)

        # small combinatory space as specified
        return combinations, cams

    """ if input is empty, set up all the combinations """
    combinations = [
        np.array(np.meshgrid(  # 0 (HBC, 3)
            range(0, 8), range(12, 20), range(23, 31))).T.reshape(-1, 3),
        np.array(np.meshgrid(  # 1 (VBC, 3)
            range(64, 72), range(76, 84), range(56, 63))).T.reshape(-1, 3),
        np.array(np.meshgrid(  # 2 (HBC, 4)
            range(0, 6), range(9, 15), range(18, 24),
            range(25, 31))).T.reshape(-1, 4),
        np.array(np.meshgrid(  # 3 (VBC, 4)
            range(64, 70), range(73, 79), range(81, 87),
            range(57, 63))).T.reshape(-1, 4),
        np.array(np.meshgrid(  # 4 (HBC, 5)
            range(0, 5), range(6, 11), range(13, 18),
            range(20, 25), range(26, 31))).T.reshape(-1, 5),
        np.array(np.meshgrid(  # 5 (VBC, 5)
            range(64, 69), range(70, 75), range(77, 82),
            range(82, 87), range(58, 63))).T.reshape(-1, 5),
        np.array(np.meshgrid(  # 6 (HBC, 6)
            range(0, 4), range(6, 10), range(12, 16),
            range(18, 22), range(22, 26), range(27, 31))).T.reshape(-1, 6),
        np.array(np.meshgrid(  # 7 (VBC, 6)
            range(64, 68), range(70, 74), range(76, 80),
            range(82, 86), range(54, 58), range(59, 63))).T.reshape(-1, 6),
        np.array(np.meshgrid(  # 8 (HBC, 7)
            range(0, 3), range(5, 8), range(10, 13), range(14, 16),
            range(17, 20), range(22, 25), range(28, 31))).T.reshape(-1, 7),
        np.array(np.meshgrid(  # 9 (VBC, 7)
            range(64, 67), range(69, 72), range(74, 77), range(78, 80),
            range(81, 84), range(54, 57), range(60, 63))).T.reshape(-1, 7),
        np.array(np.meshgrid(  # 10  (HBC, 8)
            range(0, 2), range(4, 6), range(7, 9), range(12, 14),
            range(16, 18), range(21, 23), range(25, 27),
            range(29, 31))).T.reshape(-1, 8),
        np.array(np.meshgrid(  # 11 (HBC, 9)
            range(0, 2), range(3, 5), range(6, 8), range(9, 11),
            range(13, 15), range(18, 20), range(21, 23),
            range(25, 27), range(28, 30))).T.reshape(-1, 9),
        np.array(np.meshgrid(  # 12 (HBC + VBC, 6)
            range(2, 6), range(13, 17), range(25, 29),
            range(66, 70), range(79, 83), range(58, 62))).T.reshape(-1, 6),
        np.array(np.meshgrid(  # 13 (HBC + VBC, 8)
            range(0, 3), range(10, 13), range(17, 20), range(28, 31),
            range(64, 67), range(74, 77), range(81, 84),
            range(60, 63))).T.reshape(-1, 8)]

    cams = ['HBCm',      # 0
            'VBC',      # 1
            'HBCm',     # 2
            'VBC',      # 3
            'HBCm',     # 4
            'VBC',      # 5
            'HBCm',     # 6
            'VBC',      # 7
            'HBCm',     # 8
            'VBC',      # 9
            'HBCm',     # 10
            'HBCm',     # 11
            'HBC_VBC',  # 12
            'HBC_VBC',  # 13
            'HBCm',     # 14
            'VBC',      # 15
            'ALL']      # 16

    return combinations, cams
