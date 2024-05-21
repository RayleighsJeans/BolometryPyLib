""" ********************************************************************** """

import os
import sys
import matplotlib
import matplotlib.pyplot as p
import request as request
import importlib
import urllib
import math
import numpy as np
import time
import scipy as scp
import pprint
import warnings
import MDSplus

""" ********************************************************************** """

folders = ['W:/Documents/LABGIT/IDL2PY-PORT/',
           'C:/Users/Philipp Hacker/Documents/gitlab/IDL2PY-PORT/',
           '//sv-it-fs-1/roaming$/pih/Documents/git/QSB_Bolometry/',
           '/home/pha/Documents/gitlab/IDL2PY-PORT/',
           '/home/pha/Documents/IDL2PY-PORT/',
           '/home/pha/Dokumente/QSB_Bolometry/',
           '//share.ipp-hgw.mpg.de/documents/pih/Documents/git/QSB_Bolometry/']
root_folder = folders[6]

# QSB libraries
sys.path.append(root_folder + r'files/customs/')
sys.path.append(root_folder + r'libprad/')
sys.path.append(root_folder + r'libprad/libwebapi/')
sys.path.append(root_folder + r'libprad/liboutput/')
sys.path.append(root_folder + r'libprad/libtraining/')
sys.path.append(root_folder + r'libprad/libradcalc/')
sys.path.append(root_folder + r'libprad/libscaling/')
# inversion libraries
sys.path.append(root_folder + r'libinversion/')
sys.path.append(root_folder + r'libinversion/libcalc/')
sys.path.append(root_folder + r'libinversion/libaccessoires/')
sys.path.append(root_folder + r'libinversion/liboutput/')
# HEXOS libraries
sys.path.append(root_folder + '../hexos-data-viewer/')
sys.path.append(root_folder + '../HRXIS_DataAccessTools/')
# Thomas Wegner subscript liobraries
lib = '//x-drive/OP1.2b/Impurity Transport Group/Scaling/Database/Scripts/'
sys.path.append(lib)
sys.path.append(lib + 'Subscripts/')
# felix geommetry library
sys.path.append(root_folder + '../geometry/')
sys.path.append(root_folder + '../geometry/w7x/')

warnings.simplefilter("ignore", np.RankWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.filterwarnings("ignore", "KeyboardInterrupt")
warnings.filterwarnings("ignore", "Reloaded modules")

# matplotlib.rcParams.update({'font.size': 20.0})
os.chdir(root_folder + 'libprad/')
