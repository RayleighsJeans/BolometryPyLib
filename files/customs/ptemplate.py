""" *********************************************************************** """
# header
import matplotlib.pyplot as p
import matplotlib
import json
matplotlib.use('Qt5agg')
#from matplotlib.pyplot import cm
#from mpl_toolkits.mplot3d import Axes3D
# eo header
""" *********************************************************************** """

file = r'W:\Documents\LABGIT\IDL2PY-PORT\results\TMP\INVERSION\errors.json'
loc = r'W:\Documents\LABGIT\IDL2PY-PORT\results\TMP\tmp.pdf'

with open(file, 'r') as infile:
    dct = json.load(infile)
infile.close()

# importante
dct = dct['values']['data']

plabels = ['rel.error', 'radial error', 'z_error', #axp
           'file size'] #twp
axlabels = ['EIM_000 interpolation scheme', # title
            'interpolation points', 'accuracy', #ax labels
            'file size'] #twax labels
dictnm = ['acc', 'rel_err', #axp1
          'acc', 'R_err', #axp2
          'acc', 'Z_err', #axp3
          'acc', 'sizes'] #twp4
          
"""
STANDARDIZED
"""

p.ioff()
fig = p.figure()
ax = fig.add_subplot(111)
#ax2 = fig.add_subplot(212)
fig.subplots_adjust(right=0.87)

ax.set_xlabel(axlabels[1])
ax.set_ylabel(axlabels[2])
ax.set_title(axlabels[0])

p1, = ax.plot(dct[dictnm[0]],
              dct[dictnm[1]],
              c='cyan',linewidth=0.7,linestyle=':',
              marker='^', markersize=5.,
              label=plabels[0])

p2, = ax.plot(dct[dictnm[2]],
              dct[dictnm[3]],
              c='blue',linewidth=0.7,linestyle='-',
              marker='+', markersize=5.,
              label=plabels[1])

p3, = ax.plot(dct[dictnm[4]],
              dct[dictnm[5]],
              c='green',linewidth=0.7,linestyle='-.',
              marker='*', markersize=5.,
              label=plabels[2])

twinax = ax.twinx()
twinax.spines['right'].set_position(('axes', 1.0))
twinax.grid(b=True, which='major', linestyle='-.')
ax.grid(b=True, which='major', linestyle='-.')

tp4, = twinax.plot(dct[dictnm[6]],
                   dct[dictnm[7]],
                   c='r',linewidth=0.7,linestyle='--',
                   marker='x', markersize=5.,
                   label=plabels[3])

twinax.yaxis.label.set_color(tp4.get_color())
twinax.spines["right"].set_edgecolor(tp4.get_color())
twinax.tick_params(axis='y', colors=tp4.get_color())
twinax.set_ylabel(axlabels[3])

ax.legend(loc=3)

fig.savefig(loc)
p.close(fig)
p.close("all")

