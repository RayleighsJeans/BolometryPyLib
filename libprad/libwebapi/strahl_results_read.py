import netCDF4 as nc
import os
import utils.data as data
fl = ['C_00002t0.000_0.687_1','C_00003t1.220_2.220_1','C_00005t2.421_3.421_1','C_00004t6.154_7.154_1']

os.chdir('C:\\Users\\flr\\rzgshare\\work\\W7X\\python\\git\\codes\\strahl\\result')

dlst = []
for i,fn in enumerate(fl):
    dat = data.data()
    dat.nc = nc.Dataset(fn)
    dat.r = dat.nc.variables['rho_poloidal_grid']
    dat.nimp = dat.nc.variables['impurity_density'][-1,:,:].T
    dat.ne = dat.nc.variables['electron_density'][-1,:]
    dat.te = dat.nc.variables['electron_temperature'][-1,:]
    dat.pi = dat.nc.variables['impurity_radiation'][-1,:,:].T
    #dat.ti = dat.nc.variables['ion_temperature'][-1,:]
    dat.nimpt = np.sum(dat.nimp,1)
    ni = dat.nimp.shape[1]
    tmp = np.zeros(dat.nimp.shape)
    for i in range(ni):
        tmp[:,i] = dat.nimp[:,i]/dat.nimpt
    dat.fa = tmp
    dlst.append(dat)


import utils.colors as colors
clrs = colors.clrs()


i,j= 0,1
fig,axs = plt.subplots(3,1,sharex=True)
axs = axs.flatten()
axs[0].axvline(1.0,ls='--',c='k',label='_nolegend_')
axs[1].axvline(1.0,ls='--',c='k',label='_nolegend_')
axs[2].axvline(1.0,ls='--',c='k',label='_nolegend_')

clrs.reset()
for k in range(ni):
    c = clrs.next()
    axs[0].plot(dlst[i].r,dlst[i].fa[:,k],'-',c=c,lw=2.)
    if k < 6:
      axs[1].plot(dlst[i].r,dlst[i].pi[:,k],'-',c=c,lw=2.)
axs[2].plot(dlst[i].r,dlst[i].ne,'k-')
tax = axs[2].twinx()
tax.plot(dlst[i].r,dlst[i].te,'r-')
leg = []
clrs.reset()
for k in range(ni):
    c = clrs.next()
    axs[0].plot(dlst[j].r,dlst[j].fa[:,k],'--',c=c)
    if k < 6:
      axs[1].plot(dlst[j].r,dlst[j].pi[:,k],'--',c=c)
    leg.append(str(k))
axs[2].plot(dlst[j].r,dlst[j].ne,'k--')
tax.plot(dlst[j].r,dlst[j].te,'r--')

axs[0].set_ylabel('$f_A$ ')
axs[1].set_ylabel('$P_{rad}$ [$W/cm^{-3}s$]')
axs[2].set_ylabel('$n_e$ [$cm^{-3}$]')
tax.set_ylabel('$T_e$ [eV]',color='r')
axs[2].set_xlabel('$rho_{Vol}$')
axs[0].legend(leg)

tax.set_ylim((0.,50.))
axs[0].set_xlim((0.7,1.15))

