
# coding: utf-8

# In[2]:

import os 
os.chdir('C:\\Users\\flr\\rzgshare\\work\\W7X\\python\\git')
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
fsep = False
fsav = True


# Plot Geometry

# In[4]:

import bolometry.w7x.qsb as qsb;reload(qsb); qsb = qsb.QSB()
import geometry.w7x.plotw7x as plotw7x; reload(plotw7x)
qsb.plot_w7x_geom()


# Get LOS Plane

# In[3]:

import bolometry.w7x.qsb as qsb;reload(qsb); qsb = qsb.QSB()
import utils.geom as geom; importlib.reload(geom)
geo = qsb.read_geom(fplt=False)
qsbd = geo['HBCm']
#Define plane with point and normal
v1 = qsbd['vl'][0,:]
v2 = qsbd['vl'][-2,:]
m  = qsbd['e'][-1,0,:]
n = geom.normvec(np.cross(v1,v2))
print(m)
print(n)
#Plot geometry
if False:
    from mayavi import mlab
    mlab.figure()
    mlab.plot3d([m[0],m[0]+v1[0]],[m[1],m[1]+v1[1]],[m[2],m[2]+v1[2]])
    mlab.plot3d([m[0],m[0]+v2[0]],[m[1],m[1]+v2[1]],[m[2],m[2]+v2[2]])
    mlab.plot3d([m[0],m[0]+tmp[0]],[m[1],m[1]+tmp[1]],[m[2],m[2]+tmp[2]],color=(1,0,0))
    mlab.show()


# Get Wall

# In[4]:

import geometry.w7x.meshsrv as wall; reload(wall); wall = wall.WALL()
wc = wall.cut_wall_plane(m,n)


# Optimize Fliedlinetracer Startpoints (O-points)

# In[4]:

import geometry.w7x.fieldlinetracer as flt; importlib.reload(flt)
tracer = flt.TRACER()
IP = [-15.E3 for i in range(5)]
IP.extend([0.,0.])
Itc = np.zeros(5)#[48.,72.,48.,48.,48.]#Ben's case #20180906.40
Icc = np.zeros(10)
tracer.set_currents(IP,Itc,Icc)


# In[6]:

import geometry.w7x.fieldlinetracer as flt; importlib.reload(flt)
tracer = flt.TRACER()
tracer.set_traceinit('test',phi=0.)
tracer.poincare_phi(0.)
fn = 'EJM-test-res0a.pik'
res0a = tracer.trace()


# In[7]:

#fig = tracer.plot_2d(res0a,color='k',shw=False)
#plt.ylim((-1.2,1.2))
#O-points EJM
r  = [ 5.643, 6.237, 5.643, 5.517, 5.517, 5.950, 6.206, 6.266]
z  = [ 0.927, 0.000,-0.927,-0.610, 0.610, 0.000, 0.025, 0.001]


# In[8]:

tracer.set_traceinit('EJM')
tracer.poincare_phi(0.)
fn = 'EJM-EJM-res0b.pik'
res0b = tracer.trace()


# In[9]:

fig = tracer.plot_2d(res0a,color='k',shw=False)
_ = tracer.plot_2d([res0b[0]],fig=fig,color='b',shw=False)
_ = tracer.plot_2d([res0b[1]],fig=fig,color='r',shw=False)
_ = tracer.plot_2d([res0b[2]],fig=fig,color='g',shw=False)
_ = tracer.plot_2d([res0b[3]],fig=fig,color='y',shw=False)
plt.ylim((-1.02,1.02))
plt.xlim((5.35,6.3))


# Check impact of coils currents on equilibrium

# In[125]:

import geometry.w7x.fieldlinetracer as flt; importlib.reload(flt)
tracer = flt.TRACER()
IP = [-15.E3 for i in range(5)]
IP.extend([0.,0.])
Itc = np.zeros(5)
Icc = np.zeros(10)
tracer.set_currents(IP,Itc,Icc)


# In[126]:

tracer.set_traceinit('O-points')
tracer.poincare_phi(108.,step=0.01,steps=500,single=True)
res = tracer.trace()


# In[127]:

if fsav:
    fn = '../data/FLT/EIM-PHI.flt'
    _ = tracer.save_trace(res,fn+'rc')
    _ = tracer.save_poincare(res,fn)
    #tst = tracer.load_trace(fn+'rc')


# Projection to QSB plane

# In[128]:

tracer.set_traceinit('O-points')
tracer.poincare_plane(m,n,step=0.01,steps=500)
res2 = tracer.trace()


# In[129]:

if fsav:
    fn = '../data/FLT/EIM-QSB.flt'
    _ = tracer.save_trace(res2,fn+'rc')
    _ = tracer.save_poincare(res2,fn)


# In[130]:

if fsep:
    sep2,rsep2,psep2 = tracer.get_separatrix(m=m,n=n,step=0.01,steps=100)


# In[131]:

import geometry.w7x.fieldlinetracer as flt; importlib.reload(flt)
tracer = flt.TRACER()
fig = plt.figure(figsize=(10.3,10.3))
fig = tracer.plot_2d(res,phi=[100.,120.],color='k',fig=fig,shw=False)
fig = tracer.plot_2d(res2,phi=[100.,120.],color='r',fig=fig,shw=False)
plt.plot([],'k',label='$\Phi=108$')
plt.plot([],'r',label='QSB-Plane')
plt.title('EJM at QSB-Plane')
plt.legend()
plt.ylim((-0.6,0.6))
plt.show()


# Trim Coils

# In[132]:

import geometry.w7x.fieldlinetracer as flt; importlib.reload(flt)
tracer = flt.TRACER()
IP = [-15.E3 for i in range(5)]
IP.extend([0.,0.])
#Ben's case #20180906.40 [48.,72.,48.,72.]
#Itc = np.zeros(5)
Itc = np.array([1.,1.,1.,1.,1.])*1.E3
Icc = np.zeros(10)
tracer.set_currents(IP,Itc,Icc)


# In[133]:

tracer.set_traceinit('O-points')
tracer.poincare_plane(m,n,step=0.01,steps=500)
res3 = tracer.trace()


# In[134]:

if fsav:
    fn = '../data/FLT/EIM-QSB-ITC1.flt'
    _ = tracer.save_trace(res3,fn+'rc')
    _ = tracer.save_poincare(res3,fn)


# In[135]:

if fsep:
    sep3,rsep3,psep3 = tracer.get_separatrix(m=m,n=n,step=0.01,steps=1000)


# In[136]:

fig = plt.figure(figsize=(10.3,10.3))
fig = tracer.plot_2d(res2,phi=[100.,120.],color='k',fig=fig,shw=False)
fig = tracer.plot_2d(res3,phi=[100.,120.],color='r',fig=fig ,shw=False)
plt.plot([],'k',label='ITC=0kA')
plt.plot([],'r',label='ITC=1kA')
plt.title('Trim Coil Scan EJM at QSB-Plane')
plt.legend()
plt.ylim((-0.6,0.6))
plt.show()


# In[137]:

if fsep:
    fig = tracer.plot_2d([sep2],phi=[100.,120.],color='gray',ms=5,fig=None ,shw=False)
    fig = tracer.plot_2d([sep3],phi=[100.,120.],color='orange',ms=5,fig=fig ,shw=False)
    plt.plot([],'k',label='ITC=0kA')
    plt.plot([],'r',label='ITC=1kA')
    plt.title('Trim Coil Scan EJM at QSB-Plane')
    plt.legend()
    plt.ylim((-0.6,0.6))
    plt.show()


# Control Coils

# In[138]:

import geometry.w7x.fieldlinetracer as flt; importlib.reload(flt)
tracer = flt.TRACER()
IP = [-15.E3 for i in range(5)]
IP.extend([0.,0.])
Itc = np.zeros(5)
Icc = np.ones(10)*1.0E3
tracer.set_currents(IP,Itc,Icc)


# In[139]:

tracer.set_traceinit('O-points')
tracer.poincare_plane(m,n,step=0.01,steps=500)
res4 = tracer.trace()


# In[140]:

if fsav:
    fn = '../data/FLT/EIM-QSB-ICC1.flt'
    _ = tracer.save_trace(res4,fn+'rc')
    _ = tracer.save_poincare(res4,fn)


# In[141]:

if fsep:
    sep4,rsep4,psep4 = tracer.get_separatrix(m=m,n=n,step=0.01,steps=1000)


# In[142]:

import geometry.w7x.fieldlinetracer as flt; importlib.reload(flt)
tracer = flt.TRACER()
IP = [-15.E3 for i in range(5)]
IP.extend([0.,0.])
Itc = np.zeros(5)
Icc = np.ones(10)*2.0E3
tracer.set_currents(IP,Itc,Icc)


# In[143]:

tracer.set_traceinit('O-points')
tracer.poincare_plane(m,n,step=0.01,steps=500)
res5 = tracer.trace()


# In[144]:

if fsav:
    fn = '../data/FLT/EIM-QSB-ICC2.flt'
    _ = tracer.save_trace(res5,fn+'rc')
    _ = tracer.save_poincare(res5,fn)


# In[145]:

tracer.set_traceinit('O-points')
tracer.poincare_plane(m,n,step=0.001,steps=500)
#fn = '../data/FLT/EJM-PHI-ICC2-res5.flt'
res5a = tracer.trace()


# In[147]:

tracer.set_traceinit('O-points')
tracer.poincare_plane(m,n,step=0.0001,steps=500)
#fn = '../data/FLT/EJM-PHI-ICC2-res5.flt'
res5b = tracer.trace()


# In[148]:

if fsep:
    sep5,rsep5,psep5 = tracer.get_separatrix(m=m,n=n,step=0.01,steps=1000)


# In[149]:

import geometry.w7x.fieldlinetracer as flt; importlib.reload(flt)
tracer = flt.TRACER()
fig = plt.figure(figsize=(10.3,10.3))
fig = tracer.plot_2d(res2,phi=[100.,120.],fig=fig,color='k',shw=False)
_ = tracer.plot_2d(res4,phi=[100.,120.],fig=fig,color='b',shw=False)
_ = tracer.plot_2d(res5,phi=[100.,120.],fig=fig,color='r',shw=False)
plt.plot([],'k',label='ICC=0kA')
plt.plot([],'b',label='ICC=1kA')
plt.plot([],'r',label='ICC=2kA')
plt.title('Control Coil Scan EJM at QSB-Plane')
plt.legend()
plt.ylim([-0.6,0.6])
plt.show()


# In[150]:

fig = plt.figure(figsize=(10.3,10.3))
fig = tracer.plot_2d(res5,phi=[100.,120.],fig=fig,color='k',shw=False)
_ = tracer.plot_2d(res5a,phi=[100.,120.],fig=fig,color='b',shw=False)
_ = tracer.plot_2d(res5b,phi=[100.,120.],fig=fig,color='r',shw=False)
plt.plot([],'k',label='ICC=2kA')
plt.plot([],'b',label='ICC=2kA (step/10)')
plt.plot([],'r',label='ICC=2kA (step/10)')
plt.title('Control Coil Scan EJM at QSB-Plane')
plt.legend()
plt.ylim([-0.6,0.6])
plt.show()


# In[151]:

if fsep:
    fig = tracer.plot_2d([sep2],phi=[100.,120.],color='k',shw=False)
    _ = tracer.plot_2d([sep4],phi=[100.,120.],fig=fig,color='b',shw=False)
    _ = tracer.plot_2d([sep5],phi=[100.,120.],fig=fig,color='r',shw=False)
    plt.plot([],'k',label='ICC=0kA')
    plt.plot([],'b',label='ICC=1kA')
    plt.plot([],'r',label='ICC=2kA')
    plt.title('Control Coil Scan EJM at QSB-Plane')
    plt.legend()
    plt.ylim([-0.6,0.6])
    plt.show()


# Plasma current on axis

# In[5]:

import geometry.w7x.fieldlinetracer as flt; importlib.reload(flt)
tracer = flt.TRACER()
IP = [-15.E3 for i in range(5)]
IP.extend([0.,0.])
Itc = np.zeros(5)
Icc = np.zeros(10)
tracer.set_currents(IP,Itc,Icc)
tracer.set_plasmacurrent(2.E3,config='EJM')
#tracer.plot_axis(config='EJM')


# In[6]:

tracer.set_traceinit('O-points')
tracer.poincare_plane(m,n,step=0.01,steps=500)
res6 = tracer.trace()


# In[7]:

fn = '../data/FLT/EIM-QSB-IPC2.flt'
if fsav:
    _ = tracer.save_trace(res6,fn+'rc')
    _ = tracer.save_poincare(res6,fn)


# In[8]:

if fsep:
    sep6,rsep6,psep6 = tracer.get_separatrix(m=m,n=n,step=0.01,steps=1000)


# In[ ]:

import geometry.w7x.fieldlinetracer as flt; importlib.reload(flt)
tracer = flt.TRACER()
fig = plt.figure(figsize=(10.3,10.3))
fig = tracer.plot_2d(res2,phi=[100.,120.],fig=fig,color='k',shw=False)
_ = tracer.plot_2d(res6,phi=[100.,120.],fig=fig,color='r',shw=False)
plt.plot([],'k',label='IPC=0kA')
plt.plot([],'r',label='IPC=2kA')
plt.title('Poincare at $\Phi = 108\degree$')
plt.legend()
plt.ylim([-0.6,0.6])
plt.show()


# In[ ]:

if fsep:
    fig = tracer.plot_2d([sep2],phi=[100.,120.],color='k',shw=False)
    _ = tracer.plot_2d([sep6],phi=[100.,120.],fig=fig,color='r',shw=False)
    plt.plot([],'k',label='IPC=0kA')
    plt.plot([],'r',label='IPC=2kA')
    plt.title('Poincare at $\Phi = 108\degree$')
    plt.legend()
    plt.ylim([-0.6,0.6])
    plt.show()


# Add bolometer

# In[ ]:

geo = qsb.read_geom(fplt=False)
#print(geo['HBCm']['r'].shape)
qsbd = geo['HBCm']
#Define plane with point and normal
v1 = qsbd['vl'][0,:]
v2 = qsbd['vl'][-2,:]
m  = qsbd['e'][-1,0,:]
n = geom.normvec(np.cross(v1,v2))


# In[ ]:

fig = plt.figure(figsize=(10.3,10.3))
fig = wall.plot_wall(wc,phi=[100.,120.],fig=fig,shw=False,fig=None)
fig = tracer.plot_2d(res2,phi=[100.,120.],fig=fig,color='k')
_ = tracer.plot_2d(res5,phi=[100.,120.],fig=fig,color='r')
plt.plot([],'k',label='ICC=0kA')
plt.plot([],'r',label='ICC=2kA')
plt.legend()
qsbd = geo['HBCm']
_ = qsb.plot_los_cycl('HBCm',fig=fig,geo=qsbd,lw=0.5,annot=True)
qsbd = geo['VBCr']
_ = qsb.plot_los_cycl('VBCr',fig=fig,geo=qsbd,lw=0.5,color='b',annot=True)
qsbd = geo['VBCl']
_ = qsb.plot_los_cycl('VBCl',fig=fig,geo=qsbd,lw=0.5,color='g',annot=True)
fig.gca().set_ylim([-1.,1.])
plt.show()


# ### Configuration Scan

# EIM - Standard

# In[11]:

fload = False
fsav = True


# In[12]:

import geometry.w7x.fieldlinetracer as flt; importlib.reload(flt)
tracer = flt.TRACER()
IP = [-15.E3 for i in range(5)]
IP.extend([0.,0.])
Itc = np.zeros(5)
Icc = np.zeros(10)
tracer.set_currents(IP,Itc,Icc)
tracer.set_traceinit('O-points')
tracer.poincare_plane(m,n,step=0.01,steps=500)
if not(fload):
    res2 = tracer.trace()


# In[13]:

fn = '../data/FLT/EJM-QSB.flt'
if fload:
    res2 = tracer.load(fn+'rc')
if fsav:
    _ = tracer.save_trace(res2,fn+'rc')
    _ = tracer.save_poincare(res2,fn)


# In[14]:

if fsep:
    sep2,rsep2,psep2 = tracer.get_separatrix(m=m,n=n,step=0.001,steps=1000)


# In[17]:

fig = plt.figure(figsize=(10.3,10.3))
fig = wall.plot_wall(wc,phi=[100.,120.],fig=fig,shw=False)
fig = tracer.plot_2d(res2,phi=[100.,120.],fig=fig,color='k')
plt.plot([],'k',label='EIM - Standard')
plt.title('Standard (EIM) at $\Phi = 108\degree$')
plt.legend()
qsbd = geo['HBCm']
_ = qsb.plot_los_cycl('HBCm',fig=fig,geo=qsbd,lw=0.5,color='b',annot=True)
qsbd = geo['VBCr']
_ = qsb.plot_los_cycl('VBCr',fig=fig,geo=qsbd,lw=0.5,color='b',annot=True)
qsbd = geo['VBCl']
_ = qsb.plot_los_cycl('VBCl',fig=fig,geo=qsbd,lw=0.5,color='g',annot=True)
fig.gca().set_ylim([-1.,1.])
plt.savefig(fn[:-3]+'.png')
plt.show()


# DBM - Low iota

# In[18]:

from mayavi import mlab
import geometry.w7x.fieldlinetracer as flt; importlib.reload(flt)
tracer = flt.TRACER()
IP = [-12222.22 for i in range(5)]
IP.extend([-9166.67,-9166.67])
Itc = np.zeros(5)
Icc = np.zeros(10)
tracer.set_currents(IP,Itc,Icc)
tracer.set_traceinit('O-points')
tracer.poincare_plane(m,n,step=0.001,steps=500)
if not(fload):
    res7 = tracer.trace()


# In[19]:

fn = '../data/FLT/DBM-QSB.flt'
if fload:
    res7 = tracer.load(fn+'rc')
if fsav:
    _ = tracer.save_trace(res7,fn+'rc')
    _ = tracer.save_poincare(res7,fn)


# In[20]:

if fsep:
    sep7,rsep7,psep7 = tracer.get_separatrix(m=m,n=n,step=0.001,steps=1000)


# In[21]:

fig = plt.figure(figsize=(10.3,10.3))
fig = wall.plot_wall(wc,phi=[100.,120.],fig=fig,shw=False)
fig = tracer.plot_2d(res7,phi=[100.,120.],color='k',fig=fig,shw=False)
plt.plot([],'r',label='DBM - Low Iota')
plt.title('Low Iota (DBM) at $\Phi = 108\degree$')
plt.legend()
plt.ylim([-0.6,0.6])
qsbd = geo['HBCm']
_ = qsb.plot_los_cycl('HBCm',fig=fig,geo=qsbd,lw=0.5,color='r',annot=True)
qsbd = geo['VBCr']
_ = qsb.plot_los_cycl('VBCr',fig=fig,geo=qsbd,lw=0.5,color='b',annot=True)
qsbd = geo['VBCl']
_ = qsb.plot_los_cycl('VBCl',fig=fig,geo=qsbd,lw=0.5,color='g',annot=True)
fig.gca().set_ylim([-1.,1.])
plt.savefig(fn[:-3]+'.png')
plt.show()


# FTM - High iota

# In[22]:

from mayavi import mlab
import geometry.w7x.fieldlinetracer as flt; importlib.reload(flt)
tracer = flt.TRACER()
IP = [-14814.81 for i in range(5)]
IP.extend([10222.22 ,10222.22 ])
Itc = np.zeros(5)
Icc = np.zeros(10)
tracer.set_currents(IP,Itc,Icc)
tracer.set_traceinit('O-points')
tracer.poincare_plane(m,n,step=0.001,steps=500)
if not(fload):
    res8 = tracer.trace()


# In[23]:

fn = '../data/FLT/FTM-QSB.flt'
if fload:
    res8 = tracer.load(fn+'rc')
if fsav:
    _ = tracer.save_trace(res8,fn+'rc')
    _ = tracer.save_poincare(res8,fn)


# In[24]:

if fsep:
    sep8,rsep8,psep8 = tracer.get_separatrix(m=m,n=n,step=0.001,steps=1000)


# In[25]:

fig = plt.figure(figsize=(10.3,10.3))
fig = wall.plot_wall(wc,phi=[100.,120.],fig=fig,shw=False)
fig = tracer.plot_2d(res8,phi=[100.,120.],color='k',fig=fig,shw=False)
plt.plot([],'r',label='FTM - High Iota')
plt.title('High Iota (FTM) at $\Phi = 108\degree$')
plt.legend()
plt.ylim([-0.6,0.6])
qsbd = geo['HBCm']
_ = qsb.plot_los_cycl('HBCm',fig=fig,geo=qsbd,lw=0.5,color='r',annot=True)
qsbd = geo['VBCr']
_ = qsb.plot_los_cycl('VBCr',fig=fig,geo=qsbd,lw=0.5,color='b',annot=True)
qsbd = geo['VBCl']
_ = qsb.plot_los_cycl('VBCl',fig=fig,geo=qsbd,lw=0.5,color='g',annot=True)
fig.gca().set_ylim([-1.,1.])
plt.savefig(fn[:-3]+'.png')
plt.show()


# FMM02 - Iota Scan: High-Iota --> Standard

# In[26]:

from mayavi import mlab
import geometry.w7x.fieldlinetracer as flt; importlib.reload(flt)
tracer = flt.TRACER()
IP = [-13423 for i in range(5)]
IP.extend([3544,3544])
Itc = np.zeros(5)
Icc = np.zeros(10)
tracer.set_currents(IP,Itc,Icc)
tracer.set_traceinit('O-points')
tracer.poincare_plane(m,n,step=0.001,steps=500)
if not(fload):
    res9 = tracer.trace()


# In[27]:

fn = '../data/FLT/FMM02-QSB.flt'
if fload:
    res9 = tracer.load(fn+'rc')
if fsav:
    _ = tracer.save_trace(res9,fn+'rc')
    _ = tracer.save_poincare(res9,fn)


# In[28]:

if fsep:
    sep9,rsep9,psep9 = tracer.get_separatrix(m=m,n=n,step=0.001,steps=1000)


# In[29]:

fig = plt.figure(figsize=(10.3,10.3))
fig = wall.plot_wall(wc,phi=[100.,120.],fig=fig,shw=False)
fig = tracer.plot_2d(res9,phi=[100.,120.],fig=fig,color='k',shw=False)
plt.plot([],'r',label='FMM02 - Iota Scan (Low-Std)')
plt.title('Iota Scan Low-Std (FMM02) at $\Phi = 108\degree$')
plt.legend()
plt.ylim([-0.6,0.6])
qsbd = geo['HBCm']
_ = qsb.plot_los_cycl('HBCm',fig=fig,geo=qsbd,lw=0.5,color='r',annot=True)
qsbd = geo['VBCr']
_ = qsb.plot_los_cycl('VBCr',fig=fig,geo=qsbd,lw=0.5,color='b',annot=True)
qsbd = geo['VBCl']
_ = qsb.plot_los_cycl('VBCl',fig=fig,geo=qsbd,lw=0.5,color='g',annot=True)
fig.gca().set_ylim([-1.,1.])
plt.savefig(fn[:-3]+'.png')
plt.show()


# KJM - High mirror

# In[30]:

from mayavi import mlab
import geometry.w7x.fieldlinetracer as flt; importlib.reload(flt)
tracer = flt.TRACER()
IP = [-14400, -14000, -13333.33, -12666.67, -12266.67]
IP.extend([0.,0.])
Itc = np.zeros(5)
Icc = np.zeros(10)
tracer.set_currents(IP,Itc,Icc)
tracer.set_traceinit('O-points')
tracer.poincare_plane(m,n,step=0.001,steps=500)
if not(fload):
    res10 = tracer.trace()


# In[31]:

fn = '../data/FLT/KJM-QSB.flt'
if fload:
    res10 = tracer.load(fn+'rc')
if fsav:
    _ = tracer.save_trace(res10,fn+'rc')
    _ = tracer.save_poincare(res10,fn)


# In[32]:

if fsep:
    sep10,rsep10,psep10 = tracer.get_separatrix(m=m,n=n,step=0.001,steps=1000)


# In[33]:

fig = plt.figure(figsize=(10.3,10.3))
fig = wall.plot_wall(wc,phi=[100.,120.],fig=fig,shw=False)
fig = tracer.plot_2d(res8,phi=[100.,120.],fig=fig,color='k',shw=False)
plt.plot([],'r',label='KJM - High Mirror')
plt.title('High Mirror (KJM) at $\Phi = 108\degree$')
plt.legend()
plt.ylim([-0.6,0.6])
qsbd = geo['HBCm']
_ = qsb.plot_los_cycl('HBCm',fig=fig,geo=qsbd,lw=0.5,color='r',annot=True)
qsbd = geo['VBCr']
_ = qsb.plot_los_cycl('VBCr',fig=fig,geo=qsbd,lw=0.5,color='b',annot=True)
qsbd = geo['VBCl']
_ = qsb.plot_los_cycl('VBCl',fig=fig,geo=qsbd,lw=0.5,color='g',annot=True)
fig.gca().set_ylim([-1.,1.])
plt.savefig(fn[:-3]+'.png')
plt.show()


# FOM - Limiter

# In[34]:

from mayavi import mlab
import geometry.w7x.fieldlinetracer as flt; importlib.reload(flt)
tracer = flt.TRACER()
IP = [-15000, -15000, -15000, -15000, -15000]
IP.extend([-370.,-370.])
Itc = np.zeros(5)
Icc = np.zeros(10)
tracer.set_currents(IP,Itc,Icc)
#tracer.set_traceinit('O-points')
tracer.poincare_plane(m,n,step=0.01,steps=500)
if not(fload):
    res11 = tracer.trace()


# In[35]:

fn = '../data/FLT/FOM-QSB.flt'
if fload:
    res11 = tracer.load(fn+'rc')
if fsav:
    _ = tracer.save_trace(res11,fn+'rc')
    _ = tracer.save_poincare(res11,fn)


# In[36]:

if fsep:
    sep11,rsep11,psep11 = tracer.get_separatrix(m=m,n=n,step=0.001,steps=1000)


# In[37]:

fig = plt.figure(figsize=(10.3,10.3))
fig = wall.plot_wall(wc,phi=[100.,120.],fig=fig,shw=False)
fig = tracer.plot_2d(res11,phi=[100.,120.],fig=fig,color='k',shw=False)
plt.plot([],'r',label='FOM - Limiter')
plt.title('Limiter (FOM) at $\Phi = 108\degree$')
plt.legend()
plt.ylim([-0.6,0.6])
qsbd = geo['HBCm']
_ = qsb.plot_los_cycl('HBCm',fig=fig,geo=qsbd,lw=0.5,color='r',annot=True)
qsbd = geo['VBCr']
_ = qsb.plot_los_cycl('VBCr',fig=fig,geo=qsbd,lw=0.5,color='b',annot=True)
qsbd = geo['VBCl']
_ = qsb.plot_los_cycl('VBCl',fig=fig,geo=qsbd,lw=0.5,color='g',annot=True)
fig.gca().set_ylim([-1.,1.])
plt.savefig(fn[:-3]+'.png')
plt.show()


# Visualize QSB with islands (3D)

# In[ ]:

from mayavi import mlab
reload(geom)
import geometry.w7x.fieldlinetracer as flt; importlib.reload(flt)
tracer = flt.TRACER()
IP = [15.E3 for i in range(5)]
IP.extend([0.,0.])
Itc = np.zeros(5)
Icc = np.zeros(10)
tracer.set_currents(IP,Itc,Icc)

R  = np.array([ 5.643, 6.237, 5.643, 5.517, 5.517])#, 5.950]#, 6.206, 6.266]
z  = np.array([ 0.927, 0.000,-0.927,-0.610, 0.610])#, 0.000]#, 0.025, 0.025]
x,y,z = geom.cylindertocartesian(R,z,np.zeros(R.shape))
x,y,z = geom.rotation(x,y,z,phi=144-72,deg=True)
print(x,y,z)


# In[ ]:

tracer.set_traceinit('external',x1=x,x2=y,x3=z)
#tracer.set_machine([165,166,167,168,169])
#print(tracer.machine.meshedModelsIds)
tracer.fieldline_span(108)
flf = tracer.trace()
#tracer.config.inverseField = True
#flr = tracer.trace()
#tracer.config.inverseField = False
#tracer.set_machine([164])


# In[ ]:

lcol = [(0, 0, 255/255.),(255/255., 153/255., 0),(0, 153/255., 0),(204/255., 51/255., 153/255.),(255/255., 255/255., 0),(0,0,0),(0,0,0),(0,0,0)]
#b,o,g,l,y
fig = mlab.figure(bgcolor = (1.0, 1.0, 1.0))
for i in range(5):
    mlab.plot3d(flf[0].lines[i].vertices.x1,                 flf[0].lines[i].vertices.x2,                 flf[0].lines[i].vertices.x3,                 tube_radius=3.E-2,tube_sides=4,color=lcol[i],figure=fig)
if False:
    for i in range(5):
        mlab.plot3d(flr[0].lines[i].vertices.x1,                     flr[0].lines[i].vertices.x2,                     flr[0].lines[i].vertices.x3,                     tube_radius=3.E-2,tube_sides=4,color=lcol[i],figure=fig)


# In[ ]:

import geometry.w7x.vmec as vmec; importlib.reload(vmec)
vid = 'w7x/1000_1000_1000_1000_+0000_+0000/01/00jh_l' #std. EJM
fst,phi = vmec.prep_toroidal_fs(vid,phi=[144-72.,144+36.],nt=100,cart=True)#,scl=1.4,nr=7)


# In[ ]:

x,y,z = vmec.prep_surf_3D(fst)
#vmec.plot_fst_3D(fst,cm='bone')
xs = x[:,-1,:]
ys = y[:,-1,:]
zs = z[:,-1,:]
_ = mlab.mesh(xs,ys,zs,color=(200/255., 200/255., 200/255.),opacity=0.4,figure=fig)


# In[127]:

if True:
    import geometry.w7x.plotw7x as plotw7x; importlib.reload(plotw7x)
    fig = plotw7x.plot_geometry(fig=fig,hmco=[],mdi=[2],mba=[2],hmfs=[])
    #engine = mlab.get_engine()
    #engine.save_visualization('w7x-div.mv')


# In[56]:

if False:
    fig = mlab.figure()
    engine = mlab.get_engine()
    engine.load_visualization('w7x-div.mv')
    fig = mlab.gcf()


# In[ ]:

lcol2 = [(0,0,1),(0,1,0),(1,0,0),(0,0,1),(0,1,0),(1,0,0),(0,0,0)]
lcol2 = [(0,0,1),(0,0,1),(0,0,1),(1,0,0),(1,0,0),(1,0,0),(1,0,0)]
for i,key in enumerate(['HBCm','VBCr','VBCl','AEJDIV','AEJCOR','AELBOT','AELTOP']):
    qsbd = geo[key]
    for j in range(qsbd['nl']-1):
        x = qsbd['vlos'][j,:,0]
        y = qsbd['vlos'][j,:,1]
        z = qsbd['vlos'][j,:,2]
        if key in ['AEJDIV','AEJCOR','AELBOT','AELTOP']:
            x,y,z = geom.rotation(x,y,z,phi =-72.,deg=True)
        mlab.plot3d(x,y,z,                    line_width=0.1,tube_radius=0.005,color=lcol2[i],figure=fig)


# In[ ]:

tmp = tracer.save_poincare(res2,phi=[phi-dphi,phi+dphi])
mlab.points3d(tmp['x'], tmp['y'], tmp['z'], color = (0.,0.,0.), scale_factor=0.1)


# In[ ]:

mlab.show()


# Plot 2D geometry

# In[46]:

import geometry.w7x.meshsrv as wall; reload(wall); wall = wall.WALL()
wc = wall.cut_wall_phi(0.)
wc2 = wall.cut_wall_plane(m,n)


# In[47]:

tracer.set_traceinit('O-points')
ns = 10000
tracer.poincare_phi(108.,step=0.01,steps=ns,single=True)
res6 = tracer.trace()
tracer.poincare_phi(108.-36.,step=0.01,steps=ns,single=True)
res7 = tracer.trace()
tracer.poincare_phi(108.+36.,step=0.01,steps=ns,single=True)
res8 = tracer.trace()


# In[48]:

import geometry.w7x.fieldlinetracer as flt; importlib.reload(flt)
tracer = flt.TRACER()
tracer.set_traceinit('O-points')
tracer.poincare_plane(m,n,step=0.01,steps=ns)
res9 = tracer.trace()


# In[49]:

import geometry.w7x.fieldlinetracer as flt; importlib.reload(flt)
tracer = flt.TRACER()
lcol = [(0,0,0),(0,0,0),(0,0,0),(0, 0, 255/255.),(255/255., 153/255., 0),(0, 153/255., 0),(204/255., 51/255., 153/255.),(255/255., 255/255., 0)]
fig = tracer.plot_2d(res9[0:8],phi=[100.,120.],color=lcol)
fig = wall.plot_wall(wc2,phi=[100.,120.],color='k',lw=1.5,fig=fig)
fig.gca().set_aspect('equal')
plt.ylim((-1.05,1.05))
fig = tracer.plot_2d(res7[0:8],color=lcol)
fig = wall.plot_wall(wc,color='k',lw=1.5,fig=fig)
plt.ylim((-1.3,1.3))
fig = tracer.plot_2d(res8[0:8],color=lcol)
fig = wall.plot_wall(wc,color='k',lw=1.5,fig=fig)
plt.ylim((-1.3,1.3))


# In[ ]:

geo = qsb.read_geom(fplt=False)
fig = plt.figure(figsize=(10.3,10.3))
fig = wall.plot_wall(wc2,phi=[100.,120.],color='k',lw=1.5,fig=fig,shw=False)
fig = tracer.plot_2d(res9[0:8],phi=[100,120],color=lcol,ms=0.5,fig=fig)
fig = qsb.plot_los_cycl('HBCm',fig=fig,geo=geo['HBCm'],color='k',lw=0.5)
fig = qsb.plot_los_cycl('VBCr',fig=fig,geo=geo['VBCr'],color='k',lw=0.5)
fig = qsb.plot_los_cycl('VBCl',fig=fig,geo=geo['VBCl'],didx=[i+15 for i in range(10)],color='k',lw=0.5)


# In[ ]:



