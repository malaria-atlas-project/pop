from __future__ import division
from attribute_pops import *
import tables as tb
import numpy as np
from numpy import pi
import pylab as pl
import matplotlib
import pymc as pm
from pointprocess import *
from pp_step_methods import *
from mpl_toolkits import basemap
from matplotlib.nxutils import points_inside_poly
import os

hf = tb.openFile('admin_units.hdf5')
vertices = hf.root.vertices
extents = hf.root.info.cols.extents
country_names = hf.root.info.cols.ADM0_NAME
c=hf.root.info.cols

# Pick a big admin unit in Angola
i=84979

# Control parameters. [size, area]
mu_size = np.log(1000)
tau_size = .7
cutoff_size=5000
def size_to_a(size):
    return np.sqrt(size)/np.sqrt(1000000)
tau_a = 1

lam = pm.Exponential('lam',1e-8,6e3)
int_params = {  'mu_size':mu_size,
                'tau_size':tau_size,
                'cutoff_size':cutoff_size,
                'size_to_a':size_to_a,
                'tau_a':tau_a,
                'admin_poly':vertices[i], 
                'lam':lam}
bb = (np.hstack([extents[i][0,:2]*np.pi/180, 0, 0]), np.hstack([extents[i][1,:2]*np.pi/180, cutoff_size, size_to_a(cutoff_size)*.5]))                 

def area_of_region(bb, r, N):
    coords = np.random.random(size=(N,2))
    
    for i in xrange(2):
        coords[:,i]*=bb[1][i]-bb[0][i]
        coords[:,i]+=bb[0][i]
    
    ins=inregion(coords[:,0], coords[:,1], r)
    return np.sum(ins) / N * np.prod(bb[1]-bb[0])

P_size_OK = np.sum(pm.rlognormal(mu_size, tau_size, size=100000)<cutoff_size)/100000    
mu = P_size_OK * area_of_region((bb[0][:2],bb[1][:2]), int_params['admin_poly'], 100000)
    
    

def log_intensity(x, mu_size, tau_size, cutoff_size, size_to_a, tau_a, admin_poly, lam):
    """
    Log-intensity function for settlement location, size and area.
    """
    out = 0
    if len(x)==0:
        return out-lam*mu
    
    # Check that no size is too big
    sizes=np.array([xx[3] for xx in x])
    if np.any(sizes > cutoff_size):
        # print 'Size too big', sizes, cutoff_size, sizes.__class__, sizes>cutoff_size
        return -np.Inf

    # Check that all locations are in the admin unit.
    lons=[xx[0] for xx in x]
    lats=[xx[1] for xx in x]
    if not np.all(inregion(lons,lats,admin_poly)):
        # print 'Points not in polygon: ', lons, lats, admin_poly
        return -np.Inf
    
    # Sum up intensities of sizes and areas.
    for xx in x:
        size = xx[2]
        area = xx[3]
        out += pm.lognormal_like(size, mu_size, tau_size)
        out += pm.lognormal_like(area, np.log(size_to_a(size)), tau_a)
        
    # Return with overall scaling factor.
    return out + np.log(lam)*len(x)-lam*mu
   
PP = PoissonProcess('PP', log_intensity, dims=4, bounding_box=bb, **int_params)
BD = PPBirthDeathMetropolis(PP, birth_prob=.5)
while(len(PP.value)==0):
    BD.step()

hf.close()

# tot_int = integrate_intensity(PP,100000)[0]

# 
# Ns = [sum(unkilled_sizes[np.random.randint(0, len(unkilled_sizes), size=pm.rpoisson(tot_int))]) for i in xrange(1000)]
# def N_pot():
    
tot_N_logmean = np.log(1e5)
tot_N_tau = 10

@pm.deterministic
def N(PP=PP):
    return sum([pp_now[2] for pp_now in PP])
    
@pm.potential
def N_pot(N=N,mu=tot_N_logmean, tau=tot_N_tau):
    return pm.lognormal_like(N,mu,tau)


M = pm.MCMC()
M.use_step_method(PPBirthDeathMetropolis, PP, birth_prob=.5)
M.use_step_method(PPMoveMetropolis, PP, exp_number_moved = 10, move_sig=1)

def show(PP, N, name, path):
    matplotlib.interactive(False)
    b = basemap.Basemap(llcrnrlon=PP.bounding_box[0][0], llcrnrlat=PP.bounding_box[0][1],
                        urcrnrlon=PP.bounding_box[1][0], urcrnrlat=PP.bounding_box[1][1])
    
    max_pop=np.max([np.max([pp_now_now[2] for pp_now_now in pp_now]) for pp_now in PP.trace()])
                        
    def drawcircle(lon, lat, size, area):
        N_per_circle = 100
        r=np.sqrt(area/pi)*pi/180
        col = (.8*(size/max_pop+.1), .1*(size/max_pop+.1), .2*(size/max_pop+.1))
        x = lon + np.cos(np.linspace(0,2.*pi,N_per_circle))*r
        y = lat + np.sin(np.linspace(0,2.*pi,N_per_circle))*r
        pl.fill(x,y,facecolor=col,alpha=.8)
        
    indices = np.array(np.linspace(0,len(PP.trace())-1,N),dtype=int)

    for i in xrange(N):
        pl.clf()
        b.drawcoastlines(color='r')
        b.drawcountries(color='r')
        
        # b.drawmeridians()
        # b.drawparallels()
        
        for r in int_params['admin_poly']:
            b.plot(r[:,0], r[:,1],'r-')
    
        pp_now = PP.trace()[indices[i]]
    
        for pp_now_now in pp_now:
            lon = pp_now_now[0]
            lat = pp_now_now[1]
            size = pp_now_now[2]
            area = pp_now_now[3]
            drawcircle(lon,lat,size,area)
            
    
        # b.plot(lon,lat,'k.')    
        pl.title('sample %i'%indices[i])
        
        pl.savefig('%s/%s_%i.png'%(path,name,i))
    os.system('convert -delay 15 -loop 0 %s/*.png %s/%s.gif'%(path,path,name))
    os.system('rm %s/*.png'%path)
    matplotlib.interactive(True)
    
M.sample(100000,10000,100)
show(PP,100,'frame','figs')