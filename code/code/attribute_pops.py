from __future__ import division
import shapelib, dbflib
import tables as tb
import numpy as np
import ptinpoly as inp
import pylab as pl
import matplotlib
from matplotlib.nxutils import points_inside_poly
from mpl_toolkits import basemap

__all__ = ['inregion', 'shapefile_to_hdf5', 'lon_lat', 'raster_blaster', 'shaolin_admin', 'display_raster']

def inregion(x,y,r):
    """
    ins = inregion(x,y,r)

    Returns an array of booleans indicating whether the x,y pairs are
    in region r. If region r contains multiple polygons, the ones inside 
    the biggest one are assumed to be holes; the ones inside holes
    are assumed to be islands; and so on.
    
    :Parameters:
      x : array
        x coordinates of test points
      y : array
        y coordinates of test points
      r : ShapeObject
        The region.
    """
    xy = np.vstack((x,y)).T

    # Record whether each point is inside each polygon.
    ins = []
    for v in r:
        ins.append(points_inside_poly(xy,v))
    ins = np.array(ins)
    
    # Return an array of booleans. An element is True if
    # the corresponding point is inside an odd number of polygons.
    return np.sum(ins, axis=0) % 2 == 1

def shapefile_to_hdf5(shp_name, to_name=None):
    """
    shapefile_to_hdf5(shp_name)
    
    Converts the shapefile and dbffile with name 'shp_name' to the
    hdf5 archive 'shp_name.hdf5'.
    
    Returns the main table of the archive, wrapped to provide natural
    naming access to columns.
    """

    if to_name==None:
        to_name = shp_name

    hf = tb.openFile(to_name + '.hdf5', 'w')
    
    # Read in shapefile and dbf file.
    s = shapelib.ShapeFile(shp_name)
    d = dbflib.DBFFile(shp_name)
    N_objects = s.info()[0]
    examp_dict = d.read_record(0)
    
    # Make column descriptor from stuff in dbf file and extents in shapefile.
    descrip_dict = {'extents' : tb.Col.from_atom(tb.FloatAtom(shape=(2,4)))}
    for name in examp_dict.iterkeys():
        try:
            float(examp_dict[name])
            descrip_dict[name] = tb.Col.from_atom(tb.FloatAtom())
        except:
            descrip_dict[name] = tb.Col.from_atom(tb.StringAtom(120))
    hf.createTable('/','info',descrip_dict,expectedrows=N_objects)
    
    # Vertex sets will be pickled and stored in a VLArray. Not the most efficient
    # but otherwise you end up with lots of complexity.
    hf.createVLArray('/','vertices',tb.ObjectAtom(),expectedsizeinMB=10.0)
    vertices = hf.root.vertices
    
    t = hf.root.info
    r = t.row
    
    for i in xrange(N_objects):
    
        # Write in extents and db info for object i
        p = s.read_object(i)        
        r['extents'] = np.array(p.extents())
        q = d.read_record(i)
        for name in examp_dict.iterkeys():
            r[name] = q[name]        
        r.append()
        
        # Write in vertices for object i
        v = p.vertices()
        hf.root.vertices.append([np.array(vi)*np.pi/180 for vi in v])
            
    t.flush()
    hf.close()
    
def lon_lat(res):
    """
    Figures out a global mesh over the surface of the earth
    at resolution 'res' radians.
    """
    N_lon = int(2*np.pi/res)
    N_lat = int(np.pi/res)
    lon = np.linspace(-np.pi,np.pi,N_lon)
    lat = np.linspace(-np.pi/2,np.pi/2,N_lat)
    
    dlon = lon[1]-lon[0]
    dlat = lat[1]-lat[0]
    
    return N_lon, N_lat, lon, lat, dlon, dlat
    
def make_pixels(shp_name, res, cache=True):
    """
    Adds joint pixel draws to the hdf5 archive with name shp_name.hdf5.
    
    Will add the following data structure to the hdf5 root:
    pixels: group
      obj0: table
         lat lon samps  :  one row per pixel.
         
    Note: one radian on the earth's surface is 2030.21229 km
    so a 5km mesh is roughly .0025 radians.
    """
    
    # Read in information
    hf = tb.openFile(shp_name+'.hdf5','a')
    info = hf.root.info
    cinfo = info.cols
    v = hf.root.vertices
    N = len(info)
    
    N_lon, N_lat, lon, lat, dlon, dlat = lon_lat(res)
    
    # If desired, if information at this resolution already exists,
    # destroy it.
    if cache:
        if hasattr(hf.root, 'admin_units_%i_%i'%(N_lon,N_lat)):
            return
    else:
        try:
            hf.removeNode('/','admin_units_%i_%i'%(N_lon,N_lat))
        except tb.NodeError:
            pass
        try:
            hf.removeNode('/','owned_indices_%i_%i'%(N_lon,N_lat))
        except tb.NodeError:
            pass
        try:
            hf.removeNode('/','lon_%i'%(N_lon))
        except tb.NodeError:
            pass
        try:
            hf.removeNode('/','lat_%i'%(N_lat))
        except tb.NodeError:
            pass
    
    # Initialize information at this resolution.

    # admin_units_<N_lon>_<N_lat> is an array of integers over the global mesh.
    # Each pixel carries the index of the admin unit to which it belongs.
    admin_units = hf.createCArray('/','admin_units_%i_%i'%(N_lon,N_lat), tb.IntAtom(), (2*np.pi/res, np.pi/res))
    admin_units[:,:] = -1

    # owned_indices_<N_lon>_<N_lat> is a variable-length array. Each row
    # corresponds to an admin unit and contains the index pairs in the global
    # mesh that are owned by that unit.
    owned_indices = hf.createVLArray('/','owned_indices_%i_%i'%(N_lon,N_lat), tb.IntAtom(shape=(2,)))
    
    hf.createArray('/','lon_%i'%(N_lon),lon)
    hf.createArray('/','lat_%i'%(N_lat),lat)
    
    # owned_indices = getattr(hf.root, 'owned_indices_%i_%i'%(N_lon,N_lat))
    # admin_units = getattr(hf.root, 'admin_unis_%i_%i'%(N_lon,N_lat))
    
    for i in xrange(N):
        
        if i%1000 == 0:
            print i
        
        # Figure out which locations are candidates for being in admin unit i
        # based on the bounding box.
        this_extent = cinfo.extents[i] * np.pi / 180.
        lon_min = this_extent[0,0]
        lat_min = this_extent[0,1]
        lon_max = this_extent[1,0]
        lat_max = this_extent[1,1]
        
        i_min = int((lon_min + np.pi) / dlon)
        i_max = int((lon_max + np.pi) / dlon + 1)
        j_min = int((lat_min + np.pi/2) / dlat)
        j_max = int((lat_max + np.pi/2) / dlat + 1)        
        
        unclipped_ind = np.meshgrid(np.arange(i_min, i_max+1), np.arange(j_min, j_max+1))
        unclipped_ind = np.asmatrix(np.vstack((unclipped_ind[0].ravel(), unclipped_ind[1].ravel())))
        
        unclipped_loc = np.meshgrid(lon[i_min: i_max+1], lat[j_min: j_max+1])
        unclipped_lon = unclipped_loc[0].ravel() 
        unclipped_lat = unclipped_loc[1].ravel()
        
        # Clip the mesh to the polygon boundary.
        # print i,len(v[i])
        ins = inregion(unclipped_lon, unclipped_lat, v[i])
        ins = np.where(ins)[0]
        N_in = len(ins)
        
        if N_in > 0:
            clipped_ind = unclipped_ind[:,ins]
            owned_indices.append(clipped_ind.T)
            # print '\t',N_in, unclipped_ind.shape, clipped_ind.shape
            for j in xrange(clipped_ind.shape[1]):
                admin_units[clipped_ind[0,j],clipped_ind[1,j]] = i 
        else:
            owned_indices.append(np.empty((0,2)))
        
    hf.flush()
    hf.close()

def binomial_process(N_pixels, log_pop_mu, log_pop_V, N):
    """
    Assumes each person has equal chance of finding self
    anywhere.
    """
    pop = np.exp(np.random.normal(np.sqrt(log_pop_V), size=N) + log_pop_mu)
    p = np.empty(N_pixels)
    p.fill(1/N_pixels)

    out = np.empty((N,N_pixels))
    for i in xrange(N):
        out[i,:] = np.random.multinomial(pop[i], p)
    return out

def eq_dirichlet_process(N_pixels, log_pop_mu, log_pop_V, N):
    """
    Assumes any pixel configuration is equally likely.
    """
    pop = np.exp(np.random.normal(np.sqrt(log_pop_V)) + log_pop_mu)
    alpha = np.ones(N_pixels)
    return np.random.dirichlet(alpha, size=N) * pop

def closedform_binomial_sd(N_pix, log_pop_mu, log_pop_V):
    pop = np.exp(np.random.normal(np.sqrt(log_pop_V), size=100) + log_pop_mu)
    V = np.var(pop/N_pix) + np.mean(pop/N_pix*(1-1/N_pix))
    return np.sqrt(V)
    
def closedform_binomial_sd_to_mean(N_pix, log_pop_mu, log_pop_V):
    pop = np.exp(np.random.normal(np.sqrt(log_pop_V), size=100) + log_pop_mu)
    V = np.var(pop/N_pix) + np.mean(pop/N_pix*(1-1/N_pix))
    sd = np.sqrt(V)
    m = np.mean(pop/N_pix)
    return sd / m
        

def closedform_raster(shp_name, pix_name, res, pix_func):
    # Parse up datafile
    hf = tb.openFile(shp_name+'.hdf5','a')
    info = hf.root.info
    cinfo = info.cols
    v = hf.root.vertices
    N_units = len(info)
    
    # Get global mesh
    N_lon, N_lat, lon, lat, dlon, dlat = lon_lat(res)
    
    # Allocate space for samples. Compress the shit out of them.
    if hasattr(hf.root,pix_name):
        hf.removeNode('/',pix_name,recursive=True)    
    pix = hf.createCArray('/',pix_name,tb.FloatAtom(),(N_lon, N_lat),filters=tb.Filters(complevel=9))
    
    pix_indices = getattr(hf.root,'owned_indices_%i_%i'%(N_lon,N_lat))
    for i in xrange(N_units):
        if i%100==0:
            print i
        
        # Extract information for this unit.
        try:
            pop = float(cinfo.GPW2008[i])
        except:
            pop = 0.
        mu_pop = min(np.log(pop),0.)
        V_pop = 1.
        these_pix_indices = pix_indices[i]
        N_pix = these_pix_indices.shape[0]

        # Draw samples and write in.
        if N_pix > 0:
            these_pix = pix_func(N_pix, mu_pop, V_pop)
            for j in xrange(N_pix):
                this_pix_index = these_pix_indices[j]
                pix[this_pix_index[0],this_pix_index[1]] = these_pix
                
    hf.flush()
    hf.close()



def draw_samples(shp_name, pix_name, N, res, pix_func):
    """
    Adds attribute 'pix_name' to the hdf5 archive.
    pix_name will be an N_lon X N_lat X N array.
    The last dimension iterates over sample number.
    
    Function pix_func is used to draw samples within
    each administrative unit.
    """
    
    # Parse up datafile
    hf = tb.openFile(shp_name+'.hdf5','a')
    info = hf.root.info
    cinfo = info.cols
    v = hf.root.vertices
    N_units = len(info)
    
    # Get global mesh
    N_lon, N_lat, lon, lat, dlon, dlat = lon_lat(res)
    
    # Allocate space for samples. Compress the shit out of them.
    if hasattr(hf.root,pix_name):
        hf.removeNode('/',pix_name,recursive=True)    
    pix = hf.createCArray('/',pix_name,tb.FloatAtom(),(N_lon, N_lat, N),filters=tb.Filters(complevel=9))
    
    pix_indices = getattr(hf.root,'owned_indices_%i_%i'%(N_lon,N_lat))
    for i in xrange(N_units):
        print i
        
        # Extract information for this unit.
        try:
            pop = float(cinfo.GPW2008[i])
        except:
            pop = 0.
        mu_pop = min(np.log(pop),0.)
        V_pop = .1
        these_pix_indices = pix_indices[i]
        N_pix = these_pix_indices.shape[0]

        # Draw samples and write in.
        if N_pix > 0:
            try:
                these_samps = pix_func(N_pix, mu_pop, V_pop, N)
                for j in xrange(N_pix):
                    this_pix_index = these_pix_indices[j]
                    pix[this_pix_index[0],this_pix_index[1],:] = these_samps[:,j]
            except:
                for j in xrange(N_pix):
                    pix[this_pix_index[0],this_pix_index[1],:] = -1                    
                
    hf.flush()
    hf.close()

def raster_blaster(shp_name, pix_name, raster_name, redux_fun):
    """
    Reduces pixel samples stored in pix_name to make the raster surface
    raster_name using redux_fun. Hdf5 archive is shp_name.
    """
    
    # Read info, allocate output.
    hf = tb.openFile(shp_name+'.hdf5','a')
    
    samps = getattr(hf.root, pix_name)
    if hasattr(hf.root, raster_name):
        hf.removeNode('/',raster_name,recursive=True)
    ras = hf.createCArray('/',raster_name,tb.FloatAtom(),samps.shape[:2])
    N_lon = ras.shape[0]
    N_lat = ras.shape[1]
    
    # Reduce the samples.
    for i in xrange(N_lon):
        for j in xrange(N_lat):
            ras[i,j] = redux_fun(samps[i,j,:])
    
    hf.flush()
    hf.close()
    
def burden(samps, owned_indices, info, lon, lat, klass):
    """
    From samples of population raster, indices belonging
    in this admin unit, info for this admin unit, longitude
    and latitude, computes expected number of people living
    in a particular endemicity class.
    """
    pass
    
def shaolin_admin(shp_name, pix_name, summary_name, redux_fun, **args):
    # Read info, allocate output.
    hf = tb.openFile(shp_name+'.hdf5','a')
    
    samps = getattr(hf.root, pix_name)
    N_lon = samps.shape[0]
    N_lat = samps.shape[1]

    owned_indices = getattr(hf.root,'owned_indices_%i_%i'%(N_lon,N_lat))
    N_admin = len(owned_indices)
    
    if hasattr(hf.root, summary_name):
        hf.removeNode('/',summary_name,recursive=True)
    summary = hf.createCArray('/',summary_name,tb.FloatAtom(),samps.shape[:2])
    
    for i in xrange(N_admin):
        summary[i] = redux_fun(samps, owned_indices[i], hf.info[i], hf.lon, hf.lat,**args)
    
def std_to_mean_ratio(x):
    m = np.mean(x)
    if m>0:
        return np.std(x) / m
    else:
        return 0
        
def display_raster(shp_name, ras_name, postprocess=lambda x:x):
    b = basemap.Basemap()
    hf = tb.openFile(shp_name + '.hdf5')
    ras = getattr(hf.root, ras_name)
    ras_to_show = np.asarray(ras[:,:].T).copy()
    rr = ras_to_show.ravel()
    print rr.max(),rr.min(),rr[np.where(rr<60)].max()
    for i in xrange(len(rr)):
        if rr[i]>70 or np.isnan(rr[i]): 
            # print i,rr[i],'Greater than 60, setting to 0'
            rr[i]=0
        # else:
        #     print i,rr[i],'Leaving alone'
    print rr.max(),rr.min(),rr[np.where(rr<60)].max()        
    ras_to_show = rr.reshape(ras_to_show.shape)
    print rr.max(), rr.min()
    # ras_to_show = postprocess(ras_to_show)
    b.imshow(ras_to_show)
    pl.colorbar()
    b.drawcoastlines(color='r')
    b.drawcountries(color='r')
    pl.title(ras_name)
    hf.close
    f=pl.figure(1)
    f.set_alpha(0)
    
if __name__ == '__main__':
    pass
    # ==============
    # = Test bit 0 =
    # ==============
    
    # make_pixels('merge7', .0025, cache=True)
    # draw_samples('merge7','binomial',1000,.0025,binomial_process)
    # closedform_raster('merge7', 'binomial_std', .0025, closedform_binomial_sd)
    # closedform_raster('merge7', 'binomial_std_to_mean', .0025, closedform_binomial_sd_to_mean)
    
    
    # ==============
    # = Test bit 1 =
    # ==============
    
    # hf=shapefile_to_hdf5('merge7').root
    # q = hf.info
    # v = hf.vertices

    # ==============
    # = Test bit 2 =
    # ==============

    # s = shapelib.ShapeFile('merge7')
    # d = dbflib.DBFFile('merge7')
    # N = s.info()[0]
    # 
    # i = 94318
    # N = 100
    # 
    # p = s.read_object(i)
    # x_test = np.random.uniform(p.extents()[0][0], p.extents()[1][0], size=N)
    # y_test = np.random.uniform(p.extents()[0][1], p.extents()[1][1], size=N)    
    # ins = inregion(x_test, y_test, p)
    # 
    # pl.clf()
    # v = [np.array(p_el) for p_el in p.vertices()]    
    # for item in v:
    #     pl.fill(item[:,0],item[:,1],facecolor=(.8,.8,.8), alpha=.4)
    #     # raw_input()
    # pl.plot(x_test,y_test,'r.',markersize=4)
    # 
    # [pl.text(x_test[i],y_test[i],str(ins[i])[0], fontsize='8') for i in range(len(x_test))]
    # 
    # 
