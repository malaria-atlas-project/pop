# This is the top-level file! The other useful ones are attribute_pops_no_MCMC and country_curve_inference.

from __future__ import division
import numpy as np
import country_curve_inference
from mpl_toolkits import basemap
import matplotlib.pyplot as pl
import tables as tb
import matplotlib
import pymc as pm
import sys
from attribute_pops_no_MCMC import inregion, settlement_size_samples
from shapely import geometry, iterops, wkb, wkt
import shapely.geometry as geom
import xlrd as excel
from csv import reader
import psycopg2

# an-na-bo:
# ncols         2122
# nrows         2951
xllcorner=     11.679219245911
yllcorner =    -28.971429824829
cellsize   =   0.0083333333333

tb_filters=None

def get_settlement_pops(country):
    pops = []
    r = reader(file('settlement_sizes.csv'))
    for i in xrange(2):
        r.next()
    countries = np.array(r.next())
    country_index = np.where(countries==country)[0][0]
    for i in xrange(2):
        r.next()
    total_pop = np.float(r.next()[country_index])
    r.next()
    for line in r:
        if len(line[country_index])==0:
            break
        this_pop = np.int(line[country_index])
        if this_pop>0:
            pops.append(this_pop)
        else:
            break
    return np.array(pops), total_pop
    

def polygon_area(v):
    """
    polygon_area(v)
    Returns area of polygon
    """
    v_first = v[:-1][:,[1,0]]
    v_second = v[1:]
    return np.diff(v_first*v_second).sum()/2.0

def unit_to_grid(unit, lon_min, lat_min, cellsize):
    """
    unit_to_grid(unit, lon_min, lat_min, cellsize)
    
    unit : a shapely polygon or multipolygon
    lon_min : The llc longitude of the master raster
    lat_min : The llc latitude of the master raster
    cellsize: of the master raster
    
    Retruns two arrays, lon and lat, of points on the raster in the unit.
    """
    llc = unit.bounds[:2]
    urc = unit.bounds[2:]
    
    lon_min_in = int((llc[0]-lon_min)/cellsize)
    lon_max_in = int((urc[0]-lon_min)/cellsize)+1
    lat_min_in = int((llc[1]-lat_min)/cellsize)
    lat_max_in = int((urc[1]-lat_min)/cellsize)+1  
    
    x_extent = np.arange(lon_min_in, lon_max_in+1)*cellsize + lon_min
    y_extent = np.arange(lat_min_in, lat_max_in+1)*cellsize + lat_min
    
    nx = len(x_extent)
    ny = len(y_extent)
    
    xm,ym = np.meshgrid(x_extent, y_extent)
    x=xm.ravel()
    y=ym.ravel()
    
    p=[geom.Point([x[i],y[i]]) for i in xrange(len(x))]
    
    return iterops.contains(unit, p, True)
    
def exclude_ues(xy, unit, ue_shapefile):
    """
    exclude_ues(xy,unit,ue_shapefile)
    
    xy : sequence of points.
    unit : The shapely polygon or multipolygon containing xy.
    ue_shapefile : A NonSuckyShapeFile object.
    
    Returns x and y filtered to be outside the polygons in the shapefile.
    """
    if not unit.is_valid:
        raise ValueError, 'invalid unit'
    intersect_fracs = []
    for ue in ue_shapefile:
        if unit.intersects(ue):
            xy = iterops.contains(unit, xy, True)
            intersect_fracs.append(unit.intersection(ue).area / ue.area)
        else:
            intersect_fracs.append(0)
    return xy, intersect_fracs

def plot_unit(b, unit, *args, **kwargs):
    """
    plot_unit(b, unit, *args, **kwargs)
    
    b : a Basemap.
    unit : a ShapeObject.
    args, kwargs : Passed to plot.
    """

    if isinstance(unit, geom.Polygon):
        v = np.array(unit.exterior)
        b.plot(v[:,0],v[:,1],*args, **kwargs)
    else:
        for subunit in unit.geoms:
            v = np.array(subunit.exterior)
            b.plot(v[:,0],v[:,1],*args, **kwargs)
    
        
def obj_to_poly(shape_obj, reverse=False):
    """
    Converts ShapeObject to shapely polygon or multipolygon
    """
    v = shape_obj.vertices()
    if len(v)==1:
        if reverse:
            return geom.Polygon(v[0][::-1])
        else:
            return geom.Polygon(v[0])
    else:
        if reverse:
            holes = []
            for g in v[1:]:
                holes.append(g[::-1])
            return geom.MultiPolygon([(v[0][::-1], holes)])
        return geom.MultiPolygon([(v[0], v[1:])])
    

class NonSuckyShapefile(object):
    """
    S = NonSuckyShapefile(fname)
    
    Holds some information about fname, and supports iteration and getiteming.
    Also has method plotall(b, *args, **kwargs).
    """
    def __init__(self, fname):
        
        self.fname = fname
        self.sf = basemap.ShapeFile(fname)
        self.llc = self.sf.info()[2]
        self.urc = self.sf.info()[3]
        self.n = self.sf.info()[0]
        
        self.polygons = []
        for i in xrange(self.n):
            self.polygons.append(obj_to_poly(self.sf.read_object(i)))
            if not self.polygons[-1].is_valid:
                self.polygons[-1] = obj_to_poly(self.sf.read_object(i), reverse=True)
            if not self.polygons[-1].is_valid:
                raise ValueError, 'Invalid polygon %i. What the hell are you trying to pull?'%i
        
        self.index = 0
        
    def __iter__(self):
        return self.polygons.__iter__()
    
    def __len__(self):
        return self.n
                
    def __getitem__(self, i):
        return self.polygons[i]
    
    def __array__(self):
        return np.array([el for el in self])
    
    def __getslice__(self, sl):
        return [self[i] for i in xrange(sl.start, sl.stop, sl.step)]
        
    def plotall(self, b, *args, **kwargs):
        matplotlib.interactive(False)
        for obj in self:
            plot_unit(b, obj, *args, **kwargs)            
        matplotlib.interactive(True)


def query2rec(fields, table):
    conn = psycopg2.connect("dbname='MAP1' user='anand' port=2345 host='localhost' password='app_2009'")
    cur = conn.cursor()
    query_str = "SELECT "+', '.join(fields)+", st_astext(st_buffer(the_geom,0)) from " + table
    cur.execute(query_str)
    results = cur.fetchall()
    good_res = []
    for i in xrange(len(results)):
        this_res = results[i]
        # st_multi(st_buffer(the_geom, 0))
        mod_res = this_res[:-1] + (wkt.loads(str(this_res[-1])),)
        if not mod_res[-1].is_valid:
            err_str = ('Table %s: Multipolygon %i is invalid. \nMetadata:\n\t'%(table,i))+'\n\t'.join([fields[i]+': '+str(this_res[i]) for i in xrange(len(fields))])
            print err_str
        good_res.append(mod_res)
    return np.rec.fromrecords(good_res, names=fields+['geom'])

        

        
# ==========================
# = Initialize output file =
# ==========================
outfile = tb.openFile('pop_samples_out.hdf5','w')
countries = outfile.createGroup('/','countries')
errs = []

country_names = {'an_na_bo': ['Angola', 'Namibia', 'Botswana'],
                'sl_it_co': ['Slovenia', 'Italy', 'Croatia']}
N_settlement_samps = 1000
N_nonzero_pixel_samps = 1000

# Run ssh -L 2345:localhost:5432 -fN anand@map1.zoo.ox.ac.uk first
ue_rec = query2rec(['gr_00','country'],'population_urban_extents') 
unit_rec = query2rec(['gr_00','country'], 'population_admin_units')
for ra in [ue_rec, unit_rec]:
    for name in ra.dtype.names:
        if np.any([item is None for item in ra[name]]):
            raise ValueError, 'One of the %s column is None.'%name

# =======================
# = Loop over countries =
# =======================
# for country_collection in ['sl_it_co', 'an_na_bo']:
for country_collection in ['an_na_bo']:
    # Read in population information from Excel files.

    col_unit_pops = unit_rec.gr_00
    col_ue_pops = ue_rec.gr_00
    col_ue_country_names = ue_rec.country
    col_country_names = unit_rec.country.replace('SVN','SLV')
    country_names = set(col_country_names)
    col_units = unit_rec.geom
    col_ue = ue_rec.geom

    
    # for country in country_names:
    for country in ['ITA']:
        
        country_indices = np.where(col_country_names==country)
        unit_pops = col_unit_pops[country_indices]
        country_units = col_units[country_indices]


        ue_country_indices = np.where(col_ue_country_names==country)
        country_ue = col_ue[ue_country_indices]
        ue_pops = col_ue_pops[ue_country_indices]
        
        
        cutoff = ue_pops.min()
            
        # Read in shapefiles
        this_ctry = outfile.createGroup(countries,country)
        # Fit the country data.
        data, tot_pop = get_settlement_pops(country)
        M = pm.MAP(country_curve_inference.make_model(data, tot_pop))
        M.fit()
        mu = M.moo.value
        tau = M.tau.value
    
        outfile.createArray(this_ctry,'mu',[mu])
        outfile.createArray(this_ctry,'tau',[tau])
        outfile.createArray(this_ctry,'known_settlements',data)
        outfile.createArray(this_ctry,'total_population',[tot_pop])    
        outfile.createGroup(this_ctry,'admin_units')

        # ==========================
        # = Loop over admin units. =
        # ==========================
        for adm_index in xrange(len(country_units)):
            print 'Country %s unit %i of %i'%(country, adm_index, len(country_units))

            try:        
                # Find pixels inside admin unit, but outside urban extents.
                this_adm = outfile.createGroup(this_ctry.admin_units, 'unit_'+str(adm_index))
                print '\tCreating grid...'
                xy = unit_to_grid(country_units[adm_index], xllcorner, yllcorner, cellsize)
                print '\tComputing areas of urban extents in grid and population accounted for...'
                xy,intersect_fracs = exclude_ues(xy, country_units[adm_index], country_ue)           
                print '\tConverting points in grid to array...'
                xy = np.array([np.array(pt) for pt in xy])
                x = xy[:,0]
                y = xy[:,1]
                n_pixels = len(x)
            
                # TODO: pop_growth_factor will come from the temporal part.
                pop_growth_factor = 1.2
                census_size = unit_pops[adm_index]
                census_urban_size = np.sum(intersect_fracs * ue_pops)

                outfile.createArray(this_adm, 'pg_mu', [pop_growth_factor])
                outfile.createArray(this_adm, 'pg_tau', [1e6])
                outfile.createArray(this_adm, 'census_size', [census_size])
                outfile.createArray(this_adm, 'census_urban_size', [census_urban_size])
                hf_lon = outfile.createCArray(this_adm, 'lon', tb.Float64Atom(), (n_pixels,), filters=tb_filters)
                hf_lat = outfile.createCArray(this_adm, 'lat', tb.Float64Atom(), (n_pixels,), filters=tb_filters)
                hf_lon[:] = x
                hf_lat[:] = y
    
                # =================
                # = Draw samples. =
                # =================

                print '\tDrawing settlement size samples...'
                # TODO: Draw pop_growth_factor from its posterior. Use pg_mu and pg_tau.
                these_samps = settlement_size_samples(mu + np.log(pop_growth_factor), 
                                                        tau, 
                                                        cutoff*pop_growth_factor, 
                                                        np.log(census_size*pop_growth_factor), 
                                                        2, 
                                                        census_urban_size*pop_growth_factor, 
                                                        N_settlement_samps)

            except:
                print 'Country %s: Error at %i'%(country,adm_index)
                errs.append((country, adm_index, sys.exc_info()))
                continue
                
            outfile.createVLArray(this_adm, 'settlement_size_samples', tb.ObjectAtom(), filters=tb_filters)
            for samp in these_samps:
                this_adm.settlement_size_samples.append(samp)
        
            # ======================================
            # = Find pixel-wise size distribution. =
            # ======================================
            pix_hist = np.empty(N_nonzero_pixel_samps)

            n_samples_present = np.array([len(samp) for samp in these_samps])
            all_ns = np.sort(list(set(n_samples_present)))
            p_of_n = np.empty(len(all_ns))
            for i in xrange(len(all_ns)):
                p_of_n[i] = np.sum(n_samples_present == all_ns[i])
            lp_of_n = np.log(p_of_n) - np.log(N_settlement_samps)
            
            # Log of probability that the population of the pixel is zero.
            lp_zero_given_n = all_ns*pm.utils.log_difference(0,-np.log(n_pixels))
            lp_nonzero_given_n = np.array([pm.utils.log_difference(0, lpz) for lpz in lp_zero_given_n])
            lp_zero = pm.flib.logsum(lp_zero_given_n + lp_of_n)
            lp_nonzero = pm.utils.log_difference(0,lp_zero)
            
            # Log-'distribution' of n given there is some population in the pixel.
            lp_n_given_nonzero = lp_nonzero_given_n + lp_of_n
            lp_n_given_nonzero -= pm.flib.logsum(lp_nonzero_given_n + lp_of_n)
            
            # Draw population sizes for the pixel given there is some population.
            print '\tDrawing population sizes given some population present...'
            for i in xrange(N_nonzero_pixel_samps):
                # Choose a value for number of settlements
                this_n = all_ns[pm.rcategorical(np.exp(lp_n_given_nonzero))]
                # Choose a sample with the right number of settlements.
                where_good = np.where(n_samples_present == this_n)[0]
                j = where_good[np.random.randint(len(where_good))]
                this_samp = these_samps[j]
                # Pick a settlement at random and assign it to this pixel.
                guaranteed_index = np.random.randint(len(this_samp))
                guaranteed_settlement = this_samp[guaranteed_index]
                # Give the other settlements the option to choose this pixel.
                other_settlements_in = np.random.random(size=this_n)<1./n_pixels
                other_settlements_in[guaranteed_index] = 0
                where_in = np.where(other_settlements_in)
                other_settlements = this_samp[where_in]
                # Add up the total population size in this pixel.
                pix_hist[i] = guaranteed_settlement + np.sum(other_settlements)
                
            outfile.createArray(this_adm, 'lp_zero', [lp_zero])
            outfile.createArray(this_adm, 'lp_nonzero', [lp_nonzero])
            outfile.createArray(this_adm, 'lp_n_given_nonzero', [lp_n_given_nonzero])
            outfile.createCArray(this_adm, 'per_pixel_nonzero_samps', tb.Float64Atom(), (N_nonzero_pixel_samps,), filters=tb_filters)
            this_adm.per_pixel_nonzero_samps[:] = pix_hist
