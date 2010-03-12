import numpy as np
import pymc as pm

__all__ = ['PointProcess', 'PoissonProcess', 'PairwiseInteractionProcess', 'integrate_intensity']
__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'

class PointProcess(pm.Stochastic):
    """
    A general point process: a stochastic variable valued as a tuple of 
    points.
    
    PointProcess' logp attribute returns the density of its current value 
    with respect to a Poisson process with intensity 1.
    
    If that doesn't make sense, see PoissonProcess and 
    PairwiseInteractionProcess.
    
    :Parameters:
      - name : string
      - log_density : function
        Should take a tuple of points and arbitrary keyword arguments.
        Its output should be the log-density of the point tuple with respect
        to the Poisson process with intensity 1. This should be a float.
      - log_subset_density : function
        Should take two tuples of points and arbitrary keyword arguments.
        Its output should be proportional to the log-density of the first tuple
        given the second tuple with respect to the standard Poisson process.
        This should be a float. It will not always be feasible to supply this
        argument.
      - value : tuple
        Each element is a list/array/tuple of coordinates.
      - dims : integer
        The dimension of the space in which the points are scattered.
      - bounding_box : length-2 tuple
        Two corners of the bounding box: the one where each coordinate takes its
        minimal value, and the one where each coordinate takes its maximal value.
        The logp attribute will raise a ZeroProbability if any of the points in 
        the current value are outside this bounding box.
      - doc : string
      - random : function
        If it's possible to generate realizations efficiently, this function can 
        be supplied.
      - trace : boolean
        Whether a trace should be maintained for this variable.
      - rseed : float
        A seet for self's RNG.
      - isdata : boolean
        Whether self's value has been observed
      - cache_depth : integer
        How many previous values and log-probabilities to cache.
      - density_params : optional keyword arguments
        Any parameters needed by log_density. These may be Python objects or PyMC 
        variables.
    
    :Reference:
      Jesper Moller and Rasmus Plenge Waagepetersen. Statistical inference
      and simulation for spatial point processes. CRC Press, 2004.
    
    :SeeAlso:
      MarkovPointProcess, PairwiseInteractionProcess, PoissonProcess
    """
    def __init__(self, 
                name,     
                log_density,
                log_subset_density=None,
                value=[],   
                dims=None,  
                bounding_box=None,             
                doc='A point process',                 
                random=None,
                trace=True, 
                rseed=False, 
                isdata=False, 
                cache_depth=2,
                verbose=0,
                **density_params):
                
        self.density_params = pm.DictContainer(density_params)
        self.log_density_fun = log_density
        self.log_subset_density_fun = log_subset_density

        self.bounding_box = bounding_box
        if bounding_box is not None:
            if dims is None:
                dims = len(bounding_box[0])
            else:
                if not dims == len(bounding_box[0]):
                    raise ValueError, 'Bounding box has wrong number of spatial dimensions'
            self.bounding_box = (np.array(self.bounding_box[0]), np.array(self.bounding_box[1]))
            if np.any(self.bounding_box[1] <= self.bounding_box[0]):
                raise ValueError, 'All elements of bounding_box[0] must be less than corresponding elements of bounding_box[1]'
        
        def logp_fun(value, log_density, density_params):
            return log_density(value, **density_params)
            
        parents = {'density_params': density_params, 'log_density': log_density}
        value = tuple(value)
                
        pm.Stochastic.__init__(self, 
                            logp=logp_fun,
                            doc=doc,
                            name=name,
                            parents=parents,
                            random=random,
                            trace=trace,
                            value=value,
                            dtype=object,
                            rseed=rseed,
                            isdata=isdata,
                            cache_depth=cache_depth,
                            plot=False,
                            verbose=verbose)
                            
        if dims is not None:
            self.dims = dims
        else:
            if len(value) > 0:
                self.dims = len(np.atleast_1d(value[0]))
            else:
                raise ValueError, 'Point process %s: argument dims not provided & cannot guess because initial value is None or empty.' % name     
                        
    def set_value(self, new_val):
        if self.isdata:
            raise AttributeError, "Stochastic %s's value cannot be updated if isdata flag is set" % self.__name__
        self.last_value = self._value
        self._value = tuple(new_val)
        
    value = property(pm.Stochastic.get_value, set_value)
                            
    
                                


class PoissonProcess(PointProcess):
    """
    A Poisson point process: a stochastic variable valued as a tuple of
    points.
    
    PoissonProcess' logp attribute returns the sum of the log of the 
    intensity function evaluated at all points in the current value.
    
    :Parameters:
      - name : string
      - log_intensity : function
        Should take a tuple of points and arbitrary keyword arguments.
        Its output should be the sum of the log of the intensity function
        evaluated at all points in the current value. This should be a 
        float.
      - value : tuple
        Each element is a list/array/tuple of coordinates.
      - dims : integer
        The dimension of the space in which the points are
        scattered.
      - bounding_box : length-2 tuple
        Two corners of the bounding box: the one where each coordinate takes its
        minimal value, and the one where each coordinate takes its maximal value.
        The logp attribute will raise a ZeroProbability if any of the points in 
        the current value are outside this bounding box.        
      - doc : string
      - random : function
        If it's possible to generate realizations efficiently, this function can be 
        supplied.
      - trace : boolean
        Whether a trace should be maintained for this variable.
      - rseed : float
        A seet for self's RNG.
      - isdata : boolean
        Whether self's value has been observed
      - cache_depth : integer
        How many previous values and log-probabilities to cache.
      - density_params : optional keyword arguments
        Any parameters needed by log_density. These may be Python objects or PyMC 
        variables.
    
    :Reference:
      Jesper Moller and Rasmus Plenge Waagepetersen. Statistical inference
      and simulation for spatial point processes. CRC Press, 2004.
    
    :SeeAlso:
      PointProcess, PairwiseInteractionProcess, MarkovPointProcess,
      integrate_intensity
    """
    def __init__(self, name, log_intensity, **kwargs):
        
        def log_subset_density(x, y, **density_params):
            return log_intensity(x, **density_params)
        
        PointProcess.__init__(self, name, log_intensity, log_subset_density, **kwargs)
        
        self.log_intensity_fun = log_intensity
        
    

class PairwiseInteractionProcess(PointProcess):
    """
    A pairwise-interaction point process: a stochastic variable valued as 
    a tuple of points.
    
    PairwiseInteractionProcess' logp attribute returns the product of a 
    function evaluated at all the points and another function evaluated at 
    all the pairs of points.
    
    :Parameters:
      - name : string
      - log_intensity : function
        Should take a tuple of points and arbitrary keyword arguments.
        Its output should be the sum of the log of the intensity function
        evaluated at all points in the current value. This should be a 
        float.
      - log_interaction : function
        Should take two tuples of points and arbitrary keyword arguments.
        Its output should be the sum of the log of the interaction function
        evaluated at all pairs in the Cartesian product of the two tuples. 
        This should be a float.
      - value : tuple
        Each element is a list/array/tuple of coordinates.
      - dims : integer
        The dimension of the space in which the points are
        scattered.
      - bounding_box : length-2 tuple
        Two corners of the bounding box: the one where each coordinate takes its
        minimal value, and the one where each coordinate takes its maximal value.
        The logp attribute will raise a ZeroProbability if any of the points in 
        the current value are outside this bounding box.                
      - doc : string
      - random : function
        If it's possible to generate realizations efficiently, this function can be 
        supplied.
      - trace : boolean
        Whether a trace should be maintained for this variable.
      - rseed : float
        A seet for self's RNG.
      - isdata : boolean
        Whether self's value has been observed
      - cache_depth : integer
        How many previous values and log-probabilities to cache.
      - density_params : optional keyword arguments
        Any parameters needed by log_density. These may be Python objects or PyMC 
        variables.
    
    :Reference:
      Jesper Moller and Rasmus Plenge Waagepetersen. Statistical inference
      and simulation for spatial point processes. CRC Press, 2004.
    
    :SeeAlso:
      PointProcess, MarkovPointProcess, PoissonProcess
    """
    def __init__(self, name, log_intensity, log_interaction, **kwargs):
        
        self.log_intensity_fun = log_intensity
        self.log_interaction_fun = log_interaction
        
        def log_density_fun(value, density_params):
            return log_intensity_fun(value, **density_params) \
                + log_interaction_fun(value, **density_params)
        
        def log_subset_density_fun(this, other, density_params):
            return log_intensity_fun(this, **density_params) \
                + log_interaction_fun(this, this, **density_params) \
                + log_interaction_fun(this, other, **density_params)
                
        PointProcess.__init__(self, name, log_density_fun, log_subset_density_fun, **kwargs)
        

def integrate_intensity(PP,N):
    """
    Integrates the intensity function of a Poisson process using standard
    Monte Carlo integration.
    
    N uniformly-distributed points are generated over the Poisson process's
    bounding box, and the average of the intensity function evaluated at these
    points is used to approximate the integral.
    
    Returns the estimate of the integrated intensity, and also a vector giving 
    the log-intensity evaluated at all the randomly-generated points.
    
    :Parameters:
      - PP : Poisson process
        The Poisson process whose intensity is to be integrated.
      - N : Integer
        The number of Monte Carlo samples to be used to estimate the integral.
        Higher value gives better estimate, takes longer.
        
    :SeeAlso: PoissonProcess
    """
    
    if not hasattr(PP,"bounding_box"):
        raise ValueError, 'Poisson process %s lacks bounding box' %pp.__name__
    lif = PP.log_intensity_fun
    dp = PP.density_params.value
    nd = len(PP.bounding_box[0])
    bb = PP.bounding_box
    
    coords = np.random.random(size=(N,nd))
    ints = np.empty(N)
    
    for i in xrange(nd):
        coords[:,i]*=bb[1][i]-bb[0][i]
        coords[:,i]+=bb[0][i]
    
    for i in xrange(N):
        ints[i]=lif((coords[i,:],), **dp)
        # print coords[i,:], ints[i]        
        
    return np.mean(np.exp(ints)) * np.prod(bb[1]-bb[0]), ints
    

# TODO: Aggregate(P, region) function that integrates intensity over region,
# producing a Poisson RV automatically linked against density_params.
# Need to accomodate marks somehow.

# TOOD: Restrict(P, region) function that restricts point process to region.
# Produces analogous point process object automatically linked against
# density_params. Need to accomodate marks here too.

# TODO: When merging this into trunk, make a 'spatial' subpackage and move the
# distances out of GP.
    
if __name__ == '__main__':
    def la(x, mu, tau,lam):
        return pm.mv_normal_like(np.array(x), mu, tau)+np.log(lam)
        
    PP = PoissonProcess('PP', la, value=[np.array([1,2,3]),np.array([4,5,6])], bounding_box = [(-10.,)*3,(10.,)*3], mu=np.ones(3), tau = np.eye(3),lam=100)