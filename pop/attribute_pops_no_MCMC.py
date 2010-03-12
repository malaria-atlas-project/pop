# Copyright (C) 2010  Anand Patil
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division
# from attribute_pops import *
import tables as tb
import numpy as np
from numpy import pi, log, where, hstack, exp
import pylab as pl
import matplotlib
import pymc as pm
from mpl_toolkits import basemap
from matplotlib.nxutils import points_inside_poly
import os
import scipy
from scipy import integrate
from pymc.flib import logsum as log_sum
import sys

I = np.complex(0,1)

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

def truncnorm_1(mu, tau, cutoff, size):
    norms = np.random.normal(size=size)/np.sqrt(tau)+mu
    not_OK = np.where(norms>cutoff)
    tries = 0
    while len(not_OK[0])>0:
        N_not_OK = len(not_OK[0])
        # print N_not_OK
        new_norms = np.random.normal(size=N_not_OK)/np.sqrt(tau)+mu
        norms[not_OK] = new_norms
        not_OK = np.where(norms>cutoff)        
        tries += 1
        if tries == 1000:
            raise ValueError, 'Too many damn tries.'
        
    return norms
    
def truncnorm_2(mu, tau, cutoff, size):
    shifted_cutoff = (mu-cutoff) * np.sqrt(tau)
    alpha = (shifted_cutoff + np.sqrt(shifted_cutoff**2 + 4))/2.
    exps = np.random.exponential(1./alpha, size=size) + shifted_cutoff
    not_OK = np.where(np.random.random(size=size) > np.exp(-((alpha-exps)**2)/2.))
    while len(not_OK[0])>0:
        new_exps = np.random.exponential(1./alpha, size=len(not_OK[0])) + shifted_cutoff
        exps[not_OK] = new_exps
        new_not_OK = np.where(np.random.random(size=len(not_OK[0])) > np.exp(-((alpha-new_exps)**2)/2.))
        not_OK = tuple([this_not_OK[new_not_OK] for this_not_OK in not_OK])
    return mu-exps/np.sqrt(tau)

def compare_truncnorms(mu,sigma,cutoff):
    pl.clf()
    pl.subplot(1,2,1)
    try:
        pl.hist(truncnorm_1(mu,sigma**-2,cutoff,1000),50,facecolor='r',normed=True)
        pl.axis('tight')
    except ValueError:
        pass
    pl.subplot(1,2,2)
    pl.hist(truncnorm_2(mu,sigma**-2,cutoff,1000),50,facecolor='b',normed=True)    
    pl.axis('tight')    
    
    
def geto_truncnorm(mu, tau, cutoff, size):
    """
    Returns a sample of normal RV's conditioned to be less
    than a cutoff value.
    
    :Parameters:
      - mu : float
        The mean
      - tau : float
        The precision
      - cutoff : float
        The cutoff value
      - size : integer or container of integers
        Same as every other size argument to an RNG.
    """
    if mu <= cutoff:
        return truncnorm_1(mu, tau, cutoff, size)
    else:
        return truncnorm_2(mu, tau, cutoff, size)

def robust_CF(mu, tau, cutoff, N=10000, start=.00000001,interval=10, N_per_int=1000):
    """
    Computes the discretized characteristic function of the lognormal
    distribution with parameters mu and tau truncated at cutoff.
    
    Computes the CF over the range from a low frequency to a high frequency.
    The low frequency is supplied by the user, and the high frequency is
    found by stepping up on the log scale. The stopping condition is that
    the T-test should not be able to tell that either the real or imaginary
    part of the last step has mean different from 0.
    
    Returns omega, CF
    omega is a vector of frequencies that is evenly-spaced on the log scale,
    CF is the corresponding evaluation of the characteristic function.
    
    :Parameters:
      - mu : float
        Mean parameter of the lognormal distribution.
      - tau : float
        Precision parameter of the lognormal distribution.
      - cutoff : float
        The value at which the lognormal distribution is truncated.
      - N : integer, optional
        This many Monte Carlo iterations will be used to do the integral.
      - start : float, optional
        The lowest frequency to use.
      - interval : float, optional
        The step size on the log scale.
      - N_per_int : float, optional
        The number of frequencies per interval at which the CF will be evaluated.
    """
    #Draw truncated lognormal variates.
    lnorms = np.exp(geto_truncnorm(mu, tau, log(cutoff), N))
    
    # Create omega mesh.
    omin = np.log(start)
    omax = omin + np.log(interval)
    o = np.linspace(omin, omax, N_per_int)
    this_o = o
    
    # Estimate characteristic function by Monte Carlo integration.
    new_FT = np.mean(np.exp(I*np.outer(lnorms, np.exp(this_o))),axis=0)
    FT=new_FT

    # Stopping condition: T-test can't tell that real part or imaginary part is different from 0.
    while not (scipy.stats.ttest_1samp(new_FT.imag, 0)[1] > .05 and scipy.stats.ttest_1samp(new_FT.real, 0)[1] > .05):
        omin = omax
        omax = omin + np.log(interval)
        this_o = np.linspace(omin, omax, N_per_int)
        o = np.hstack((o, this_o))
        new_FT = np.mean(np.exp(I*np.outer(lnorms, np.exp(this_o))),axis=0)
        FT = np.hstack((FT, new_FT))
        
    return o,FT
    
def robust_ICF(o, FT, N, mu, sig, N_per_sd=100):
    """
    Computes the density of the sum of N random variables, each of
    which has a truncated lognormal distribution with mean mu and
    standard deviation sig.
    
    Computes the density over an evenly-spaced interval that goes out
    to the tails (where the density fals to .001 of its value at its
    peak). Initially a mean +/- 1sd envelope is computed, then the
    envelope is expanded one standard deviation at a time until the
    entire distribution is characterized.
    
    Returns mesh, d
    mesh is an evenly-spaced mesh, and d is the density evaluated over
    that mesh.
    
    :Parameters:
      - o : array of floats
        The mesh in frequency space. Assumed positive only.
      - FT : array of floats
        The characteristic function evaluated over o. Its evaluation
        over -o is assumed to be its complex conjugate, and of course
        its evaluation at 0 is assumed to be 1.
      - N : integer
        The number of random variables that should be summed.
      - mu : float
        The mean of each random variable.
      - sig : float
        The standard deviation of each random variable
      - N_per_sd : integer, optional
        The number of mesh points to put in each standard deviation
        of the mesh.
    """

    mu = mu*N
    sig = sig*sqrt(N)

    # After raising CF to a power, you can throw away lots of
    # high frequencies.
    FT = FT**N
    OK = where(abs(FT)>.001)
    FT = FT[OK]
    o = o[OK]

    # Concatenate and reverse CF, because it's only stored on
    # positive frequencies and I'm too stupid to do this better.
    o = hstack((exp(o[::-1]),[0],exp(o)))
    FT = hstack((FT[::-1].conj(),np.complex(1,0), FT))
    
    # Prepare output mesh on [mu-sig, mu+sig]
    mesh_now = np.linspace(mu-sig, mu+sig, 2*N_per_sd)
    dx = mesh_now[1]-mesh_now[0]
    d_now =  scipy.integrate.trapz(y=FT * exp(-I*outer(mesh_now,o)),x=o).real
    mesh = mesh_now
    d = d_now
    
    # Go low until you find low density
    N_lo = 1
    while abs(d[0]/d.max()) >= .01 and mesh[0]>dx:

        N_lo += 1
        if mu - (N_lo)*sig > 0:
            lb = mu-N_lo*sig
            N_here = N_per_sd-1
        else:
            N_here = int((mu - (N_lo-1)*sig)/dx)-1
            lb = mu - (N_lo-1)*sig - (N_here)*dx
            
        mesh_now = np.linspace(lb, mu-(N_lo-1)*sig-dx, N_here)
        d_now = scipy.integrate.trapz(y=FT * exp(-I*outer(mesh_now,o)),x=o).real
        mesh = np.hstack((mesh_now, mesh))
        d = np.hstack((d_now, d))
        
    # Go high until you find low density
    N_hi = 1
    while abs(d[-1]/d.max()) >= .01:
        N_hi += 1
        mesh_now = np.linspace(mu+(N_hi-1)*sig, mu+N_hi*sig-dx, N_per_sd-1)
        d_now = scipy.integrate.trapz(y=FT * exp(-I*outer(mesh_now,o)),x=o).real
        mesh = np.hstack((mesh,mesh_now))
        d = np.hstack((d,d_now))

    # Restrict density to be positive
    for i in xrange(len(d)):
        d[i] = np.max(d[i],0)

    return mesh, d
    
def likelihood_of_sum(mu, tau, cutoff, sum_val, N, oFT=None, sum_moments=None, logl=False):
    """
    Computes the likelihood of a sum of RV's with truncated lognormal 
    distributions given there were arange(N_min, N_max, N_interval) of them. 
    Returns arange(N_min, N_max, N_interval) and the likelihood evaluated 
    at all points.
    
    For small N, a Monte Carlo estimate is used. For intermediate N, the
    characteristic function is raised to a power and then inverse Fourier
    transformed. For large N, the central limit approximation to the sum's
    distribution is used.
    
    :Parameters:
      - mu : float
        The mean of the lognormal distribution.
      - tau : float
        The precision.
      - cutoff : float
        The truncation point
      - sum_val : float
        The value of the sum for which the likelihood will be
        computed.
      - N : container of integers
        The range of N's for which the likelihood should be computed.
      - oFT : tuple, optional
        The omega mesh and discretized characteristic function for the
        distribution of a single RV, computed with robust_CF.
      - sum_moments: tuple, optional
        The mean and variance of the distribution of a single RV.
    """
    sum_range = sum_val*.02    
    
    l = []    
    
    # Prepare stuff for specialized algorithms where necessary.
    # Will use CF for intermediate values.
    if np.any(N<50) and np.any(N>3):
        if oFT is None:
            o,FT = robust_CF(mu, tau, cutoff)
        else:
            o,FT = oFT

    # Will use CLT for large values.
    if np.any(N>=50):
        if sum_moments is None:
            lnorms = np.exp(geto_truncnorm(mu, tau, log(cutoff), 10000))
            sum_mean = np.mean(lnorms)
            sum_var = np.var(lnorms)
        else:
            sum_mean, sum_var = sum_moments
    
    for n in N:
        # There's no way to get positive population from zero settlements.
        if n==0:
            l.append(0)
            if logl:
                l[-1] = -np.inf
        
        # For small n, use straight Monte Carlo
        elif n<=3:
            s = np.sum(np.exp(geto_truncnorm(mu, tau, log(cutoff), (n,10000))),axis=0)
            l.append(np.sum(np.abs(s-sum_val)<=sum_val*.01)/10000)
            if logl:
                l[-1] = np.log(l[-1])
        
        # For intermediate n, use the characteristic function
        elif n<50:
            FT_r = FT**n
            OK = where(abs(FT_r)>.001)
            FT_r = FT_r[OK]
            o_r = o[OK]

            o_r = hstack((exp(o_r[::-1]),[0],exp(o_r)))
            FT_r = hstack((FT_r[::-1].conj(),np.complex(1,0), FT_r))
        
            l.append(max(scipy.integrate.trapz(y=FT_r * exp(-I*o_r*sum_val),x=o_r).real,0))
            if logl:
                l[-1] = np.log(l[-1])

        # For large n, use the central limit theorem
        else:
            l.append(pm.normal_like(sum_val, sum_mean*n, 1/n/sum_var))
            if not logl:
                l[-1] = np.exp(l[-1])

    return np.array(l)

def robust_posterior(mu, tau, cutoff, sum_val, prior_fun=None, sum_moments=None, oFT=None):
    """
    Computes the likelihood of a sum of RV's with truncated lognormal 
    distributions given there were arange(N_min, N_max, N_interval) of them. 
    Returns arange(N_min, N_max, N_interval) and the likelihood evaluated 
    at all points.
    
    For small N, a Monte Carlo estimate is used. For intermediate N, the
    characteristic function is raised to a power and then inverse Fourier
    transformed. For large N, the central limit approximation to the sum's
    distribution is used.
    
    :Parameters:
      - mu : float
        The mean of the lognormal distribution.
      - tau : float
        The precision.
      - cutoff : float
        The truncation point
      - sum_val : float
        The value of the sum for which the likelihood will be
        computed.
      - prior_fun : function, optional
        Returns the prior density of its argument, N.
      - oFT : tuple, optional
        The omega mesh and discretized characteristic function for the
        distribution of a single RV, computed with robust_CF.
      - sum_moments: tuple, optional
        The mean and variance of the distribution of a single RV.      
    """
    # Prepare the mean and variance and the characteristic function for
    # a single draw to save the likelihood function the bother.
    if sum_moments is None:
        lnorms = np.exp(geto_truncnorm(mu, tau, log(cutoff), 10000))        
        sum_moments = np.mean(lnorms), np.var(lnorms)
    if oFT is None:
        oFT = robust_CF(mu, tau, cutoff)
    
    # Estimate the peak of the posterior using the moments of the likelihood.
    sum_mean, sum_var = sum_moments
    Nhat = sum_val / sum_mean
    Vhat = sum_var / sum_mean**2 * Nhat
    sighat = np.sqrt(Vhat)
    
    # Create a dummy prior if none was provided.
    if prior_fun is None:
        def prior_fun(N):
            return 1
    
    # Prepare the chunk vector
    if sighat > 100:
        N_chunk = np.array(np.linspace(0,sighat,100),dtype=int)
    else:
        N_chunk = np.arange(sighat)
    N_offset = (N_chunk.max()+1)
    
    # Initialize mesh and posterior
    # N_now = np.hstack((N_chunk-N_offset, N_chunk)) + int(Nhat)
    N_now = max(int(Nhat),1) + np.hstack((N_chunk, N_chunk + N_offset))    
    p_now = likelihood_of_sum(mu, tau, cutoff, sum_val, N_now, oFT=oFT, sum_moments=sum_moments) * prior_fun(N_now)
    N = N_now
    p = p_now
    if np.any(np.isnan(p_now)):
        raise ValueError, 'p_now contains nan'
    if np.any(N==0):
        raise ValueError, 'N contains zero'
    
    # Go low until you find low probability
    while abs(p[0]/p.max()) >= .01 and N[0]>N_offset:
        N_now = N_chunk + N[0] - N_offset
        p_now = likelihood_of_sum(mu, tau, cutoff, sum_val, N_now, oFT=oFT, sum_moments=sum_moments) * prior_fun(N_now)      
        N = np.hstack((N_now, N))
        p = np.hstack((p_now, p))
        if np.any(np.isnan(p_now)):
            raise ValueError, 'p_now contains nan'
        if np.any(N==0):
            raise ValueError, 'N contains zero'            
        
    # Go high until you find low probability
    N_hi = 1
    while abs(p[-1]/p.max()) >= .01:
        N_now = N_chunk + N[-1] + 1
        p_now = likelihood_of_sum(mu, tau, cutoff, sum_val, N_now, oFT=oFT, sum_moments=sum_moments) * prior_fun(N_now)
        N = np.hstack((N, N_now))
        p = np.hstack((p, p_now))
        if np.any(np.isnan(p_now)):
            raise ValueError, 'p_now contains nan'
        if np.any(N==0):
            raise ValueError, 'N contains zero'            
    
    # Normalize and return
    return N, p/sum(p)

def SIR_simplex_sample(mu, tau, cutoff, sum_val, N, N_proposals=1000, N_samps=1000):
    """
    Returns raw log-weights, indices chosen and SIR samples for sets of N draws
    from a truncated lognormal distribution, conditioned so that their sum is
    equal to sum_val.
    
    This SIR algorithm will fail miserably unless sum_val is relatively likely
    given N and the parameters of the lognormal distribution.
    
    :Parameters:
      - mu : float
        The mean parameter.
      - tau : float
        The precision parameter.
      - cutoff : float
        The truncation value.
      - sum_val : float
        The sum that is being conditioned on.
      - N : integer
        The number of variates in each vector
      - N_proposals : integer
        The number of vectors to propose.
      - N_samps : integer
        The number of vectors to return.
    """
    # Draw samples, compute missing values, evaluate log-weights.
    samps = np.exp(geto_truncnorm(mu, tau, log(cutoff), (N-1,N_proposals)))
    last_vals = sum_val - np.sum(samps,axis=0)
    weights = np.array([pm.lognormal_like(last_val_now, mu, tau) for last_val_now in last_vals])

    # Check that there are at least some positive weights.
    where_pos = np.where(weights>-1e308)
    if len(where_pos[0])==0:
        raise RuntimeError, 'No weights are positive. You have used a shitty value for N.'

    # Normalize and exponentiate log-weights.
    weights[where(last_vals>cutoff)]=-np.Inf
    weights -= log_sum(weights)
    
    # Append missing values to samples.
    samps = np.vstack((samps,last_vals))
    
    # Slice and return.
    ind=np.array(pm.rcategorical(p=np.exp(weights),size=N_samps),dtype=int)
    return weights, ind, samps[:,ind]
    
def settlement_size_samples(mu, tau, cutoff, sum_mu, sum_tau, pop_accounted, N):
    """
    Returns N samples from the distribution of unsampled settlement sizes.
    Settlement sizes are drawn from a truncated lognormal distribution 
    conditional on their sum being equal to sum_val.
    
    At the SIR stage, 100 samples are proposed and 10 are retained.
    
    :Parameters:
    - mu : float
      The mean parameter.
    - tau : float
      The precision parameter.
    - cutoff : float
      The truncation value.
    - sum_mu : float
      The mean of the lognormal distribution of total population.
    - sum_tau : float
      The precision parameter of the lognormal distribution of total population.
    - pop_accounted : integer
      The total population accounted for by the GRUMP urban extents.
    - N : integer
      The number of samples to return.    
    """

    N_sum_vals = N/10
    
    # Compute moments and characteristic function for single population size,
    # to be used in all posterior evaluations.
    lnorms = np.exp(geto_truncnorm(mu, tau, log(cutoff), 10000))        
    sum_moments = np.mean(lnorms), np.var(lnorms)
    oFT = robust_CF(mu, tau, cutoff)
    
    # Generate values for total population in region not accounted for by
    # GRUMP urban extents.
    sum_vals = pm.rlognormal(sum_mu, sum_tau, size=N_sum_vals)-pop_accounted
    where_not_OK = np.where(sum_vals < 0)
    while len(where_not_OK[0]) > 0:
        sum_vals[where_not_OK] = pm.rlognormal(sum_mu, sum_tau, size=len(where_not_OK[0]))-pop_accounted
        where_not_OK = np.where(sum_vals < 0)        
    
    # Create 10 samples using SIR for each sum.
    samps = []
    for sum_val in sum_vals:
        
        tries = 0
        while tries < 10:
            try:
                # Find posterior of N given this sum, and draw a single sample from it.
                Nmesh, p = robust_posterior(mu, tau, cutoff, sum_val, prior_fun=None, sum_moments=sum_moments, oFT=oFT)
                N = Nmesh[int(pm.rcategorical(p))]

                # Draw 10 samples for the sizes of the constituent populations given their number and
                # the total population size.
                w,i,s = SIR_simplex_sample(mu, tau, cutoff, sum_val, N, N_proposals=1000, N_samps=10)
                break
            except RuntimeError:
                print 'Failed at N=%f, Nmesh=%s, p=%s. Trying again'%(N,Nmesh,p)
                tries += 1
                if tries==10:
                    import sys
                    a,b,c = sys.exc_info()
                    raise a,b,c
                
        samps.extend(list(s.T))

    # Return, you're done!
    return samps
        


        
