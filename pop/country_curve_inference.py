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

#!/usr/bin/env python
# encoding: utf-8
"""
country_curve_inference.py

Created by Anand Patil on 2008-11-25.
Copyright (c) 2008. All rights reserved.
"""

import pymc as pm
import numpy as np
import scipy
from scipy import special
from scipy.special import gammaln, erf, erfc
from attribute_pops_no_MCMC import likelihood_of_sum
import pylab as pl

# ==========================
# = PyMC probability model =
# ==========================
def make_model(data, tot_size):
    data = np.sort(data)
    n=len(data)
    missing_size = tot_size - np.sum(data)
    cutoff = data[0]

    moo = pm.Uninformative('moo',value=np.log(cutoff / 10.))    
    N = pm.Uniform('N',n+1,1e6,value=missing_size/np.exp(moo.value))
    tau = pm.Uniform('tau',0,1e6,1)

    @pm.observed
    @pm.stochastic
    def x(N=N, mu=moo, tau=tau, n=n, value=np.log(data)):
       k = N-n
       dev = (value[0]-mu)*np.sqrt(tau)   
       out = gammaln(N+1) - gammaln(k) + (k-1)*np.log(pm.utils.normcdf(dev)) + pm.normal_like(value, mu, tau)
       if np.isnan(out):
           raise ValueError
       return out

    @pm.observed
    @pm.stochastic
    def missing_pop(N=N, n=n, mu=moo, tau=tau, cutoff = cutoff, value=missing_size, observed=True):
        # from IPython.Debugger import Pdb
        # Pdb(color_scheme='Linux').set_trace()   
        return likelihood_of_sum(mu, tau, cutoff, value, np.round([N-n]), logl=True)
        
    # from IPython.Debugger import Pdb
    # Pdb(color_scheme='Linux').set_trace()   
    
    return locals()


# ====================================================
# = Uncomment to use maximum a posteriori estimation =
# ====================================================
if __name__ == '__main__':
    
    # =================
    # = Simulate data =
    # =================
    # 'Hidden' unknown parameters
    N_ = 1000
    mu_ = 1
    tau_ = 1

    # Number of settlements in dataset
    n = 100

    # 'Hidden' full dataset
    data_ = np.exp(np.random.normal(size=N_)/np.sqrt(tau_) + mu_)

    # Top n
    data = np.sort(data_)[-n:]

    # Total population size and population size to be accounted
    tot_size = np.sum(data_)
    missing_size = tot_size - np.sum(data)
    
    M = pm.MAP(make_model(1000, data))
    M.fit()

# =========================================
# = Uncomment to use normal approximation =
# =========================================
# N = pm.NormApprox([N,mu,tau,x,missing_pop])
# N.fit()

# =========================
# = Uncomment to use MCMC =
# =========================
# M = pm.MCMC([N,mu,tau])
# M.use_step_method(pm.AdaptiveMetropolis, [N,mu,tau], delay=10000)
# M.isample(100000,0,100)
# pm.Matplot.plot(M)