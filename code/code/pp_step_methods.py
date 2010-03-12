import numpy as np
import pymc as pm
from pointprocess import *



__all__ = ['PPBirthDeathMetropolis', 'PPMoveMetropolis']
__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'



class PPBirthDeathMetropolis(pm.StepMethod):
    """
    Updates a point process's value using birth/death proposals.
    Sufficient to explore the posterior, but more efficient when used
    in conjunction with PPMoveMetropolis:
    
    M.use_step_method(PPBirthDeathMetropolis, P)
    M.use_step_method(PPMoveMetropolis, P)
    
    :Parameters:
      - pp : point process
        The stochastic variable to update
      - birth_prob : float between 0 and 1
        Every time step() is called, a birth is proposed with this probability.
        Otherwise a death is proposed.
      - rbirth_density : function
        Should take pp's density parameters (pp.density_params) and return the
        coordinates of a new proposed point. 
      - birth_density_logp : function
        Should take the coordinates of a point and pp's density parameters and
        return the log-density corresponding to rbirth_density.
      - verbose : integer
        The level of verbosity of the step method.
    
    :Reference:
      Jesper Moller and Rasmus Plenge Waagepetersen. Statistical inference
      and simulation for spatial point processes. CRC Press, 2004.
      
    :SeeAlso: PPMoveMetropolis
    """
    def __init__(self, pp, birth_prob=.5, rbirth_density=None, birth_density_logp=None, verbose=0):

        pm.StepMethod.__init__(self, [pp], verbose=verbose)

        self.pp = pp
        self.stochastic = pp
        self.lbp = pm.logit(birth_prob)
        self.birth_prob = birth_prob
        self.rbirth_density = rbirth_density
        self.birth_density_logp = birth_density_logp
        
        if (rbirth_density is not None) + (birth_density_logp is not None) == 1:
            raise ValueError, 'rbirth_density and birth_density_logp must both be provided.'

        if rbirth_density is None:
            if pp.bounding_box is None:
                raise ValueError, 'Either birth_density must be provided or point process %s must have a bounding box.'
        
        # _accepted, _rejected and _asf are used to tune moves.
        self._rejected = 0
        self._accepted = 0  
        self._asf = 1      
        self._id = 'PPBirthDeath_%s'%self.pp.__name__        

    def step(self):
        # Propose birth or death
        
        birth_prob = self.birth_prob
        loglike = self.loglike
        
        if np.random.random() < birth_prob:
            if self.verbose > 0:
                print '\t' + self._id + ' proposing birth, current length is',len(self.pp.value)

            logp = self.pp.log_subset_density_fun((), self.pp.value, **self.pp.density_params.value)
            new_pt = self.birth_propose()   
            try:         
                logp_p = self.pp.log_subset_density_fun((new_pt,), self.pp.value, **self.pp.density_params.value)
            except pm.ZeroProbability:
                if self.verbose > 0:
                    print '\t' + self._id + 'Rejecting birth proposal due to ZeroProbability'
                self._rejected += 1
                return
            if self.verbose > 1:
                print '\t\t' + self._id + 'Proposing birth at',new_pt            
            self.pp.value = self.pp.value + (new_pt,)

            # Log-density of forward and backward jumps
            lp_for = self.birth_logp(birth_prob, new_pt)
            lp_bak = self.death_logp(birth_prob, len(self.pp.value))
            
        else:
            if self.verbose > 0:
                print '\t' + self._id + ' proposing death, current length is',len(self.pp.value)

            if len(self.pp.value)==0:
                # Nothing to kill...
                lp_for = 0
                lp_bak = 0

                if self.verbose > 0:
                    print '\t' + self._id + 'Length is already 0, doing nothing'
                return         

            i = self.death_propose()
            killed_pt = self.pp.value[i]
            self.pp.value = self.pp.value[:i] + self.pp.value[i+1:]            
            logp_p = self.pp.log_subset_density_fun((), self.pp.value, **self.pp.density_params.value)            
            logp = self.pp.log_subset_density_fun((killed_pt,), self.pp.value, **self.pp.density_params.value)
            if self.verbose > 1:
                print '\t\t' + self._id + 'Proposing death at',killed_pt
            
            # Log-density of forward and backward jumps
            lp_for = self.death_logp(birth_prob, len(self.pp.value)+1)
            lp_bak = self.birth_logp(birth_prob, killed_pt)
        
        try:    
            loglike_p = self.loglike
        except pm.ZeroProbability:
            if self.verbose > 0:
                print '\t' + self._id + 'rejecting due to ZeroProbability'
            self.pp.revert()
            self._rejected += 1
            return

        if self.verbose > 1:
            print '\t\t' + self._id + ' forward log-probability: ', lp_for
            print '\t\t' + self._id + ' backward log-probability: ', lp_bak
            print '\t\t' + self._id + ' logp_p: ', logp_p
            print '\t\t' + self._id + ' logp: ', logp            
            print '\t\t' + self._id + ' loglike_p', loglike_p
            print '\t\t' + self._id + ' loglike', loglike            
        
        if np.log(np.random.random()) < logp_p + loglike_p + lp_bak - logp - loglike - lp_for:
            if self.verbose > 0:
                print '\t' + self._id + 'accepting'
            self._accepted += 1
        else:
            if self.verbose > 0:
                print '\t' + self._id + 'rejecting'            
            self.pp.revert()
            self._rejected += 1
            

    def birth_logp(self, birth_prob, new_pt):
        if self.birth_density_logp is not None:
            return np.log(birth_prob) + self.birth_density_logp(new_pt, **self.pp.density_params.value)
        else:
            return np.log(birth_prob) - np.log(np.prod(self.pp.bounding_box[1] - self.pp.bounding_box[0]))
            
    def death_logp(self, birth_prob, n):
        return np.log(1-birth_prob) - np.log(n)
        
    def birth_propose(self):
        # Either propose births from provided distribution
        if self.rbirth_density is not None:
            new_pt = self.rbirth_density(**self.pp.density_params.value)
        # Or uniformly on the bounding box.
        else:
            new_pt = pm.runiform(self.pp.bounding_box[0], self.pp.bounding_box[1])
        return new_pt

    def death_propose(self):
        i = np.random.randint(len(self.pp.value))
        return i



class PPMoveMetropolis(pm.StepMethod):
    """
    Updates a point process's value using move proposals. Not sufficient
    to explore the posterior, should be used in conjunction with 
    PPBirthDeathMetropolis.
    
    :Parameters:
      - pp : point process
        The point process whose value is to be updated.
      - move_sig : float
        The standard deviation of maves.
      - exp_number_moved : positive integer
        Points will be selected randomly for move proposals in such a way that
        the expected number of points selected is exp_number_moved.
      - verbose : integer
        Level of verbosity of the step method.
        
    :Reference:
      Jesper Moller and Rasmus Plenge Waagepetersen. Statistical inference
      and simulation for spatial point processes. CRC Press, 2004.
      
    :SeeAlso: PPBirthDeathMetropolis
    """
    def __init__(self, pp, move_sig=1, exp_number_moved=10, verbose=0):
        pm.StepMethod.__init__(self, pp, verbose=verbose)   
        
        self.pp = pp
        self.stochastic = self.pp
        self.move_sig = move_sig
        self.exp_number_moved = exp_number_moved
        
        self._rejected = 0
        self._accepted = 0  
        self._asf = 1
        self._id = 'PPMove_%s'%self.pp.__name__
        
    def step(self):

        if len(self.pp.value) == 0:
            if self.verbose > 1:            
                print '\t'+self._id + 'length of pp is 0, returning early'
            return

        if self.verbose > 1:
            print '\t\t'+self._id + 'step %i'%i
        
        loglike = self.loglike
        if self.verbose > 1:
            print '\t\t'+self._id + 'log-likelihood before proposal is %f'%loglike

        list_value = np.array(self.pp.value)
        move_prob = min(1, len(list_value)/self.exp_number_moved)            
        move_flags = np.random.random(size=len(list_value)) < move_prob
        N_move = np.sum(move_flags)

        this_value = np.empty((N_move, list_value.shape[1]))
        other_value = np.empty((list_value.shape[0]-N_move, list_value.shape[1]))
        from_ind = np.empty(N_move, dtype=int)
        
        if N_move==0:
            return
        if self.verbose > 1:
            print '\t\t'+self._id + 'proposing moves at %i points of %i'%(N_move, len(self.pp.value))

        j=0
        k=0
        for l in xrange(len(list_value)):
            if move_flags[l]:
                this_value[j]=list_value[l]
                from_ind[j]=l
                j+=1
            else:
                other_value[k]=list_value[l]
                k+=1
        
        logp = self.pp.log_subset_density_fun(this_value, other_value, **self.pp.density_params.value)
        if self.verbose > 1:
            print '\t\t'+self._id + 'log-density of current subset is %f'%logp

        this_value = np.random.normal(this_value, self.move_sig * self._asf)
        list_value[from_ind] = this_value[:]
        
        try:    
            logp_p = self.pp.log_subset_density_fun(this_value, other_value, **self.pp.density_params.value)
        except pm.ZeroProbability:
            print '\t'+self._id+'rejecting due to ZeroProbability'
            self._rejected += 1
            return
            
        if self.verbose > 1:
            print '\t\t'+self._id + 'log-density of proposed value at current subset is %f'%logp
    
        self.pp.value = tuple(list_value)
        
        try:
            loglike_p = self.loglike
        except pm.ZeroProbability:
            if self.verbose > 0:
                print '\t'+self._id + 'rejecting due to ZeroProbability'
            self._rejected += 1
            return
            
        if self.verbose > 1:
            print '\t\t'+self._id + 'log-likelihood after proposal is %f'%loglike
            
        if np.log(np.random.random()) < logp_p + loglike_p - logp - loglike:
            if self.verbose > 1:
                print '\t\t'+self._id + 'accepting'
            self._accepted += 1
        else:
            if self.verbose > 1:
                print '\t\t'+self._id + 'rejecting'
            self._rejected += 1
            self.pp.revert()
            

    
if __name__ == '__main__':
    from pylab import *
    
    def la(x, mu, tau, lam):
        if len(x) > 0:
            return pm.mv_normal_like(np.array(x), mu, tau) + len(x)*np.log(lam)
        else:
            return 0

    lam = pm.Uninformative('lam',100)
    PP = PoissonProcess('PP', la, value=(), bounding_box = [(-10.,)*3,(10.,)*3], mu=np.ones(3), tau = np.eye(3), lam=lam)
    M = PPMoveMetropolis(PP, exp_number_moved = 10, move_sig=1,verbose=0)
    def rmvn_wrap(mu, tau, lam):
        return pm.rmv_normal(mu, tau)
    def mvnl_wrap(x,mu,tau,lam):
        return pm.mv_normal_like(x,mu,tau)
    BD = PPBirthDeathMetropolis(PP, rbirth_density = rmvn_wrap, birth_density_logp = mvnl_wrap, birth_prob=.1)
    # BD = PPBirthDeathMetropolis(PP, birth_prob=.1)    
    len_trace = []
    # pp_trace = []
    for i in xrange(1000):
        BD.step()
        M.step()
        if i%1000==0:
            M.tune()
            BD.tune()
        if i>5000:
            len_trace.append(len(PP.value))
            # pp_trace.append(PP.value)
            
    print mean(len_trace)