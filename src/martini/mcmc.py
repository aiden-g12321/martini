"""
Metropolis-Hastings MCMC
"""

import numpy as np


class MetropolisHastings:

    def __init__(self, target_lnpdf, proposal_dist):
        self.target_lnpdf = target_lnpdf
        self.proposal_dist = proposal_dist
        
        self.samples = None
        self.lnpdf_samples = None
        self.acceptance_rate = None

    def run(self, num_samples):

        # initialize chain
        samples = np.zeros((num_samples, self.proposal_dist.ndim))
        lnpdf_samples = np.zeros(num_samples)

        # start at draw from proposal
        samples[0] = self.proposal_dist.sample()[0]
        lnpdf_samples[0] = self.target_lnpdf(samples[0])

        # track acceptance rate
        accept_count = 0
        reject_count = 0

        # main MCMC loop
        for i in range(num_samples - 1):
            
            # update progress
            print(f'{np.round(i / num_samples * 100, 2)}%', end='\r')
            
            # proposal
            x_prop = self.proposal_dist.sample()[0]
            lnpdf_prop = self.target_lnpdf(x_prop)

            # acceptance ratio
            log_accept_ratio = lnpdf_prop - lnpdf_samples[i] \
                + self.proposal_dist.logpdf(samples[i]) - self.proposal_dist.logpdf(x_prop)

            if np.log(np.random.uniform()) < log_accept_ratio: # accept
                samples[i + 1] = x_prop
                lnpdf_samples[i + 1] = lnpdf_prop
                accept_count += 1
            else:   # reject
                samples[i + 1] = samples[i]
                lnpdf_samples[i + 1] = lnpdf_samples[i]
                reject_count += 1
        
        # calculate acceptance rate
        acceptance_rate = accept_count / (accept_count + reject_count)
            
        self.samples = samples
        self.lnpdf_samples = lnpdf_samples
        self.acceptance_rate = acceptance_rate
        
        return None
