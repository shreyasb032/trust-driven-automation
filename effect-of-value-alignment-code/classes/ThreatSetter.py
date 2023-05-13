import numpy as np
from numpy.random import default_rng


class ThreatSetter:
    
    def __init__(self, N=100, region_size=10, priors=[0.3, 0.8, 0.5, 0.5, 0.7, 0.2, 0.4, 0.5, 0.9, 0.6],
                 k1=10., k2=10., scanner_accuracy=0.8, seed=None):

        assert N // region_size == len(priors), "Length of priors does not match the region size"

        self.N = N
        self.region_size = region_size
        self.num_regions = int(self.N // self.region_size)
        
        # Prior information about the probability of threat in a region
        self.priors = priors

        # For setting the noisy danger levels around the prior
        self.k1 = k1

        # For the probability of scans
        self.k2 = k2
        
        # Initialize the repeated priors list
        self.prior_levels = np.zeros((self.N, ), dtype=float)

        # For after scan probs
        self.scanner_accuracy = scanner_accuracy
        
        self.seed = seed

    def setThreats(self):
        # We have priors. We can set danger levels around these priors (Beta distribution should work)
        # Threat presence should be dependent on the danger levels
        # After scan threat levels should be close to the actual presence of threats
        
        self.setDangerLevels()
        
        # Setting the acutal presence of threats based on the generated noisy danger level data
        self.threats = np.zeros((self.N,), dtype=int)

        if self.seed is not None:
            rng = default_rng(self.seed)
        else:
            rng = default_rng()

        for i in range(self.N):
            r = rng.uniform(0, 1)
            if r < self.danger_levels[i]:
                self.threats[i] = 1
            else:
                self.threats[i] = 0
        
        self.setAfterScanLevels()
    
    def setDangerLevels(self):
        self.danger_levels = np.zeros((self.N,))
        
        if self.seed is not None:
            rng = default_rng(self.seed)
        else:
            rng = default_rng()
        
        # Setting the danger levels noisily around the prior distribution
        for i in range(0, self.N, self.region_size):

            prior = self.priors[i//self.region_size]
            self.prior_levels[i:i+self.region_size] = prior
            if prior == 0 or prior == 1:
                # For debugging. When the prior is set to 0 or 1, the beta distribution will not work
                # Here we set the danger levels to 0 or 1 exactly.
                self.danger_levels[i:i+self.region_size] = prior
            else:
                self.danger_levels[i:i+self.region_size] = rng.beta(self.k1 * prior, self.k1 * (1 - prior), self.region_size)
        
    def setAfterScanLevels(self):
        
        # The robot scans and gets a value of probability of threat from sensor readings
        # This is updated by multiplying by the prior probability of threat presence
        self.after_scan = np.zeros_like(self.danger_levels)

        if self.seed is not None:
            rng = default_rng(self.seed)
        else:
            rng = default_rng()
        
        for i in range(self.N):

            # saying that there is a threat = accuracy * threat + (1 - accuracy)*(1-threat)
            # saying that there is not a threat = (1-accuracy) * threat + accuracy * (1 - threat)
            
            # threat_scanned = self.scanner_accuracy * self.threats[i] + (1. - self.scanner_accuracy) * (1 - self.threats[i])
            # nothreat_scanned = 1.0 - threat_scanned

            # if nothreat_scanned == 0 or nothreat_scanned == 1:
            #     # Debugging case, when the scanner accuracy is 100%, we detect all threats correctly, with 100% probability
            #     self.after_scan[i] = threat_scanned
            # else:
            #     # Add noise around these values and use the prior to get the posterior
            #     self.after_scan[i] = rng.beta(self.k2 * threat_scanned, self.k2 * nothreat_scanned) #* self.priors[i//self.region_size]

            self.after_scan[i] = rng.beta(self.k2 * self.danger_levels[i], self.k2 * (1.-self.danger_levels[i]))


def main():
    setter = ThreatSetter()
    setter.setThreats()
    
    for i in range(setter.N):
        print(setter.threats[i], setter.priors[i//setter.region_size], setter.after_scan[i])


if __name__ == "__main__":
    main()
