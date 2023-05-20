import numpy as np
from numpy.random import default_rng


class ThreatSetter:
    """
    Class to set all threat levels: prior_levels, after_scan_levels, and threats
    """
    def __init__(self, num_sites=100, region_size=10, priors=None,
                 k1=10., k2=10., scanner_accuracy=0.8, seed=None):
        """
        :param num_sites: number of sites in the mission
        :param region_size: size of a region with the same prior level of threat
        :param priors: the prior threat level of a region. The length of this list should be equal
                    to num_sites // region_size
        ;param k1: the parameter for setting prior threat levels
        :param k2: the parameter for setting after scan threat levels
        :param scanner_accuracy: accuracy of the scanner of the drone
        :param seed: the seed for the random number generator
        """

        self.after_scan = None
        self.danger_levels = None
        self.threats = None
        if priors is None:
            priors = [0.3, 0.8, 0.5, 0.5, 0.7, 0.2, 0.4, 0.5, 0.9, 0.6]
        assert num_sites // region_size == len(priors), "Length of priors does not match the region size"

        self.N = num_sites
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

    def set_threats(self):
        """
        Sets all threat levels (prior levels, after scan levels, threat presence)
        """
        # We have priors. We can set danger levels around these priors (Beta distribution should work)
        # Threat presence should be dependent on the danger levels
        # After scan threat levels should be close to the actual presence of threats
        
        self.set_danger_levels()
        
        # Setting the actual presence of threats based on the generated noisy danger level data
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
        
        self.set_after_scan_levels()
    
    def set_danger_levels(self):
        """
        Sets danger levels according to the prior list. Essentially adds some randomness to the regions' prior
        threat level value
        """

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
        
    def set_after_scan_levels(self):
        """
        Helper function to set the after scan threat level values
        """
        # The robot scans and gets a value of probability of threat from sensor readings
        # This is updated by multiplying by the prior probability of threat presence
        self.after_scan = np.zeros_like(self.danger_levels)

        if self.seed is not None:
            rng = default_rng(self.seed)
        else:
            rng = default_rng()
        
        for i in range(self.N):

            self.after_scan[i] = rng.beta(self.k2 * self.danger_levels[i], self.k2 * (1.-self.danger_levels[i]))


def main():
    """
    Main function is for debugging.
    """
    setter = ThreatSetter()
    setter.set_threats()
    
    for i in range(setter.N):
        print(setter.threats[i], setter.priors[i//setter.region_size], setter.after_scan[i])


if __name__ == "__main__":
    main()
