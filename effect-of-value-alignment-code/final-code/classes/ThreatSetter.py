import numpy as np
from numpy.random import default_rng


class ThreatSetter:
    """
    Class to set all threat levels: prior_levels, after_scan_levels, and threats
    """
    
    def __init__(self, num_sites=40, prior=0.3, seed=None):
        """
        :param num_sites: number of sites in the mission
        :param prior: the prior threat level at any of the search sites
        :param seed: a seed for the rng
        """

        self.after_scan = None
        self.threats = None
        self.rng = None
        self.N = num_sites
        self.prior_single = prior        

        # Initialize the repeated priors list
        self.prior = np.ones((self.N,), dtype=float) * prior

        self.seed = seed

    def set_threats(self):
        """
        Sets all threat levels (prior levels, after scan levels, threat presence)
        """

        # Setting the actual presence of threats based on the generated noisy danger level data
        self.threats = np.zeros((self.N,), dtype=int)

        if self.seed is not None:
            self.rng = default_rng(self.seed)
        else:
            self.rng = default_rng()

        self.threats = self.rng.binomial(1, self.prior_single, size=self.N)

        self.set_after_scan_threat_levels()
        
    def set_after_scan_threat_levels(self):
        """
        Helper function to set the after scan threat level values
        """

        self.after_scan = np.zeros_like(self.prior)

        for i in range(self.N):
            if not self.threats[i]:
                # Rather than a uniform distribution, we may want to use a left-skewed distribution
                self.after_scan[i] = self.rng.beta(4, 28)
                # This ensures that the mode of the distribution is at 0.1
            else:
                # Rather than a uniform distribution, we may want to use a right-skewed distribution
                self.after_scan[i] = 1.0 - self.rng.beta(4, 28)
                # This ensures that the mode of the distribution is at 0.9


def main():
    """
    Main function is for debugging
    """
    setter = ThreatSetter()
    setter.set_threats()

    for i in range(setter.N):
        print(setter.threats[i], setter.prior[i], setter.after_scan[i])


if __name__ == "__main__":
    main()
