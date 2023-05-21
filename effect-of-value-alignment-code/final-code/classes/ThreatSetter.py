import numpy as np
from numpy.random import default_rng

class ThreatSetter:
    
    def __init__(self, N=40, prior=0.3, seed=None):

        self.N = N
        self.prior_single = prior        

        # Initialize the repeated priors list
        self.prior = np.ones((self.N,), dtype=float) * prior

        self.seed = seed

    def setThreats(self):
        # We have priors. We can set danger levels around these priors (Beta distribution should work)
        # Threat presence should be dependent on the danger levels
        # After scan threat levels should be close to the actual presence of threats

        # Setting the acutal presence of threats based on the generated noisy danger level data
        self.threats = np.zeros((self.N,), dtype=int)

        if self.seed is not None:
            self.rng = default_rng(self.seed)
        else:
            self.rng = default_rng()

        self.threats = self.rng.binomial(1, self.prior_single, size=self.N)

        self.setAfterScanLevels()
        
    def setAfterScanLevels(self):

        self.after_scan = np.zeros_like(self.prior)

        for i in range(self.N):
            if not self.threats[i]:
                # Rather than a uniform distribution, we may want to use a left-skewed distribution
                self.after_scan[i] = self.rng.beta(4, 28)      # This ensures that the mode of the distribution is at 0.1
                # self.after_scan[i] = self.rng.uniform(0., 0.3)
            else:
                # Rather than a uniform distribution, we may want to use a right-skewed distribution
                self.after_scan[i] = self.rng.beta(28, 4)      # This ensures that the mode of the distribution is at 0.9
                # self.after_scan[i] = self.rng.uniform(0.7, 1.0)

def main():
    setter = ThreatSetter()
    setter.setThreats()

    for i in range(setter.N):
        print(setter.threats[i], setter.prior[i], setter.after_scan[i])
    
if __name__ == "__main__":
    main()
