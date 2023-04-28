import numpy as np


class ThreatSetter:

    def __init__(self, num_sites: int, threat_level: float = 0.5, seed: int = 123):
        self.N = num_sites
        self.prior_levels = np.zeros((self.N,), dtype=float)
        self.after_scan_levels = np.zeros((self.N,), dtype=float)
        self.threats = np.zeros((self.N,), dtype=int)
        self.threat_level = threat_level
        self.rng = np.random.default_rng(seed=seed)

    def set_threats(self):

        for i in range(self.N):
            self.prior_levels[i] += self.threat_level + self.rng.normal(scale=0.1)
            r = self.rng.uniform()
            if r <= self.prior_levels[i]:
                self.threats[i] = 1
                self.after_scan_levels[i] = self.rng.beta(10, 2)
            else:
                self.threats[i] = 0
                self.after_scan_levels[i] = 1. - self.rng.beta(10, 2)
