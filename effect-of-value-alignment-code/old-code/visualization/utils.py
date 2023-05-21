import numpy as np

def get_expected_reward_robot(wh_rob, wh_hat, h, c, kappa, trust, threat_level, wt_rob):
    
    wc_rob = 1.0 - wh_rob    
    wc_hat = 1.0 - wh_hat
    
    # Disuse probabilities
    r0_hum = threat_level * wh_hat * h
    r1_hum = wc_hat * c
    
    prob_0 = np.exp(kappa * r0_hum)
    prob_1 = np.exp(kappa * r1_hum)
    
    prob_0 = prob_0 / (prob_0 + prob_1)
    prob_1 = 1.0 - prob_0
    
    # Expected reward to recommend to USE RARV
    r1 = threat_level * prob_0 * wh_rob * h + (trust + prob_1) * wc_rob * c + wt_rob * (r1_hum > r0_hum)

    # Expected reward to recommend to NOT USE RARV
    r0 = threat_level * (trust + prob_0) * wh_rob * h + prob_1 * wc_rob * c + wt_rob * (r0_hum > r1_hum)

    return r0, r1

def get_expected_reward_human(wh_hum, h, c, threat_level):
    
    wc_hum = 1.0 - wh_hum
    
    r0 = threat_level * wh_hum * h
    r1 = wc_hum * c * np.ones_like(threat_level)

    return r0, r1
