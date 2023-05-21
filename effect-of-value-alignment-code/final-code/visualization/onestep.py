import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='paper', style='white')

def main():
    
    trust = np.linspace(0., 1., 100)
    d = 0.7
    wh_r = 0.7
    wc_r = 1 - wh_r
    h = 10
    c = 10
    
    wh_h = 0.7
    wc_h = 1 - wh_h
    
    
    kappa = 0.2
    r0_hum = -d * wh_h * h
    r1_hum = -wc_h * c
    
    p0 = 1. / (1 + np.exp(kappa * (r1_hum - r0_hum)))
    p1 = 1 - p0
    
    E0 = -d*(trust + (1-trust)*p0) * wh_r*h - (1 - trust) * p1 * wc_r *c
    E1 = -d*(1-trust)*p0*wh_r*h - (trust + (1-trust)*p1)*wc_r*c
    
    fig, ax = plt.subplots(layout='tight')
    
    ax.plot(trust, E1, lw=2, c='black', label=r'$a=1$')
    ax.plot(trust, E0, lw=2, ls='dashed', c='black', label=r'$a=0$')
    ax.set_xlabel('Trust')
    ax.set_ylabel(r'$E[R(a)]$')
    ax.set_title('Expected one-step rewards')
    ax.legend()
    
    plt.show()

if __name__ == "__main__":
    main()

    