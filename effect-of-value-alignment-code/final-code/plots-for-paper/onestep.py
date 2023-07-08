import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='paper', style='white')


def main():
    
    trust = np.linspace(0., 1., 100)
    d = 0.7
    wh_r = 0.8
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
    
    expected_one_step_0 = -d*(trust + (1-trust)*p0) * wh_r*h - (1 - trust) * p1 * wc_r *c
    expected_one_step_1 = -d*(1-trust)*p0*wh_r*h - (trust + (1-trust)*p1)*wc_r*c
    
    fig, ax = plt.subplots(layout='tight', figsize=(7, 4))
    ax.plot(trust, expected_one_step_1, lw=2, c='black', label=r'$a=1$')
    ax.plot(trust, expected_one_step_0, lw=2, ls='dashed', c='black', label=r'$a=0$')
    ax.set_xlabel('Trust', fontsize=14)
    ax.set_ylabel(r'$E[R(a)]$', fontsize=14)
    ax.set_title('Expected one-step rewards - Bounded Rationality Disuse', fontsize=16)
    ax.legend()

    expected_one_step_0_rev = -trust*wh_r*h*d - (1.-trust)*wc_r*c
    expected_one_step_1_rev = -(1.-trust)*wh_r*h*d - trust*wc_r*c

    fig, ax = plt.subplots(layout='tight')
    ax.plot(trust, expected_one_step_1_rev, lw=2, c='black', label=r'$a=1$')
    ax.plot(trust, expected_one_step_0_rev, lw=2, ls='dashed', c='black', label=r'$a=0$')
    ax.set_xlabel('Trust')
    ax.set_ylabel(r'$E[R(a)]$')
    ax.set_title('Expected one-step rewards - Reverse Psychology')
    ax.legend()

    fig, ax = plt.subplots(layout='tight')
    # ax.plot(trust, expected_one_step_1, lw=2, c='black', label=r'BRD $a=1$')
    # ax.plot(trust, expected_one_step_0, lw=2, ls='dashed', c='black', label=r'BRD $a=0$')
    # ax.plot(trust, expected_one_step_1_rev, lw=2, c='red', label=r'RP $a=1$')
    # ax.plot(trust, expected_one_step_0_rev, lw=2, ls='dashed', c='red', label=r'RP $a=0$')
    ax.plot(trust, expected_one_step_1, lw=2, c='black', label='Bounded Rationality')
    ax.plot(trust, expected_one_step_1_rev, lw=2, c='red', label="Reverse Psychology")
    ax.plot(trust, expected_one_step_1, lw=2, c='black', label=r'$a=1$')
    ax.plot(trust, expected_one_step_0, lw=2, ls='dashed', c='black', label=r'$a=0$')
    ax.plot(trust, expected_one_step_0_rev, lw=2, ls='dashed', c='red')
    ax.plot()
    ax.set_xlabel('Trust')
    ax.set_ylabel(r'$E[R(a)]$')
    ax.set_title('Expected one-step rewards - Combined')
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()

    