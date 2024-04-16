from classes.Learner import WeightsLearner
from classes.Human import SigmoidHuman
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='talk', style='white')

human = SigmoidHuman(kappa=0.05)
learner = WeightsLearner(human, num_workers=5)

health = 80
time = 80

learned_wh = learner.simulate_and_learn(health, time)

print(f"dstar = {(1 - learned_wh) /learned_wh * 100}")
print(f"wh_hat = {learned_wh}")
print(f"wh = {human.get_wh(np.array([[health, time]]))}")

fig, ax = learner.plot()
plt.show()
