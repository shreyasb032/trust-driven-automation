from classes.Learner import WeightsLearner
from classes.Human import SigmoidHuman
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='talk', style='white')

human = SigmoidHuman(kappa=1.0)
learner = WeightsLearner(human, num_workers=5)

health = 80
time = 70

learned_wh = learner.simulate_and_learn(health, time)
fig, ax = learner.plot()
plt.show()
