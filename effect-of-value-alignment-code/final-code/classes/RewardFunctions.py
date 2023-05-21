"""Classes for keeping rewards consistent through the main code"""

class RewardsBase:

    def __init__(self) -> None:
        pass
    
    def reward(self, health, time, house):
        raise NotImplementedError

class Constant:
    
    def __init__(self, hl, tc) -> None:
        self.hl = hl
        self.tc = tc
    
    def reward(self, health=None, time=None, house=None):
        return (-self.hl, -self.tc)

class Linear(RewardsBase):

    def __init__(self) -> None:
        super().__init__()
    
    def reward(self, health, time, house=None):
        return (health, -time)

class Affine(RewardsBase):

    def __init__(self, max_health) -> None:
        super().__init__()
        self.max_health = max_health
    
    def reward(self, health, time, house=None):

        return (health - self.max_health, -time)

class Inverse(RewardsBase):

    def __init__(self, min_health, max_health, max_time) -> None:
        super().__init__()
        self.min_health = min_health
        self.max_health = max_health
        self.max_time = max_time
    
    def reward(self, health, time, house=None):

        if health <= 0:
            health = self.min_health + 1

        h = -self.max_health / (health - self.min_health)
        c = -min(time, self.max_time)

        return (h, c)
