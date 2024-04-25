import numpy as np


class Result:
    """The Result class for analyzing the output of  experiments."""

    def __init__(self, horizon, dim):
        self.choices = np.zeros((horizon, dim))
        self.values = np.zeros(horizon)
        self.rewards = np.zeros(horizon)
        self.regret = np.zeros(horizon)
        self.distance_to_opti = np.zeros(horizon)
        self.dim = dim
        self.it_choices = np.zeros((horizon, dim))

    def store(self, t, choice, value, value_w_noise, opti, iterates=None):
        argmax, max = opti
        self.values[t] = value
        self.rewards[t] = value
        self.regret[t] = max - value
        self.distance_to_opti[t] = np.linalg.norm(choice - opti[0])
        self.choices[t] = choice
        if iterates is not None:
            self.it_choices[t] = iterates

    def getRegret(self, bestExpectation):
        return np.cumsum(self.regret)
