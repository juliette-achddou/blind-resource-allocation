import numpy as np

from Result import Result as Result
from .Environment import Environment


class game(Environment):
    """sequential allocation problem with given environment"""

    # For when we add a random translation

    def __init__(self, function, noise=None, randomization=False, randomization_step=0.05):
        self.function = function
        self.constraints = self.function.constraints
        self.opti = function.opti()
        self.noise = noise
        self.rand = randomization
        self.rand_step = randomization_step

    def play(self, policy, horizon):
        """
        Applies given policy to one allocation resource problem with a certain environment, and horizon.
        When rand is True, the given function gets translated by a random step
        :param policy: object from the policy class
        :param horizon: int
        :return: object frm the result class with all important data stored
        """
        policy.startGame()
        if self.rand:
            offset = self.rand_step * np.random.uniform(low=-1, high=1.0, size=self.function.dim)
        else:
            offset = np.zeros(self.function.dim)
        result = Result(horizon, self.constraints.dim)
        for t in range(horizon):
            choice = policy.choice()
            arm, max = self.opti
            # the new optimum is a translation of the optimum of the original function
            opti = (arm - offset, max)
            if policy.iterate_bool:
                # the iterate_bool variable is meant to identify functions like FDS ones that rely on iterates X_k
                iterate = policy.iterate
            if not self.constraints.in_constraints(choice):
                print("Out of Bounds : ", choice)
            value = self.function.evaluate(choice + offset)
            if self.noise is not None:
                noise = self.noise.draw()
                value_w_noise = value + noise
            else:
                value_w_noise = value
            policy.getValue(value_w_noise)
            if policy.iterate_bool:
                # if the iterate_bool variable is True, we also store the iterate
                result.store(t, choice, value, value_w_noise, opti, iterate)
            else:
                result.store(t, choice, value, value_w_noise, opti)
        return result
