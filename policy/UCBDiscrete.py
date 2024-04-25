import random as random

from .policy import Policy
from .utils.utils_gen import *


def randmax(ar):
    """
    Returns a random element of the argmaxes of an array.
    """
    argmaxes = np.where(ar == max(ar))
    return random.choice(argmaxes[0].tolist())


def cartesian_product(*arrays):
    """

    :param arrays:
    :return:
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def select_arms(step, constraints):
    res = []
    d = constraints.dim
    rand_offset = step * np.random.uniform(low=-1, high=1.0, size=d)
    x = np.arange(0, 1, step)
    a = [x] * d
    matrix_y = rand_offset + cartesian_product(*a)
    for z in matrix_y:
        if constraints.in_constraints(np.array(z)):
            res.append(z)
    ln = len(res)
    arms = np.array(res)
    return ln, arms


class UCBDiscrete(Policy):
    """
    The UCB policy was first introduced by Auer, Cesa Bianchi and Fisher.
    Here the arms are values on a regular grid in [0,1].
    """

    def __init__(self, constraints, sigma=0.1, step=None, horizon=None, dim=1):
        """
        Either alpha (smoothness parameter of the unimodal utility) is given and step_est is None and the step_est is computed as in Combes and
        Proutiere or the step_est is given and alpha is ignored
        :param alpha: smoothness of the unimodal utility
        :param horizon: horizon
        :param step: discretization step_est
        """
        self.t = 0
        self.choice_arm = 1
        self.horizon = horizon
        if step is None:
            if dim < 2:
                step = (horizon) ** (-1 / 4)
            else:
                step = (horizon) ** (-1 / (dim + 2))
        else:
            step = step
        self.step = 1 / (int(1 / step))
        self.nb_arms = int(1 / self.step) - 1
        self.arm_index = 1
        self.constraints = constraints
        self.sigma = sigma
        self.iterate_bool = False

    def set_horizon(self, horizon):
        """
        sets the horizon parameter and dependent parameters (here it is useless)
        :param horizon: horizon of the experiment
        :return: None
        """
        self.horizon = horizon
        step = (np.log(horizon) / (horizon ** (1 / 2))) ** (1 / self.alpha)
        self.step = 1 / (int(1 / step) + 1)

    def set_step(self, step):
        self.step = step

    def get_args(self):
        """
        Returns important arguments of the object as a string (for legends)
        :param self:
        :return: string describing the important arguments
        """
        return " step_est = " + str(self.step)

    def make_name_label(self):
        """
        outputs the name to include in the names of the related figures and files
        and the label for the figures
        :return: name, label
        """
        inter = "UCB" + ", step= " + str(self.step)[:min(len(str(self.step)), 3)]
        label = string_to_latex("UCB")
        name = string_to_name(inter)
        return name, label

    def startGame(self):
        """
        Initializes all important quantities to their initial values.
        """
        self.t = 0
        self.choice_arm = 1
        self.t = 1
        self.arm_index = 1
        self.nb_arms = int(1 / self.step) - 1
        self.nb_arms, self.armstable = select_arms(self.step, self.constraints)
        self.UCBtable = np.zeros(self.nb_arms)
        self.rewards = np.zeros(self.nb_arms)
        self.nbwins = np.zeros(self.nb_arms)
        self.nbplayed = np.zeros(self.nb_arms)
        self.r = random.uniform(0.0, self.step)
        # print(self.nb_arms)

    def choice(self):
        """
        Chooses an allocation based on past observations.
        Here the choice is an arm that is randomly drawn
        among those which have the best UCB.

        :Return: choice
        :rtype: float
        """
        # print("step_est", self.step_est)
        # print("nb arms = ", self.nb_arms)
        if self.t <= self.nb_arms:
            self.arm_index = self.t - 1
        else:
            self.arm_index = randmax(self.UCBtable)
        self.choice_arm = self.armstable[self.arm_index]

        return self.choice_arm

    def getValue(self, value):
        """
        Processes the observation at time t,

        Updates the number of times each arm has been played,
        the cumulated reward of each arm, and its UCB.

        :param value: Value at time t
        :type value: float
        :param m: Maximum of other bids at time t
        :type m: float
        """
        self.nbplayed[self.arm_index] += 1
        self.rewards[self.arm_index] += value
        if self.t > self.nb_arms:
            self.UCBtable = self.rewards / self.nbplayed + \
                            np.sqrt(2 * self.sigma ** 2 * np.log(self.t) / (self.nbplayed))
        self.t += 1
