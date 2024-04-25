from .policy import Policy
from .utils.grad_desc_utils import *
from .utils.utils_gen import *


class TwoPointsEstimationDescent(Policy):
    """
    Descent Policy using an estimation step based on two points, with homothetic perturbation
  ^ """

    def __init__(self, dim, init_point, center_point, radius, constraints, alpha=1, sigma=0.1, l_rate=None,
                 step_est=None):
        self.dim = dim
        self.radius = radius
        self.ini = init_point
        self.step_est_0 = radius / 100
        self.center_point = np.array(center_point)
        self.sign = 1
        self.alpha = alpha
        if step_est is not None and l_rate is not None:
            self.step_est = step_est
            self.l_rate = l_rate
        self.t = 1
        self.sigma = sigma
        if step_est is None:
            self.step_est = self.step_est_0
        self.L = 1
        self.l_rate = 2 / self.alpha
        self.dist = np.zeros(2)
        self.constraints = constraints
        self.iterate_bool = False
        self.sign = 1
        pass

    def make_name_label(self):
        inter = "Gradient_Desc_Two_points_"
        label = "2-points GD"
        name = string_to_name(inter)
        return name, label

    def startGame(self):
        self.grad = np.zeros(self.dim)  # estimated gardient
        self.sign = 1  # we are evaluating f(X + sign * yd)
        self.half_sum = 0  # to keep in memory f(X + yd) before evaluating f(X - yd)
        self.dir = np.zeros(self.dim)
        self.r = 1
        self.X = self.ini
        self.step_est_0 = self.radius / 2
        self.step_est = self.step_est_0

    def choice(self):
        self.dir = np.array(random_unit(self.dim))
        res = self.X + self.sign * self.step_est * self.dir + 1 / self.radius * self.step_est * (
                    self.center_point - self.X)
        return res

    def getValue(self, value):
        value = -value
        self.half_sum += self.sign * value
        if self.sign == -1:
            self.t += 1
            self.grad = 1 / (2 * self.step_est) * self.half_sum * self.dir
            unprojected_x = self.X - self.l_rate * self.grad
            self.X = projection_simplex(unprojected_x, self.dim)
            self.step_est = (self.t ** (-1 / 3)) * self.step_est_0
            self.l_rate = 2 / (self.alpha * self.t)
            self.half_sum = 0
        self.sign *= -1
