import random

from .policy import Policy
from .utils.utils_DS import *
from .utils.utils_gen import *


class FDS_Seq(Policy):
    def __init__(self, alpha0, init_point, theta, c, constraints, gamma=1, K=2, D_k=None, delta=1 / 10000, sigma=0.1,
                 shuffle=True):
        self.alpha0 = alpha0
        self.dim = constraints.dim
        self.X0 = init_point
        self.K = K
        self.theta = theta
        self.gamma = gamma
        self.c = c
        self.in_constraints = constraints.in_constraints
        self.Ai = constraints.matrix_constr
        self.l = constraints.l
        self.u = constraints.u
        self.delta = delta
        self.sigma = sigma
        self.const = 1
        self.iterate_bool = True
        self.shuffle = shuffle
        self.fixed_D_k = (D_k is not None)
        if self.fixed_D_k:
            self.D_k = D_k.copy()
        else:
            self.D_k = None
        self.diff = None
        self.d = None
        self.fX = None
        self.t = None
        self.iterate = None
        self.sampling_the_iterate = None
        self.current_test_point = None
        self.N_iterate = None
        self.sum_current_test_point = None
        self.sum_iterate = None
        self.N_choice = None
        self.finished_estimating = None
        self.alpha = None
        self.rounds = None
        self.set_dk = None
        pass

    def make_name_label(self):
        if self.fixed_D_k:
            pol = "FDS-Seq-fixed-D"
        else:
            pol = "FDS-Seq"
        inter = pol + ", init = " + str(self.X0)[:min(len(str(self.X0)), 3)] + ", theta = " \
                + str(self.theta)[:min(len(str(self.theta)), 3)] + ", c = " + str(
            self.c)[:min(len(str(self.c)), 3)] + ", alpha_0 = " \
                + str(self.alpha0)[:min(len(str(self.alpha0)), 3)]
        label = string_to_latex(pol)
        name = string_to_name(inter)
        return name, label

    def startGame(self):
        """
        Initializes the variables
        :return:
        """
        self.diff = 0  # ultimately, used to compute Nk(hat f(X_k) - hat f(X_k + alpha v))
        self.d = np.zeros(self.dim)  # current direction
        self.fX = 0  # f(iterate)
        self.t = 0  # current time
        self.iterate = self.X0  # current iterate
        self.sampling_the_iterate = False  # we are either sampling the iterate or X_k + alpha v.
        self.current_test_point = 0  # X_k + alpha v
        self.N_iterate = 0  # number of times we have sampled the iterate
        self.sum_current_test_point = 0  # sum (f(X_k + alpha v) + epsilon_t)
        self.sum_iterate = 0  # sum (f(X_k) + epsilon'_t)
        self.N_choice = 0  # number of times we have sampled X_k + alpha v
        self.finished_estimating = False  # have we reached N_k samples for the iterate and the test_point
        self.alpha = self.alpha0  # current value of alpha
        self.rounds = 0  # counting the actual rounds
        self.set_dk = self.compute_dk()

    def compute_dk(self):
        """
        if fixed_Dk, then return the fixed set (shuffle if need be)
        otherwise, return a minimal set, computed thanks to det_set_d_k
        :return:
        """
        if self.fixed_D_k:
            set_dk = self.D_k.copy()
            if self.shuffle:
                random.shuffle(set_dk)
        else:
            set_dk = determine_set_dk(self.dim, self.Ai, self.l, self.u, self.iterate, self.alpha)
        return set_dk

    def compute_N_max(self):
        """formula for the N_k
        """
        return self.sigma ** 2 * 32 * np.log(1 / self.delta) / (self.c * self.alpha ** self.K) ** 2

    def choice(self):

        if self.t == 0:
            self.current_test_point = self.iterate
            self.N_choice = 1
            choice = self.iterate
        else:
            if self.finished_estimating:
                # when the current point has been drawn N_k times
                self.N_choice = 0
                self.sum_current_test_point = 0
                if len(self.set_dk) == 0:
                    self.rounds += 1
                    # when a whole path through the set D_k has been done
                    # then it means that alpha has to be reduced
                    # and a new D_k has to be reset
                    self.alpha = self.theta * self.alpha
                    # print("alpha ", self.alpha, self.t)
                    self.set_dk = self.compute_dk()
                    if self.shuffle:
                        random.shuffle(self.set_dk)
                #  Then, whether it's a new D_k or not, one has to draw dir and test a new choice
                exist_dk = False
                while exist_dk == False:
                    ex_in_constraints = False
                    while ex_in_constraints == False and len(self.set_dk) != 0:
                        d = np.array(self.set_dk.pop())
                        ex_in_constraints = (self.in_constraints(self.iterate + self.alpha * d))
                        if ex_in_constraints:
                            self.d = d
                    exist_dk = ex_in_constraints
                    if not ex_in_constraints and len(self.set_dk) == 0:
                        self.rounds += 1
                        # when a whole path through the set D_k has been done
                        # then it means that alpha has to be reduced
                        # and a new D_k has to be reset
                        self.alpha = self.theta * self.alpha
                        # print("alpha ", self.alpha, self.t)
                        self.set_dk = self.compute_dk()
                    # if self.alpha <0.001:
                    #    raise ValueError

                self.current_test_point = self.iterate + self.alpha * self.d
                choice = self.current_test_point
                self.N_choice = 1
                self.sampling_the_iterate = False
            if not self.finished_estimating:
                # when the current point has not been drawn N_k times
                # one has to draw the last iterate until it reaches N_k
                # and the current choice also
                if self.N_iterate <= self.N_choice:
                    choice = self.iterate
                    self.N_iterate += 1
                    self.sampling_the_iterate = True
                elif self.N_choice < self.N_iterate:
                    choice = self.current_test_point
                    self.N_choice += 1
                    self.sampling_the_iterate = False
        # print("ch ",choice)
        return choice

    def getValue(self, value):
        value = -value
        # update the estimates
        if self.sampling_the_iterate:
            self.sum_iterate += value
        else:
            self.sum_current_test_point += value

        # update finished_estimating
        if self.t == 0:
            self.finished_estimating = True
        else:
            N_lim_choice = self.compute_N_max()
            N_lim_iterate = self.compute_N_max()
            N_k_reached = (self.N_choice >= N_lim_choice and self.N_iterate >= N_lim_iterate)
            diff = self.sum_iterate / self.N_iterate - self.sum_current_test_point / self.N_choice - self.c * self.alpha ** 2
            test_worked = (np.abs(diff) > np.sqrt(2 * self.const * self.sigma ** 2 * (
                    np.log(1 / self.delta) / self.N_iterate + np.log(1 / self.delta) / (self.N_choice))))
            self.finished_estimating = N_k_reached or (test_worked)

        if self.finished_estimating:
            # update the variables in case the estimation phase is finished
            if self.t == 0:
                self.fX = self.sum_current_test_point / self.N_choice
                self.iterate = self.current_test_point
                self.N_iterate = self.N_choice
                self.sum_iterate = self.sum_current_test_point
            else:
                self.fX = self.sum_iterate / self.N_iterate
                new_fX = self.sum_current_test_point / self.N_choice
                self.diff = self.fX - new_fX
                if self.diff > self.c * self.alpha ** 2:
                    self.rounds += 1
                    self.iterate = self.current_test_point
                    self.fX = new_fX
                    self.alpha *= self.gamma
                    self.set_dk = self.compute_dk()
                    if self.shuffle:
                        random.shuffle(self.set_dk)
                    self.N_iterate = self.N_choice
                    self.sum_iterate = self.sum_current_test_point
                self.N_choice = 0
                self.sum_current_test_point = 0

        self.t += 1
