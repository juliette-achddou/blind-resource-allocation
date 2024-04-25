# -*- coding: utf-8 -*-
import random

from .policy import Policy
from .utils.utils_HOO import Partitioner
from .utils.utils_gen import *


# log t courant ?


def G_values(U_values):
    def inf_change(l):
        if l == -np.inf:
            return np.inf
        else:
            return l

    return np.vectorize(inf_change)(U_values)


class HOOConstrainedSimplex(Policy):
    """
    The hierarchical optimistic optimization algorithm.
    """

    def __init__(self, v1, prec, constraints, dim, gamma=1, sigma=0.1, horizon=1000, print_=False):
        """
        :param prec: precision
        :param v1: v1 must be > 0. the diameter of all tree coverings
            must less than v1*ro^h where h
        is the depth of the node with given covering.
        :param ro: ro must be in (0,1) exclusively. the diameter of all
            tree coverings must less than v1*ro^h where h
        is the depth of the node with given covering.
        :param covering_generator_function: this function generates a
            subspace of space iterate,
        by taking the height(h) and the order-in-the-level(i) of a
            binary tree node. (For instance a root node
        has h = 0 and i = 1, its children would have h = 1, and i = 1
            and i = 2 from left to right.) This func. must
        return an object with two properties, namely "upper" & "lower"
            which define the boundaries of the subset.
        """
        # ro needs to be between 0 and 1 exclusively.

        self.in_constr = None
        self.v1 = v1
        self.gamma = gamma
        self.print_ = print_
        self.horizon = horizon
        ro = 2 ** (-2 / dim)
        if 0 >= ro:
            ro = 0.001
        elif ro >= 1:
            ro = 0.999
        self.ro = ro
        self.stopping_h = int(np.log(1 / prec) / np.log(1 / ro)) + 1
        self.step = 1 / ro ** self.stopping_h
        self.N = 2 ** self.stopping_h
        print("N", self.N)
        # print( "stop_h",self.stopping_h)
        self.dim = constraints.dim
        self.is_in_constraints = constraints.in_constraints
        min_values, max_values = np.zeros(self.dim), np.ones(self.dim)
        self.partitioner = Partitioner(min_values, max_values)
        self.tree_coverings = self.partitioner.halve_one_by_one
        self.activated = np.zeros(self.N, dtype=bool)
        self.U_values = float(np.inf) * np.ones(self.N)
        self.B_values = float(np.inf) * np.ones(self.N)
        self.means = np.zeros(self.N)
        self.counts = np.zeros(self.N)
        self.h = np.array([int(np.log2(x)) for x in range(1, self.N)])
        self.i = np.arange(1, self.N) - 2 ** self.h
        self.last_arm = None
        self.last_index = None
        self.traversed_path = []
        self.t = 1
        self.choices = []
        self.sigma = sigma
        self.iterate_bool = False

    def make_name_label(self):
        inter = "HOO" + ", rho = " + str(self.ro)[:min(len(str(self.ro)), 5)] + ", v1 = " + str(self.v1)
        label = string_to_latex("HOO")
        name = string_to_name(inter)
        return name, label

    def set_horizon(self, horizon):
        """
        sets the horizon parameter and dependent parameters (here it is useless)
        :param horizon: horizon of the experiment
        :return: None
        """
        pass

    def get_args(self):
        """
        Returns important arguments of the object as a string (for legends)
        :param self:
        :return: string describing the important arguments
        """
        return " ".join([", rho=", str(self.ro), "v1=", str(self.v1)])

    def create_children_of_node(self, index):
        """ Creates nodes for children
        :param node: parent node.
        """
        self.activated[index] = True
        # for son in (left, right):
        # self.activated[son] = True

    def selection_of_arm_from_covering(self, h, i):
        """ Selects an arm in a specified node
        :param h: chosen node's height.
        :param i: chosen node's order in the level.
        :return: received reward.
        """
        subset_space = self.tree_coverings(h, i)
        selected_arm = subset_space.lower
        return selected_arm

    def recursively_update_tree_U_values(self, round):
        """ Updates the U values at round t
        :param round: round t.
        :return: None
        """
        for index in range(self.N):
            if self.counts[index] != 0:
                self.U_values[index] = self.means[index] + \
                                       np.sqrt(8 * self.sigma ** 2 * math.log(self.t) * self.gamma * 1 / self.counts[
                                           index]) + \
                                       self.v1 * (self.ro ** (int(self.h[index] / self.dim)))

    def startGame(self):
        """
        Sets the arguments to start a new game
        :return: None
        """
        # self.rand_offset = self.step * np.random.random_sample((self.dim))
        min_values, max_values = np.zeros(self.dim), np.ones(self.dim)
        self.partitioner = Partitioner(min_values, max_values)
        self.tree_coverings = self.partitioner.halve_one_by_one
        self.tree_coverings = self.partitioner.halve_one_by_one
        self.N = 2 ** self.stopping_h
        self.activated = np.zeros(self.N, dtype=bool)
        self.U_values = float("inf") * np.ones(self.N)
        self.B_values = float("inf") * np.ones(self.N)
        self.means = np.zeros(self.N)
        self.counts = np.zeros(self.N)
        self.h = np.array([int(np.log2(x)) for x in range(1, self.N + 1)])
        self.i = np.arange(1, self.N + 1) - 2 ** self.h
        # create first two leaves with infinite B values
        # self.create_children_of_node(0)
        self.activated[0] = False
        self.last_arm = None
        self.last_index = None
        self.traversed_path = []
        self.t = 1
        self.choices = []
        self.firstpath()

    def firstpath(self):
        self.in_constr = np.ones(self.N)
        for index in range(self.N):
            h = self.h[index]
            i = self.i[index]
            c = self.selection_of_arm_from_covering(h, i + 1)
            self.in_constr[index] = self.is_in_constraints(c)
            if not self.in_constr[index]:
                self.B_values[index] = - np.inf
                self.U_values[index] = - np.inf

        for h in range(self.stopping_h - 2, -1, -1):
            P = self.in_constr[2 ** h - 1:2 ** (h + 1) - 1]
            left_S = np.array([self.in_constr[2 * k - 1] for k in range(2 ** h, 2 ** (h + 1))])
            right_S = np.array([self.in_constr[2 * k] for k in range(2 ** h, 2 ** (h + 1))])
            max_S = np.maximum(left_S, right_S)
            self.in_constr[2 ** h - 1:2 ** (h + 1) - 1] = np.minimum(P, max_S)

            B = self.B_values[2 ** h - 1:2 ** (h + 1) - 1]
            left_B = np.array([self.B_values[2 * k - 1] for k in range(2 ** h, 2 ** (h + 1))])
            right_B = np.array([self.B_values[2 * k] for k in range(2 ** h, 2 ** (h + 1))])
            max_B = np.maximum(left_B, right_B)
            self.B_values[2 ** h - 1:2 ** (h + 1) - 1] = np.minimum(B, max_B)
            self.U_values[2 ** h - 1:2 ** (h + 1) - 1] = np.minimum(B, max_B)

    def choice(self):
        """
         Chooses the appropriate arm
        :return: the chosen point
        """
        current_index = 0
        self.traversed_path = [current_index]
        # print(self.B_values)
        # print(self.U_values)

        # traverse down the tree to find the leaf with highest B value.
        while self.activated[current_index] and self.h[current_index] < self.stopping_h - 1:
            h = self.h[current_index]
            i = self.i[current_index]
            left = 2 ** (h + 1) + 2 * i - 1
            right = 2 ** (h + 1) + 2 * i

            if self.B_values[left] > self.B_values[right]:
                current_index = left

            elif self.B_values[left] < self.B_values[right]:
                current_index = right

            else:
                # tie breaking rule here is defined as choosing a
                # child at random
                rand = random.uniform(0, 1)

                if rand > 0.5:
                    current_index = left

                else:
                    current_index = right

            # append to traversed path and update height as well
            # as order in level
            self.traversed_path.append(current_index)

            # now we have selected the most promising child from the
            # tree we had,
            # we can draw an arm in iterate from the coverings
        current_choice = self.selection_of_arm_from_covering(
            self.h[current_index], self.i[current_index] + 1)
        self.choices.append(current_choice)
        self.last_index = current_index
        self.t += 1
        if self.print_ and self.t == 100:
            pass
            # print("choices", self.choices)

        return np.array(current_choice)

    def getValue(self, value):
        """
        Stores necessary information for next rounds
        :param value:
        :return: None
        """

        current_choice = self.choices[-1]
        y = value
        self.create_children_of_node(self.last_index)
        for index in self.traversed_path:
            self.counts[index] += 1
            self.means[index] = (1.0 - 1.0 / self.counts[index]) * \
                                self.means[index] + y * 1.0 / self.counts[index]

        self.recursively_update_tree_U_values(round=self.t)

        # backward computation to update all the B_values.
        for h in range(self.stopping_h - 2, -1, -1):
            U = self.U_values[2 ** h - 1:2 ** (h + 1) - 1]
            left_B = np.array([self.B_values[2 * k - 1] for k in range(2 ** h, 2 ** (h + 1))])
            right_B = np.array([self.B_values[2 * k] for k in range(2 ** h, 2 ** (h + 1))])
            max_B = np.maximum(left_B, right_B)
            self.B_values[2 ** h - 1:2 ** (h + 1) - 1] = np.minimum(U, max_B)

    def recommandArm(self):
        """ Recommends an arm at the end of the game
        :param self:
        :return: recommended arm at the end of the game
        """
        r = random.choice(self.choices)
        return r
