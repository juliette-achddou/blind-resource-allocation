# -*- coding: utf-8 -*-
import math
import random

import numpy as np

from .policy import Policy
from .utils.utils_HOO import Partitioner


class HOOStop(Policy):
    """
    The hierarchical optimistic optimization algorithm, but with a fixed depth to avoid too high computational costs
    """

    def __init__(self, v1, prec, ro=1 / 2, gamma=1, sigma=0.1, print_=False):
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
        if 0 >= ro:
            ro = 0.001
        elif ro >= 1:
            ro = 0.999

        self.ro = ro
        self.v1 = v1
        self.gamma = gamma
        self.print_ = print_
        self.stopping_h = int(np.log2(1 / prec)) + 1
        # print(self.stopping_h)
        min_values, max_values = np.zeros(1), np.ones(1)
        self.partitioner = Partitioner(min_values, max_values)
        self.tree_coverings = self.partitioner.halve_one_by_one
        self.N = 2 ** self.stopping_h
        # print(self.N)
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
        selected_arm = (1.0 * subset_space.upper + subset_space.lower) / 2.0
        return selected_arm

    def recursively_update_tree_U_values(self, round):
        """ Updates the U values at round t
        :param round: round t.
        :return: None
        """
        self.U_values = self.means + \
                        np.sqrt(2.0 * self.sigma ** 2 * math.log(round) * self.gamma * 1 / self.counts) + \
                        self.v1 * (self.ro ** self.h)

    def startGame(self):
        """
        Sets the arguments to start a new game
        :return: None
        """
        min_values, max_values = (np.zeros(1), np.ones(1))
        self.partitioner = Partitioner(min_values, max_values)
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

    def choice(self):
        """
         Chooses the appropriate arm
        :return: the chosen point
        """
        current_index = 0
        self.traversed_path = [current_index]

        # traverse down the tree to find the leaf with highest B value.
        while self.activated[current_index] and self.h[current_index] < self.stopping_h - 1:
            h = self.h[current_index]
            i = self.i[current_index]
            left = 2 ** (h + 1) + 2 * i - 1
            right = 2 ** (h + 1) + 2 * i

            if self.B_values[left] > self.B_values[right]:
                current_index = left

            elif self.B_values[left] > self.B_values[right]:
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
        # print(self.last_index)
        # print(current_choice)
        self.t += 1
        if self.print_ and self.t == 100:
            print(self.choices)

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
            # print(2**h-1,2**(h+1))
            U = self.U_values[2 ** h - 1:2 ** (h + 1) - 1]
            # print(U)
            left_B = np.array([self.B_values[2 * k - 1] for k in range(2 ** h, 2 ** (h + 1))])
            right_B = np.array([self.B_values[2 * k] for k in range(2 ** h, 2 ** (h + 1))])
            max_B = np.maximum(left_B, right_B)
            # print(max_B)
            self.B_values[2 ** h - 1:2 ** (h + 1) - 1] = np.minimum(U, max_B)

    def recommandArm(self):
        """ Recommends an arm at the end of the game
        :param self:
        :return: recommended arm at the end of the game
        """
        r = random.choice(self.choices)
        return r
