# utils for HOO

import numpy as np

"""
    This class is designed with the intent
    that it can be used as the default tree_coverings_generator for
    the HOO algorithm.
"""


class Covering(object):

    def __init__(self, lowers, uppers):
        """
        :param lowers: lower bounds of the subset(covering) of the
            generic measurable space-iterate.
        :param uppers: upper bounds of the subset(covering) of the
            generic measurable space-iterate.
        """
        self.upper = uppers
        self.lower = lowers


class Partitioner(object):

    def __init__(self, min_values, max_values, **keyword_parameters):
        """
        :param min_values: the minimum values of the domain of the environment
            (reward) function.
        :param max_values: the maximum values of the domain of the environment
            (reward) function.
        :param keyword_parameters: 'partitioning_priorities' is the only
            available optional argument,
        and when it is not provided no dimension has a priority over the
            order and at each level a different
        dimension is partitioned, but if there were a total of 3 dimensions
            in the iterate-space and the 'partitioning policy'
        was set to [2,0,1] then the partitioner partitions the zeroth dimension
            twice, leaves first dimension as it is,
        then partitions second dimension once, and repeats.
        so the default behaviour is as if the priorities is set to [1,1,...,1].
        """
        self.covering_sequence = {}
        search_space = Covering(lowers=np.array(
            min_values, dtype=float), uppers=np.array(max_values, dtype=float))
        self.dimensions = np.size(min_values)
        # the root is the entire space iterate.
        self.covering_sequence[0, 1] = search_space

        # NOTE that this is an experimental input that is
        # not in the original paper but meant as an improvement.
        if 'partitioning_priorities' in keyword_parameters:
            self.priority_array = keyword_parameters['partitioning_priorities']
        else:
            self.priority_array = np.ones(self.dimensions)
        if 'nb_sons' in keyword_parameters:
            self.nb_sons = keyword_parameters['nb_sons']
        else:
            self.nb_sons = 2

    def halve_one_by_one(self, height, place_in_level):
        """
        :param height: height of the node.
        :param place_in_level: order in the level of the node.
        :return: a subset of iterate.
        """
        # print(self.covering_sequence.keys())
        if (height, place_in_level) in self.covering_sequence.keys():
            return self.covering_sequence[height, place_in_level]
        else:
            # print("halve", height, place_in_level)
            # we need to find out which dimension we need to partition.
            dimension_to_be_partitioned = 0
            # if priorities are not explicitly defined this just
            # equals self.dimensions.
            priority_height = np.sum(self.priority_array)
            # example: if we are at height = 7 and have 3 dimensions with equal
            # priority in iterate, we should partition
            # the dimension (7-1) % 3 = 0.
            effective_height = height % priority_height
            cumulative_priorities = np.cumsum(self.priority_array)

            while effective_height >= cumulative_priorities[
                dimension_to_be_partitioned]:
                dimension_to_be_partitioned += 1

            parent_h = height - 1
            parent_i = (place_in_level + 1) // self.nb_sons
            # print(parent_h, parent_i)
            # print("parent", parent_h, parent_i)
            parent_space = self.covering_sequence[parent_h, parent_i]
            # print("index = {0},{1} and parent = {2},{3}".format(
            # height,place_in_level,parent_h,parent_i))
            # partition the dimension_to_be_partitioned into half.
            newSpace = Covering(lowers=np.copy(parent_space.lower),
                                uppers=np.copy(parent_space.upper))
            new_size = (parent_space.upper[dimension_to_be_partitioned] - parent_space.lower[
                dimension_to_be_partitioned]) / self.nb_sons

            # # left child of parent, left portion of the parent's subspace.
            # if place_in_level % 2 == 1:
            #     newSpace.upper[dimension_to_be_partitioned] = midValue
            # # right child of parent, right portion of the parent's subspace.
            # else:
            #     newSpace.lower[dimension_to_be_partitioned] = midValue

            k = (place_in_level - 1) % self.nb_sons
            # print("k", k)
            newSpace.lower[dimension_to_be_partitioned] = parent_space.lower[
                                                              dimension_to_be_partitioned] + (k) * new_size
            newSpace.upper[dimension_to_be_partitioned] = parent_space.lower[
                                                              dimension_to_be_partitioned] + (k + 1) * new_size
            # print(newSpace.lower, newSpace.upper)

            # print("partitioning dimension = {0} and midvalue =
            # {1}".format(dimension_to_be_partitioned,midValue))
            # print("parent was \nlower={0}\nupper={1}".format(parent_space.lower,
            # parent_space.upper))
            # print("added element with\nlower={0}\nupper={1}".format(
            # newSpace.lower, newSpace.upper))

            self.covering_sequence[height, place_in_level] = newSpace

            return newSpace


class TreeNodeHOO(object):

    def __init__(self):
        """"
        This is a binary tree node
        """

        # left&right pointers
        self.left = None
        self.right = None

        # values required by HOO and the covering generator algorithm
        self.activated = False
        self.counter = 0
        self.B_value = None
        self.mean = 0
        self.U_value = None
        self.h = 0
        self.i = 1
