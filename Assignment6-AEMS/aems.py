#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UW, CSEP 573, Win19
"""

from pomdp import POMDP
from onlineSolver import OnlineSolver
import numpy as np

class AEMS2(OnlineSolver):
    def __init__(self, pomdp, lb_solver, ub_solver, precision = .001, action_selection_time = .1):
        super(AEMS2, self).__init__(pomdp, precision, action_selection_time)
        self.lb_solver = lb_solver
        self.ub_solver = ub_solver
        """
        *****Your code
        You can add any attribute you want
        """
        self.rewards = np.zeros([len(self.pomdp.actions), len(self.pomdp.states)])
        for s_index, state in enumerate(self.pomdp.states):
            for a_index, action in enumerate(self.pomdp.actions):
                for o_index, observation in enumerate(self.pomdp.observations):
                    self.rewards[a_index, s_index] = np.sum(np.dot(self.pomdp.T[a_index, s_index, :] * self.pomdp.O[a_index, :, o_index], self.pomdp.R[a_index, s_index, :, o_index]))

        cur_belief = np.array(self.pomdp.prior).reshape(1, len(self.pomdp.prior))
        self.root = OrNode(cur_belief, self.lb_solver, self.ub_solver, 1, [], None)

    def expandOneNode(self):
        """
        *****Your code
        """
        # return False #remove this after your implementation
        leaves = self.__getAllLeaves(self.root)
        highestErrorLeaf = max([(leaf, self.__computeError(leaf, depth)) for (leaf, depth) in leaves], key = lambda n: n[1])[0]

        andNodes = []
        for a_index, action in enumerate(self.pomdp.actions):
            probabilities = (highestErrorLeaf.belief @ self.pomdp.T[a_index, :, :] @ self.pomdp.O[a_index, :, :])[0]

            andNode = AndNode(a_index, self.pomdp.discount, sum(highestErrorLeaf.belief @ self.rewards[a_index]), [], probabilities, highestErrorLeaf)
            andNode.children = [OrNode(self.__updateBelief(highestErrorLeaf.belief, a_index, o_index), self.lb_solver, self.ub_solver, probabilities[o_index], [], andNode) for o_index, observation in enumerate(self.pomdp.observations)]
            andNode.backtrack()
            andNodes.append(andNode)

        highestErrorLeaf.children = andNodes
        highestErrorLeaf.backtrack()
        parent_node = highestErrorLeaf.parent
        while parent_node is not None:
            parent_node.backtrack()
            parent_node = parent_node.parent

    def chooseAction(self):
        """
        *****Your code
        """
        # return 0 #remove this after your implementation
        if len(self.root.children) == 0:
            self.expandOneNode()
        return max([(action_index, andNode.upperBound) for action_index, andNode in enumerate(self.root.children)], key = lambda n: n[1])[0]

    def updateRoot(self, action, observation):
        """
        ***Your code
        """
        self.root = self.root.children[action]
        self.root = self.root.children[observation]

    def __computeError(self, node, depth):
        error = node.getError()
        if depth == 0:
            return error

        error = (self.pomdp.discount ** depth) * error

        orNode = node
        for _ in range(depth):
            andNode = orNode.parent
            parentOrNode = andNode.parent
            bestAction = max([(child.action, child.upperBound) for child in parentOrNode.children], key = lambda n: n[1])[0]
            if andNode.action != bestAction:
                return 0
            error = error * orNode.probability
            orNode = parentOrNode
        return error

    def __updateBelief(self, current_belief, action, observation):
        current_belief = np.matmul(current_belief, self.pomdp.T[action, :, :])
        current_belief = current_belief * self.pomdp.O[action, :, observation]
        return current_belief / np.sum(current_belief) if np.sum(current_belief) > 0 else current_belief

    def __getAllLeaves(self, orNode, depth = 0):
        if len(orNode.children) == 0:
            return [(orNode, depth)]

        orNodes = []
        for andNode in orNode.children:
            orNodes += andNode.children

        leaves = []
        for subOrNode in orNodes:
            leaves += self.__getAllLeaves(subOrNode, depth + 1)

        return leaves

"""
****Your code
add any data structure, code, etc you want
We recommend to have a super class of Node and two subclasses of AndNode and OrNode
"""

class Node(object):
    def __init__(self, children = [], parent = None):
        self.lowerBound = float('-inf')
        self.upperBound = float('inf')
        self.children = children
        self.parent = parent

    def getError(self):
        return self.upperBound - self.lowerBound

class AndNode(Node):
    def __init__(self, action, discount, reward, children = [], probabilities = [], parent = None):
        super(AndNode, self).__init__(children, parent)
        self.action = action
        self.discount = discount
        self.reward = reward

        if sum(probabilities) < 1:
            probabilities = probabilities / np.sum(probabilities)
        self.probabilities = probabilities

    def backtrack(self):
        self.lowerBound = self.reward + self.discount * sum([self.probabilities[index] * child.lowerBound for index, child in enumerate(self.children)])
        self.upperBound = self.reward + self.discount * sum([self.probabilities[index] * child.upperBound for index, child in enumerate(self.children)])

class OrNode(Node):
    def __init__(self, belief, lb_solver, ub_solver, probability, children = [], parent = None):
        super(OrNode, self).__init__(children, parent)
        self.belief = belief
        self.lowerBound = lb_solver.getValue(self.belief)
        self.upperBound = ub_solver.getValue(self.belief)
        self.probability = probability

    def backtrack(self):
        self.lowerBound = max([child.lowerBound for child in self.children])
        self.upperBound = max([child.upperBound for child in self.children])
