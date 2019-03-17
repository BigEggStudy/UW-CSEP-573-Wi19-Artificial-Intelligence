#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UW, CSEP 573, Win19
"""

from pomdp import POMDP
from onlineSolver import OnlineSolver


class AEMS2(OnlineSolver):
    def __init__(self, pomdp, lb_solver, ub_solver, precision = .001, action_selection_time = .1):
        super(AEMS2, self).__init__(pomdp, precision, action_selection_time)
        self.lb_solver = lb_solver
        self.ub_solver = ub_solver
        """
        *****Your code
        You can add any attribute you want
        """

    def expandOneNode(self):
        """
        *****Your code
        """
        return False #remove this after your implementation


    def chooseAction(self):
        """
        *****Your code
        """
        return 0 #remove this after your implementation



    def updateRoot(self, action, observation):
        """
        ***Your code
        """
        return None




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
