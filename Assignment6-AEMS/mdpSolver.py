"""
UW, CSEP 573, Win19
"""
from pomdp import POMDP
from offlineSolver import OfflineSolver

import numpy as np
import math

class QMDP(OfflineSolver):
    def __init__(self, pomdp, precision = .001):
        super(QMDP, self).__init__(pomdp, precision)
        """
        ****Your code
        Remember this is an offline solver, so compute the policy here
        """

        self.Q_value = np.zeros([len(self.pomdp.actions), len(self.pomdp.states)])
        self.values = np.zeros([len(self.pomdp.states)])
        self.rewards = np.zeros([len(self.pomdp.actions), len(self.pomdp.states)])
        for s_index, state in enumerate(self.pomdp.states):
            for a_index, action in enumerate(self.pomdp.actions):
                self.rewards[a_index, s_index] = np.max(np.dot(self.pomdp.T[a_index, s_index, :], self.pomdp.R[a_index, s_index, :, :]))

        time_step = 0
        max_abs_reward = np.max(np.abs(self.pomdp.R))
        while (max_abs_reward * (self.pomdp.discount ** time_step) > self.precision):
            self.updateQValueFromValues()
            time_step += 1

    def chooseAction(self, cur_belief):
        """
        ***Your code
        """
        # return 0 #remove this after your implementation
        return np.argmax(np.matmul(self.Q_value, cur_belief.T))

    def getValue(self, belief):
        """
        ***Your code
        """
        # return 0 #remove this after your implementation
        value = np.max(np.matmul(self.Q_value, belief.T))
        return math.floor(value * 100) / 100.0

    """
    ***Your code
    Add any function, data structure, etc that you want
    """

    def updateQValueFromValues(self):
        for s_index, state in enumerate(self.pomdp.states):
            for a_index, action in enumerate(self.pomdp.actions):
                self.Q_value[a_index, s_index] = self.rewards[a_index, s_index]  + self.pomdp.discount * np.dot(self.pomdp.T[a_index, s_index, :], self.values)
            self.values[s_index] = np.max(self.Q_value[:, s_index])

class MinMDP(OfflineSolver):
    def __init__(self, pomdp, precision = .001):
        super(MinMDP, self).__init__(pomdp, precision)
        """
        ***Your code
        Remember this is an offline solver, so compute the policy here
        """
        self.Q_value = np.zeros([len(self.pomdp.actions), len(self.pomdp.states)])
        self.values = np.zeros([len(self.pomdp.states)])
        self.minRewards = np.min(self.pomdp.R)
        self.minRewardsDiscount = self.pomdp.discount * (1 - 0) / (1 - self.pomdp.discount)

        self.rewards = np.zeros([len(self.pomdp.actions), len(self.pomdp.states)])
        for s_index, state in enumerate(self.pomdp.states):
            for a_index, action in enumerate(self.pomdp.actions):
                self.rewards[a_index, s_index] = np.max(np.dot(self.pomdp.T[a_index, s_index, :], self.pomdp.R[a_index, s_index, :, :]))

        time_step = 0
        max_abs_reward = np.max(np.abs(self.pomdp.R))
        while (max_abs_reward * (self.pomdp.discount ** time_step) > self.precision):
            self.updateQValueFromValues()
            time_step += 1

    def getValue(self, cur_belief):
        """
        ***Your code
        """
        # return 0 #remove this after your implementation
        value = np.max(np.matmul(self.Q_value, cur_belief.T)) + self.minRewards * self.minRewardsDiscount
        return math.floor(value * 100) / 100.0

    def chooseAction(self, cur_belief):
        """
        ***Your code
        """
        # return 0 #remove this after your implementation
        return np.argmax(np.matmul(self.Q_value, cur_belief.T))

    """
    ***Your code
    Add any function, data structure, etc that you want
    """

    def updateQValueFromValues(self, ):
        for s_index, state in enumerate(self.pomdp.states):
            for a_index, action in enumerate(self.pomdp.actions):
                self.Q_value[a_index, s_index] = self.rewards[a_index, s_index]
            self.values[s_index] = np.max(self.Q_value[:, s_index])
