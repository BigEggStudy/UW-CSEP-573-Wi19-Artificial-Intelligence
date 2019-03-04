#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UW, CSEP 573, Win19
"""

from pomdp import POMDP
from environment import Environment
import numpy as np


class OfflineSolver:
    def __init__(self, pomdp, precision = .001):
        self.pomdp = pomdp
        self.precision = precision
        
    def solve(self): 
        """ 
        solve and calulcate the total reward 
        for one run 
        """
        total_reward = 0
        environment = Environment(self.pomdp)
        time_step = 0
        Max_abs_reward = np.max(np.abs(self.pomdp.R))
        cur_belief = np.array(self.pomdp.prior).reshape(1, len(self.pomdp.prior))
        while (Max_abs_reward * (self.pomdp.discount ** time_step) > self.precision):
        # each iteration 
            action = self.chooseAction(cur_belief)
            reward, obs = environment.act(action)
            if reward == None:  # we check Terminal states to get results faster 
                break
            total_reward += reward * (self.pomdp.discount ** time_step)  
            cur_belief = self.updateBelief(cur_belief, action, obs)
            time_step +=1
        return total_reward   
    
    def evaluate(self, num_runs = 100):
        sum_reward = 0
        for j in range(num_runs):    
            sum_reward += self.solve()
        return  sum_reward/num_runs
        

    def chooseAction(self, cur_belief):
        """
        Choose action (The best action based on the given belief)
        """
        raise NotImplementedError("Subclass must implement abstract method")     
    
    def updateBelief(self, current_belief, action, observation):
        current_belief = np.matmul(current_belief , self.pomdp.T[action, :, :])
        current_belief = current_belief * self.pomdp.O[action, :, observation]
        return current_belief / np.sum(current_belief)
    
    def getValue(self, cur_belief):
        """
        Return the estimated value function of the belief given as an input
        """
        raise NotImplementedError("Subclass must implement abstract method")    
        