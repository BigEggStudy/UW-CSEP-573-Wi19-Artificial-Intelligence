#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UW, CSEP 573, Win19
"""

from pomdp import POMDP
from environment import Environment
import numpy as np
import time


class OnlineSolver:    
    def __init__(self, pomdp, precision = .001, action_selection_time = .1):
        self.pomdp = pomdp
        self.precision = precision
        self.time_limit = action_selection_time
    
    @staticmethod
    def solve(solver): 
        """ 
        solve and calulcate the total reward 
        for one run in an online solver
        """
        total_reward = 0
        environment = Environment(solver.pomdp)
        time_step = 0
        Max_abs_reward = np.max(np.abs(solver.pomdp.R))
        while (Max_abs_reward * (solver.pomdp.discount ** time_step) > solver.precision):
        # each iteration 
            start = time.time()
            while (time.time() - start < solver.time_limit):
                is_expanded = solver.expandOneNode()
                if is_expanded == False:
                    break
            action = solver.chooseAction()
            reward, observation = environment.act(action) 
            if reward == None:    # we check Terminal states to get results faster 
                break
            total_reward += reward * (solver.pomdp.discount ** time_step)
            time_step += 1
            solver.updateRoot(action, observation)
        return total_reward
    
    def expandOneNode(self):
        """
        Expand one more leaf if possible
        return Boolean:
        if one expanded return True
        if there is no node left to update (for all nodes: |V*(b) - h(b)| < precision and |V*(b) - L*(b)| < precision) return False
        """
        raise NotImplementedError("Subclass must implement abstract method")        
    
    def chooseAction(self):
        """
        Choose action (The best action based on the root)
        return action index
        """
        raise NotImplementedError("Subclass must implement abstract method") 
    
    def updateRoot(self, action, observation):
        """
        Update the root of the AND-OR tree based on performed action and observed observation
        return None
        """
        raise NotImplementedError("Subclass must implement abstract method")  
