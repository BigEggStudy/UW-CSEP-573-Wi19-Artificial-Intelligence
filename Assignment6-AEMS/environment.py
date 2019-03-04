"""
UW, CSEP 573, Win19
"""

import numpy as np
import random
from pomdp import POMDP

class Environment:
    def __init__(self, pomdp):
        self.pomdp = pomdp
        r = random.uniform(0, 1)
        s = 0
        for i in range(len(pomdp.states)):
            s += pomdp.prior[i]
            if s > r:
                self.cur_state = i
                break
            
    def act(self, action):
        """
        Perofrm the action
        return reward and observation
        reward = None means terminal state
        """
        #action
        r = random.uniform(0, 1)
        s = 0
        for i in range(len(self.pomdp.states)):
            s += self.pomdp.T[action, self.cur_state, i]
            if s > r:
                next_state = i
                break
        #observtion
        r = random.uniform(0, 1)
        s = 0
        for i in range(len(self.pomdp.observations)):
            s += self.pomdp.O[action, next_state, i]
            if s > r:
                observation = i
                break
        # reward 
        reward = self.pomdp.R[action, self.cur_state, next_state, observation]
        if reward == 0 and np.where(self.pomdp.T[:, next_state, next_state] < 1)[0].size == 0:
            reward = None
        self.cur_state = next_state 
        return reward, observation 