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