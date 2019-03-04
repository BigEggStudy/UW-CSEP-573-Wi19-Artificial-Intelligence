"""
UW, CSEP 573, Win19
"""
from pomdp import POMDP
from offlineSolver import OfflineSolver


class QMDP(OfflineSolver):
    def __init__(self, pomdp, precision = .001):
        super(QMDP, self).__init__(pomdp, precision)
        """
        ****Your code
        Remember this is an offline solver, so compute the policy here
        """     
    
    def chooseAction(self, cur_belief):
        """
        ***Your code
        """  
        return 0 #remove this after your implementation
            
    
    def getValue(self, belief):
        """
        ***Your code
        """
        return 0 #remove this after your implementation
    

    """
    ***Your code
    Add any function, data structure, etc that you want
    """
    

class MinMDP(OfflineSolver):
    
    def __init__(self, pomdp, precision = .001):
        super(MinMDP, self).__init__(pomdp, precision)
        """
        ***Your code 
        Remember this is an offline solver, so compute the policy here
        """
    
    def getValue(self, cur_belief):
        """
        ***Your code
        """
        return 0 #remove this after your implementation

    def chooseAction(self, cur_belief):
        """
        ***Your code
        """  
        return 0 #remove this after your implementation


    """
    ***Your code
    Add any function, data structure, etc that you want
    """