# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:58:36 2021

@author: vragu

Classes related to CRRA utility calculations
"""
from numpy import log as ln
from numpy import exp

class Utility:
    
    def __init__(self, uType ='CRRA',param=1):
        self.uType = uType
        self.param = param
        
    def u_of_wealth(self, wealth):
        ''' Calcualte utility of wealth
        '''
        
        if wealth <= 0:
            #print('Error! Negative wealth, {}, is not allowed.'.format(wealth))
            return(self.u_of_wealth(0.1)+wealth) #
        
        if self.uType != 'CRRA':
            print('Error! Utilities of {}, non-CRRA not implemented'.format(self.uType))
            return None
        else:
            gamma = self.param
            if gamma < 0:
                print('Error!  Negative RA, {}, not allowed'.format(gamma))
            elif gamma == 1:
                u = ln(wealth)
            else:
                u = wealth ** (1-gamma) / (1 - gamma) * 100**gamma
        return u
                
    def implied_wealth(self, u):
        ''' Calculate wealth corresponding to a given utility level
        '''
        if self.uType != 'CRRA':
            print('Errod! Utilities of {}, non-CRRA not implemented'.format(self.uType))
            return None
        else:
            gamma = self.param
            if gamma < 0:
                print('Error!  Negative RA, {}, not allowed'.format(gamma))
            elif gamma == 1:
                wealth = exp(u)
            else:
                wealth = (u / 100**gamma * (1 - gamma))**(1/(1-gamma))
        
        return wealth
    
    def expUtil(self, w_scenarios, probs = None):
        """ Calculate expected utility of expected wealth scenarios
            For now we can assume that all are equally likely, 
            at a later stage, we can implement a vector of probabilities
        """
        
        w = w_scenarios
        n = len(w)
        utils = [self.u_of_wealth(v) for v in w]
        
        return sum(utils)/n
        
# Testing
if __name__ == '__main__':
    
    for param in [0, 1, 2, 5]:
        
        # Test CRRA utility with this risk aversion
        print('\nTesting Risk Aversion = {}'.format(param))
        Util = Utility(param=param)
        wealth = 100
        u = Util.u_of_wealth(wealth)
        w = Util.implied_wealth(u)
        print('Wealth = ', wealth)
        print('Implied wealth = ', w)
        print('Utility =', u)
        
        print('Testing Expected Utility')
        w = [1,2,3]
        print('Scenarios = ', w)
        exUtil = Util.expUtil(w)
        impW = Util.implied_wealth(exUtil)
        print('Exp Util = ', exUtil)
        print('Risk-Free Equiv Wealth = ', impW)