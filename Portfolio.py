# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 16:46:30 2021

@author: vragu
Portfolio Classes for equities and options
"""
import Utility as ut
import SkewDistributions as sd
from scipy.optimize import minimize
import numpy as np
from scipy import stats
import csv

#import sys

class Portfolio:
    
    def __init__(self, cash = 100, stock_pos = 0, opt_notional = 0, \
               opt_strike = 80, opt_type = 'put', opt_mkt_px = 0):
        self.cash = cash
        self.stock_pos = stock_pos
        self.opt_notional = opt_notional
        self.opt_strike = opt_strike
        self.opt_type = opt_type
        self.opt_mkt_px = opt_mkt_px
        
    def buyStocks(self, tradeVal):
        self.cash -= tradeVal
        self.stock_pos += tradeVal

    def buyOpts(self, tradeNotional, optPrice):
        self.cash -= tradeNotional * optPrice/100
        self.opt_notional += tradeNotional
        
    def portFinalVal(self, stockPrice):
        """ Calc portfolio value as function of stock price.
            Assume that starting price of the stock was 100
        """
        val = self.cash + self.stock_pos * stockPrice / 100 \
            + self.opt_notional * self.optValAtExpiry(stockPrice) / 100
        return val
    
    def optValAtExpiry(self, stockPrice):
        """ Calculate final option value as a function of the stock price"""
        if self.opt_type != 'put':
            print("Error -only put options have been implemented.")
            return None
        else:
            val = max(self.opt_strike - stockPrice,0)
            return val
        
    def portScenarios(self, stockScenarios):
        """Return portfolio value in a range of scenarios
        """

        vals = [self.portFinalVal(s) for s in stockScenarios]
        return vals
    
    def portExpVal(self, stockScenarios):
        ev = np.mean(self.portScenarios(stockScenarios))
        return ev
    
    def portExpUtil(self, stockScenarios, util_func):
        """  Returns 3 values: expected utility, exp_value, implied risk-free value
        """
        port_vals = self.portScenarios(stockScenarios)

        exp_util = util_func.expUtil(port_vals)
        exp_value = np.mean(port_vals)
        imp_rf_value = util_func.implied_wealth(exp_util)
        
        #Check that we return scalars and not arrays:
        if isinstance(exp_util, (list, tuple, np.ndarray)):
            exp_util = exp_util[0]
        
        if isinstance(exp_value, (list, tuple, np.ndarray)):
            exp_util = exp_util[0]
        
        if isinstance(imp_rf_value, (list, tuple, np.ndarray)):
            exp_util = exp_util[0]
        
        return exp_util, exp_value, imp_rf_value
    
    def optionMktPrice(self, stockScenarios_risk_neutral, prem2ev = 0):
        """ Calculate market value of options at the start of the period
            Assume market price = (1+premToExpValue) * ExpValue
        """
        final_vals = [self.optValAtExpiry(p) for p in stockScenarios_risk_neutral]
   
        optMktPrice = np.mean(final_vals) * (1+prem2ev)
        
        return optMktPrice
          
    
if __name__ == '__main__':
    
    #Specify stock return parameters"
    ret_mean = 0.0272
    ret_std  = 0.16
    #ret_skews = [ 0, -0.25, -0.5, -1.25, -2.0 ]
    ret_skews = [ -1.3 ]
    risk_aversion_list = [0.5, 1, 2, 3, 5, 10]
    prem2ev_list = [ 0 ] # <- match BS price, 0.2, 0.5, 1.0, 2.0]
    opt_strike = 90
    use_options_list = ['Stock', 'Opts'] # ['Stock', 'Opts', 'Opts+K']
        
    for use_options in use_options_list:
        
        stockPosList = []
        optNotionalList = []
        expValList = []
        impRfValList = []
        optStrikeList = []
        
        for risk_aversion in risk_aversion_list:

            util_func = ut.Utility(uType='CRRA', param=risk_aversion)

            for ret_skew in ret_skews:
                for prem2ev in prem2ev_list:
                    
                    print('\n\nRunning ra = {:.1f}, skew = {:.2f}, prem={:.2f}, use_options={}'\
                          .format(risk_aversion, ret_skew, prem2ev, use_options))
            
                    #Generate stock returns
                    np.random.seed(1)
                    stockScenarios = np.exp(sd.createSkewDist(mean=ret_mean, sd=ret_std, \
                                                    skew=ret_skew, size=10000))*100
                    # Added a heuristic multiplier for the risk-neutral dist to match BS price
                    stockScenarios_risk_neutral = stockScenarios - 127 * (1 - np.exp(-ret_mean))
                        
                    print('Mean of stock scenarios = {:.2f}'.format(np.mean(stockScenarios)))
                    print('Std of stock scenarios  = {:.2f}'.format(np.std(stockScenarios)))
                    print('Skew of stock returns   = {:.2f}'.format(stats.skew(np.log(stockScenarios))))
                    print('Mean of rn stock scenarios = {:.2f}'.format(np.mean(stockScenarios_risk_neutral)))
                    print('Std of rn stock scenarios  = {:.2f}'.format(np.std(stockScenarios_risk_neutral)))
                    print('Skew of rn stock returns   = {:.2f}'.format(stats.skew(np.log(stockScenarios_risk_neutral))))
                    print('\n')
                    
                    
                    #Create portfolio
                    port = Portfolio(opt_strike = opt_strike)
                    opt_mkt_px = port.optionMktPrice(stockScenarios_risk_neutral, \
                                                     prem2ev = prem2ev)
                    port.opt_mkt_px = opt_mkt_px
                    
                    #Define objective functions
                    def objFun(x):
                        
                        stockPos = x[0]
                        
                        #Objective function for optimization
                        tryPort = Portfolio()
                        tryPort.buyStocks(stockPos)
                        
                        #Calculate expected utility of the trial portfolio
                        util, expVal, impRfVal = tryPort.portExpUtil(stockScenarios, util_func)
                        #print(util, expVal, impRfVal)
                        
                        return -util * 100000
                    
                    def objFun_Opts(x):
                        
                        stockPos = x[0]
                        optNotional = x[1]
                        
                        #Objective function for optimization
                        tryPort = Portfolio(opt_strike = opt_strike)
                        tryPort.buyStocks(stockPos)
                        tryPort.buyOpts(optNotional,opt_mkt_px)
                        tryPort.opt_mkt_px = opt_mkt_px
                        
                        #Calculate expected utility of the trial portfolio
                        util, expVal, impRfVal = tryPort.portExpUtil(stockScenarios, util_func)
                        #print(util, expVal, impRfVal)
                        
                        #Not allowed to sell options:
                        # if optNotional < 0:
                        #     util -= optNotional**2
                            
                        return -util * 100000
                
                    def objFun_Opts_K(x):
                        
                        stockPos = x[0]
                        optNotional = x[1]
                        optStrike = x[2]
                        
                        #Objective function for optimization
                        tryPort = Portfolio(opt_strike = optStrike)
                        tryPort.buyStocks(stockPos)
                        tryPort.opt_mkt_px= tryPort.optionMktPrice(stockScenarios_risk_neutral, 
                                                                   prem2ev = prem2ev)
                        tryPort.buyOpts(optNotional,tryPort.opt_mkt_px)
                        
                        #Calculate expected utility of the trial portfolio
                        util, expVal, impRfVal = tryPort.portExpUtil(stockScenarios, util_func)
                        
                        #Not allowed to sell options:
                        # if optNotional < 0:
                        #     util -= optNotional**2
                            
                        return -util * 100000
                
                    #Test objFun_Opts function
                    # test1 = True
                    # if test1:
                    #     print(objFun_Opts([0,20]))
                    #     sys.exit()
                        
                    # Run optimizer
                    
                    if use_options == 'Opts':
                        f = objFun_Opts
                        guess = np.array([ret_mean/(ret_std**2),0.])
                
                        #Constraint - maximum option notional          
                        max_opt_to_stock = 4
                        cons = ({'type': 'ineq', 'fun': lambda x:  x[0] * max_opt_to_stock -  x[1] })
                                #{'type': 'ineq', 'fun': lambda x:  100 - x[0]}, 
                                #{'type': 'ineq', 'fun': lambda x:  100 - x[1]} ) 
                        
                        #Optimization
                        #res = minimize(f, guess, method='SLSQP', constraints=cons)    
                        res = minimize(f, guess, method='COBYLA', constraints=cons, tol = 1.e-9) 
                        
                        stockPos_opt = res.x[0]
                        optNotional_opt = res.x[1]
                        
                    elif use_options == 'Opts+K':
                        f = objFun_Opts_K
                        guess = np.array([100.,100., 80.0])
                
                        #Constraint - maximum option notional          
                        max_opt_to_stock = 4
                        cons = ({'type': 'ineq', 'fun': lambda x:  x[0] * max_opt_to_stock -  x[1] })
                
                        #Optimization
                        #res = minimize(f, guess, method='SLSQP', constraints=cons)    
                        res = minimize(f, guess, method='COBYLA', constraints=cons, tol=1.e-9) 
                        
                        stockPos_opt = res.x[0]
                        optNotional_opt = res.x[1]
                        optStrike_opt = res.x[2]
                        
                    elif use_options == 'Stock': # Simplified case - no options
                        f = objFun
                        guess = np.array([100])
                        
                        #Optimization
                        res = minimize(f, guess, method='COBYLA', tol=1.e-9) 
                        stockPos_opt = res.x
                        optNotional_opt = 0
                    
                    else:
                        print('Unknown optimization flag:', use_options)
                        
                    if use_options == 'Opts':
                        print('Optimal Stock Position = {:.1f}'.format(stockPos_opt))
                        print('Optimal Put Notional   = {:.1f}'.format(optNotional_opt))
                        
                        stockPosList.append(stockPos_opt)
                        optNotionalList.append(optNotional_opt)
                    
                    elif use_options == 'Opts+K':
                        print('Optimal Stock Position = {:.1f}'.format(stockPos_opt))
                        print('Optimal Put Notional   = {:.1f}'.format(optNotional_opt))
                        print('Optimal Put Strike     = {:.1f}'.format(optStrike_opt))
                        
                        stockPosList.append(stockPos_opt)
                        optNotionalList.append(optNotional_opt)
                        optStrikeList.append(optStrike_opt)
                        port = Portfolio(opt_strike = optStrike_opt)
                    else:
                        print('Optimal Stock Position = {:.1f}'.format(stockPos_opt[0]))
                        stockPosList.append(stockPos_opt[0])
                    
                    # Calculate expected returns and append the list
                    port.buyStocks(stockPosList[-1])
                    if use_options != 'Stock':
                        opt_mkt_px = port.optionMktPrice(stockScenarios_risk_neutral, \
                                                         prem2ev = prem2ev)
                        port.opt_mkt_px = opt_mkt_px
                        port.buyOpts(optNotionalList[-1],port.opt_mkt_px)
                    
                    #Calculate expected utility of the optimal portfolio
                    util, expVal, impRfVal = port.portExpUtil(stockScenarios, util_func)
            
                    expValList.append(expVal - 100)
                    impRfValList.append(impRfVal - 100)
                    
                    print('Optimal Stock Positions: ', stockPosList)
                    print('Optimal Opt Notionals: ', optNotionalList)
                    print('Portfolio Return: ', expValList)
                    print('Risk-Adj Returns: ', impRfValList)
                    print('Optimal Strikes:', optStrikeList)
                    
                    #Write data for csv file
                    filename = 'Run_Data_skew_{:.1f}_{}.csv'.format(ret_skew,use_options)

                    data = [risk_aversion_list, \
                            stockPosList, optNotionalList, expValList, \
                            impRfValList, optStrikeList ]

                    with open(filename, 'w', encoding='UTF8', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(data)