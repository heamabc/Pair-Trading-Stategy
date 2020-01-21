from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline import CustomFactor
from quantopian.pipeline.data.morningstar import asset_classification, valuation
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.pipeline.experimental import QTradableStocksUS
from itertools import combinations
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import pandas as pd

class ADF(object):
    
    def __init__(self):
        self.p_val = None
        self.lookback = 120
        self.p_max = 0.1
    
    def apply_adf(self, time_series):
        model = ts.adfuller(time_series, 1)
        self.p_val = model[1]
    
    def use_p(self):
        return self.p_val < self.p_max

class Half_Life(object):
    
    def __init__(self):
        self.hl_min = 2.0
        self.hl_max = 42.0
        self.lookback = 120
        self.half_life = None
    
    def apply_half_life(self, time_series):
        lag = np.roll(time_series, 1)
        lag[0] = 0
        ret = time_series - lag
        ret[0] = 0
        
        lag2 = sm.add_constant(lag)
        result = sm.OLS(ret, lag2).fit()
        
        self.half_life = -np.log(2) / result.params[1]
        
    def use(self):
        return (self.half_life > self.hl_min) and (self.half_life < self.hl_max)
    
class Hurst(object):
    
    def __init__(self):
        self.h_max = 0.4
        self.lookback = 120
        self.lag_max = 100
        self.h_value = None
    
    def apply_hurst(self, time_series):
        
        lags = range(2, self.lag_max)
        tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log10(lags), np.log10(tau), 1)
        self.h_value = poly[0]*2.0 
        
    def use(self):
        return self.h_value < self.h_max 

#------------------------------------------------------------------------------------------------------
def hedge_ratio(Y, X):
    X = sm.add_constant(X)
    result = sm.OLS(Y, X).fit()
    return result.params[0], result.params[1]


    
def initialize(context):
    
    context.s1 = sid(5061)
    context.s2 = sid(8655)
    context.s3 = sid(17787)
    context.s4 = sid(4707)
    context.s5 = sid(35920)
    context.s6 = sid(32146)
    context.s7 = sid(4283)
    context.s8 = sid(5885)
    context.s9 = sid(12002)
    context.s10 = sid(15596)
    context.s11 = sid(679)
    context.s12 = sid(1937)
    context.s13 = sid(8347)
    context.s14 = sid(3895)
    context.s15 = sid(14516)
    context.s16 = sid(14517)
    context.s17 = sid(21787)
    context.s18 = sid(21520)
    context.s19 = sid(21524)
    context.s20 = sid(23216)
    context.s21 = sid(19725)
    context.s22 = sid(3766)
    context.s23 = sid(49506)
    context.s24 = sid(5692)
    context.s25 = sid(24757)
    context.s26 = sid(7904)
    context.s27 = sid(20774)
    context.s28 = sid(7493)
    context.s29 = sid(438)
    context.s30 = sid(4914)

    
    context.asset_pairs = [[context.s1, context.s2, {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([]), 'residual': np.array([]), 'z_scores' : np.array([])}], 
                           [context.s3, context.s4, {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([]), 'residual': np.array([]), 'z_scores' : np.array([])}],
                          [context.s5, context.s6, {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([]), 'residual': np.array([]), 'z_scores' : np.array([])}],
                          [context.s7, context.s8, {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([]), 'residual': np.array([]), 'z_scores' : np.array([])}],
                          [context.s9, context.s10, {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([]), 'residual': np.array([]), 'z_scores' : np.array([])}],
                          [context.s11, context.s12, {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([]), 'residual': np.array([]), 'z_scores' : np.array([])}],
                          [context.s13, context.s14, {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([]), 'residual': np.array([]), 'z_scores' : np.array([])}],
                          [context.s15, context.s16, {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([]), 'residual': np.array([]), 'z_scores' : np.array([])}],
                          [context.s17, context.s18, {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([]), 'residual': np.array([]), 'z_scores' : np.array([])}],
                          [context.s19, context.s20, {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([]), 'residual': np.array([]), 'z_scores' : np.array([])}],
                          [context.s21, context.s22, {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([]), 'residual': np.array([]), 'z_scores' : np.array([])}],
                          [context.s23, context.s24, {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([]), 'residual': np.array([]), 'z_scores' : np.array([])}],
                          [context.s25, context.s26, {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([]), 'residual': np.array([]), 'z_scores' : np.array([])}],
                          [context.s27, context.s28, {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([]), 'residual': np.array([]), 'z_scores' : np.array([])}],
                          [context.s29, context.s30, {'in_short': False, 'in_long': False, 'spread': np.array([]), 'hedge_history': np.array([]), 'residual': np.array([]), 'z_scores' : np.array([])}]]
    
    context.z_back = 20
    context.hedge_lag = 2
    context.z_entry = 2.0
    context.z_exit = 0.5
    context.pair_count = len(context.asset_pairs)
    context.margin = 0.5
    context.universe = QTradableStocksUS()
    
    #attach_pipeline(pipeline_data(context), 'fund_data')
    
    schedule_function(my_handle_data, date_rules.every_day(),
                      time_rules.market_close(hours=4))

    
def my_handle_data(context, data):
    if get_open_orders():
        return
    
    for i in range(len(context.asset_pairs)):
        pair = context.asset_pairs[i]
        processed_pair = process_pair(pair, context, data)
        context.asset_pairs[i] = processed_pair
        
def process_pair(pair, context, data):
    
    s1 = pair[0]
    s2 = pair[1]
    prices = data.history([s1, s2], "price", 300, "1d")
    s1_p = prices[s1]
    s2_p = prices[s2]
    in_short = pair[2]["in_short"]
    in_long = pair[2]["in_long"]
    spread = pair[2]["spread"]
    hedge_history = pair[2]["hedge_history"]
    residual = pair[2]["residual"]
    z_scores = pair[2]["z_scores"]
    
    try:
        intercept, hedge = hedge_ratio(s1_p, s2_p)
    except ValueError as e:
        log.error(e)
        return [s1, s2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history, 'residual': residual, 'z_scores': z_scores}]
    
    hedge_history = np.append(hedge_history, hedge)
    residual = np.append(residual, s1_p[-1] - intercept - hedge * s2_p[-1])
    
    if hedge_history.size < context.hedge_lag :
        log.debug("hedge history too short")
        return [s1, s2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history, 'residual': residual, 'z_scores': z_scores}]
    
    
    '''
    if len(context.pair_pos) == context.max_pairs:
        return
    else:
        get_pairs(context,data)
    '''   
        
    hedge = hedge_history[-context.hedge_lag]
    spread = np.append(spread, s1_p[-1] - hedge * s2_p[-1])
    spread_length = spread.size
    
    adf = ADF()
    half_life = Half_Life()
    hurst = Hurst()
    
    
    if (spread_length < adf.lookback) or (spread_length < half_life.lookback) or (spread_length < hurst.lookback):
        return [s1, s2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history, 'residual': residual, 'z_scores': z_scores}]
    
    try:
        adf.apply_adf(spread[-adf.lookback:])
        half_life.apply_half_life(spread[-half_life.lookback:])
        hurst.apply_hurst(spread[-hurst.lookback:])
    except:
        return [s1, s2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history, 'residual': residual, 'z_scores': z_scores}]
    
    if (not adf.use_p() and not half_life.use()) or (not adf.use_p() and not hurst.use()) or (not half_life.use() and not hurst.use()):
        if in_short or in_long:
            log.info("Test has failed, exiting position")
            order_target(s1, 0)
            order_target(s2, 0)
            in_short = in_long = False
            return [s1, s2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history, 'residual': residual, 'z_scores': z_scores}]
        
        log.debug("Test failed")
        return [s1, s2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history, 'residual': residual, 'z_scores': z_scores}]
    
    
    spreads = spread[-context.z_back:]
    z = (spreads[-1] - spreads.mean())/spreads.std()
    z_scores = np.append(z_scores, z)
    
    if s1 == context.s1:
        record(p1_z=z)
        record(p1_hedge=hedge)
    else:
        record(p2_z=z)
        record(p2_hedge=hedge)
    
    if z_scores.size >= 3:
        if in_short and z_scores[-1] < context.z_exit and z_scores[-2] < context.z_exit and z_scores[-3] < context.z_exit:
            order_target(s1, 0)
            order_target(s2, 0)
            in_short = in_long = False
            return [s1, s2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history, 'residual': residual, 'z_scores': z_scores}]
        elif in_long and z_scores[-1] > -context.z_exit and z_scores[-2] > -context.z_exit and z_scores[-3] > -context.z_exit :
            order_target(s1, 0)
            order_target(s2, 0)
            in_short = in_long = False
            return [s1, s2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history, 'residual': residual, 'z_scores': z_scores}]
    
    
        value = context.portfolio.portfolio_value/context.pair_count/context.margin
    
        if z_scores[-1] < -context.z_entry and z_scores[-2] < -context.z_entry and not in_long:
            s1_pos = 1
            s2_pos = -hedge
            in_long = True
            in_short = False
            order_target_value(s1, value * s1_pos)
            order_target_value(s2, value * s2_pos)
            return [s1, s2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history, 'residual': residual, 'z_scores': z_scores}]
        elif z_scores[-1] > context.z_entry and z_scores[-2] > context.z_entry and not in_short:
            s1_pos = -1
            s2_pos = hedge
            in_long = False
            in_short = True
            order_target_value(s1, value * s1_pos)
            order_target_value(s2, value * s2_pos)
            return [s1, s2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history, 'residual': residual, 'z_scores': z_scores}]
    
    return [s1, s2, {'in_short': in_short, 'in_long': in_long, 'spread': spread, 'hedge_history': hedge_history, 'residual': residual, 'z_scores': z_scores}]
