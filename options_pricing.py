import math
from scipy.stats import norm
import pandas as pd
from datetime import date, datetime, timedelta
import numpy as np

 
#===============================================================================
# CLASS OPTION  
#===============================================================================
class Option:
    """
    This class will group the different black-shcoles calculations for an opion
    """
    def __init__(self, right, s, k, eval_date, exp_date, price = None, rf = 0.01, vol = 0.3,
                 div = 0):
        self.k = float(k)
        self.s = float(s)
        self.rf = float(rf)
        self.vol = float(vol)
        self.eval_date = datetime.strptime(eval_date, '%m/%d/%y %H:%M')
        self.exp_date = datetime.strptime(exp_date, '%m/%d/%y')
        self.t = self.calculate_t()
        if self.t == 0: self.t = 0.000001 ## Case valuation in expiration date
        self.price = price
        self.right = right   ## 'C' or 'P'
        self.div = div
 
    def calculate_t(self):
        #d1 is expiration date, d0 is evaluation date
        if self.exp_date.day > datetime.today().day:
            EOD = datetime.today() + timedelta(exp_date.day-datetime.today().day)
            EOD = EOD.replace(hour = 16, minute=0, second = 0, microsecond = 0)
        else:
            EOD = datetime.today().replace(hour = 16,minute= 0, second=0,microsecond = 0)

        Now = datetime.today().replace(second = 0, microsecond = 0)
        diff = (EOD-Now)
        
        if diff.days >= 1:       
            div = diff.total_seconds()/3600
            return div/24
        else:
            #change div to diff between now and 4pm
            
            div = 6.5
            return float(diff.seconds/3600)/div
        
        
        
 
        
 
    def get_price_delta(self):
        d1 = ( math.log(self.s/self.k) + ( self.rf + self.div + math.pow( self.vol, 2)/2 ) * self.t ) / ( self.vol * math.sqrt(self.t) )
        d2 = d1 - self.vol * math.sqrt(self.t)
        if self.right == 'C':
            self.calc_price = ( norm.cdf(d1) * self.s * math.exp(-self.div*self.t) - norm.cdf(d2) * self.k * math.exp( -self.rf * self.t ) )
            self.delta = norm.cdf(d1)
        elif self.right == 'P':
            self.calc_price =  ( -norm.cdf(-d1) * self.s * math.exp(-self.div*self.t) + norm.cdf(-d2) * self.k * math.exp( -self.rf * self.t ) )
            self.delta = -norm.cdf(-d1) 
 
    def get_call(self):
        d1 = ( math.log(self.s/self.k) + ( self.rf + math.pow( self.vol, 2)/2 ) * self.t ) / ( self.vol * math.sqrt(self.t) )
        d2 = d1 - self.vol * math.sqrt(self.t)
        self.call = ( norm.cdf(d1) * self.s - norm.cdf(d2) * self.k * math.exp( -self.rf * self.t ) )
        #put =  ( -norm.cdf(-d1) * self.s + norm.cdf(-d2) * self.k * math.exp( -self.rf * self.t ) ) 
        self.call_delta = norm.cdf(d1)
 
 
    def get_put(self):
        d1 = ( math.log(self.s/self.k) + ( self.rf + math.pow( self.vol, 2)/2 ) * self.t ) / ( self.vol * math.sqrt(self.t) )
        d2 = d1 - self.vol * math.sqrt(self.t)
        #call = ( norm.cdf(d1) * self.s - norm.cdf(d2) * self.k * math.exp( -self.rf * self.t ) )
        self.put =  ( -norm.cdf(-d1) * self.s + norm.cdf(-d2) * self.k * math.exp( -self.rf * self.t ) )
        self.put_delta = -norm.cdf(-d1) 
 
 
    def get_theta(self, dt = 0.0027777):
        self.t += dt
        self.get_price_delta()
        after_price = self.calc_price
        self.t -= dt
        self.get_price_delta()
        orig_price = self.calc_price
        self.theta = (after_price - orig_price) * (-1)
 
    def get_gamma(self, ds = 0.01):
        self.s += ds
        self.get_price_delta()
        after_delta = self.delta
        self.s -= ds
        self.get_price_delta()
        orig_delta = self.delta
        self.gamma = (after_delta - orig_delta) / ds
 
    def get_all(self):
        self.get_price_delta()
        self.get_theta()
        self.get_gamma()
        return self.calc_price, self.delta, self.theta, self.gamma
 
 
    def get_impl_vol(self):
        """
        This function will iterate until finding the implied volatility
        """
        ITERATIONS = 100
        ACCURACY = 0.05
        low_vol = 0
        high_vol = 1
        self.vol = 0.5  ## It will try mid point and then choose new interval
        self.get_price_delta()
        for i in range(ITERATIONS):
            if self.calc_price > self.price + ACCURACY:
                high_vol = self.vol
            elif self.calc_price < self.price - ACCURACY:
                low_vol = self.vol
            else:
                break
            self.vol = low_vol + (high_vol - low_vol)/2.0
            self.get_price_delta()
 
        return self.vol
 
    def get_price_by_binomial_tree(self):
        """
        This function will make the same calculation but by Binomial Tree
        """
        N=30
        deltaT=self.t/N
        u = math.exp(self.vol*math.sqrt(deltaT))
        d=1.0/u
        # Initialize our f_{i,j} tree with zeros
        #fs = [[0.0 for j in list(range(i+1))] for i in list(range(n+1))]
        a = math.exp(self.rf*deltaT)
        p = (a-d)/(u-d)
        oneMinusP = 1.0-p 
        # Compute the leaves, f_{N,j}
        
        stock = np.zeros([N + 1, N + 1])
        for i in range(N + 1):
            for j in range(i + 1):
                stock[j, i] = self.s * (u ** (i - j)) * (d ** j)
        
 
        option = np.zeros([N + 1, N + 1])
        option[:, N] = np.maximum(np.zeros(N + 1), (stock[:, N] - self.k))
        for i in range(N - 1, -1, -1):
            for j in range(0, i + 1):
                option[j, i] = (
                    1 / (1 + self.rf) * (p * option[j, i + 1] + oneMinusP * option[j + 1, i + 1])
                )
        return option[0][0]
 
       
class Options_strategy:
    """
    This class will calculate greeks for a group of options (called Options Strategy)
    """
    def __init__(self, df_options): 
        self.df_options = df_options       #It will store the different options in a pandas dataframe
 
    def get_greeks(self):
        """
        For analysis underlying (option chain format)
        """
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        for k,v in self.df_options.iterrows():
 
            ## Case stock or future
            if v['m_secType']=='STK':
                self.delta += float(v['position']) * 1
 
            ## Case option
            elif v['m_secType']=='OPT':    
                opt = Option(s=v['underlying_price'], k=v['m_strike'], eval_date=datetime.strftime(datetime.today(), "%m/%d/%y %H:%M"), # We want greeks for today
                             exp_date=v['m_expiry'], rf = v['interest'], vol = v['volatility'],
                             right = v['m_right'])
 
                price, delta, theta, gamma = opt.get_all()
 
                self.delta += float(v['position']) * delta
                self.gamma += float(v['position']) * gamma
                self.theta += float(v['position']) * theta
 
            else:
                print("ERROR: Not known type")
 
        return self.delta, self.gamma, self.theta 
 
    def get_greeks2(self):
        """
        For analysis_options_strategy
        """
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        for k,v in self.df_options.iterrows():
 
            ## Case stock or future
            if v['m_secType']=='STK':
                self.delta += float(v['position']) * 1
 
            ## Case option
            elif v['m_secType']=='OPT':    
                opt = Option(s=v['underlying_price'], k=v['m_strike'], eval_date=datetime.strftime(datetime.today(), "%m/%d/%y %H:%M"), # We want greeks for today
                             exp_date=v['m_expiry'], rf = v['interest'], vol = v['volatility'],
                             right = v['m_right'])
 
                price, delta, theta, gamma = opt.get_all()
 
                if v['m_side']=='BOT':
                    position = float(v['position'])
                else:
                    position = - float(v['position']) 
                self.delta += position * delta
                self.gamma += position * gamma
                self.theta += position * theta
 
            else:
                print("ERROR: Not known type")
 
        return self.delta, self.gamma, self.theta 


if __name__ == '__main__':
 
 
    #===========================================================================
    # TO CHECK OPTION CALCULATIONS
    #===========================================================================
    s = 56.37
    k = 60
    exp_date = '02/15/19'
    eval_date = '02/15/19 09:32'
    rf = 0.01
    vol = 0.2074
    div = 0.014
    right = 'C'
    opt = Option(s=s, k=k, eval_date=eval_date, exp_date=exp_date, rf=rf, vol=vol, right=right,
                 div = div)
    price, delta, theta, gamma = opt.get_all()
    print("-------------- FIRST OPTION -------------------")
    print("Price CALL: " + str(price))  # 2.97869320042
    print("Delta CALL: " + str(delta))  # 0.664877358932
    print("Theta CALL: " + str(theta))  # 0.000645545628288
    print("Gamma CALL:" + str(gamma))   # 0.021127937082
 
    price = opt.get_price_by_binomial_tree()
    print("Price by BT:" + str(price))
 
    s = 110.41
    k = 112
    exp_date = '02/15/19'
    eval_date = '02/15/19 09:32'
    rf = 0.01
    vol = 0.11925
    right = 'C'
    opt = Option(s=s, k=k, eval_date=eval_date, exp_date=exp_date, rf=rf, vol=vol, right=right)
    price, delta, theta, gamma = opt.get_all()
    print("-------------- SECOND OPTION -------------------")
    print("Price CALL: " + str(price))    # 7.02049813137
    print ("Delta CALL: " + str(delta))   # 0.53837898036
    print("Theta CALL: " + str(theta))    # -0.00699852931575
    print("Gamma CALL:" + str(gamma))     # 0.0230279263655
 
    #===========================================================================
    # TO CHECK OPTIONS STRATEGIES CALCULATIONS
    #===========================================================================
    d_option1 = {'m_secType': 'OPT', 'm_expiry': '02/15/19', 'm_right': 'C', 'm_symbol': 'TLT', 'm_strike': '115', 
                 'm_multiplier': '100', 'position': '-2', 'trade_price': '3.69', 'comission': '0',
                 'eval_date': '02/15/19 09:51', 'interest': '0.01', 'volatility': '0.12353', 'underlying_price': '109.96'}
    d_option2 = {'m_secType': 'OPT', 'm_expiry': '02/15/19', 'm_right': 'C', 'm_symbol': 'TLT', 'm_strike': '135', 
                 'm_multiplier': '100', 'position': '2', 'trade_price': '0.86', 'comission': '0',
                 'eval_date': '02/15/19 09:51', 'interest': '0.01', 'volatility': '0.12353', 'underlying_price': '109.96'}
 
    df_options = pd.DataFrame([d_option1, d_option2])
    #print('DF OPTIONS')
    #print(df_options)
    opt_strat = Options_strategy(df_options)
    delta, gamma, theta = opt_strat.get_greeks()
    print("-------- OPTIONS STRATEGY --------------")
    print("Delta: " + str(delta))
    print("Gamma: " + str(gamma))
    print("Theta: " + str(theta))
 
    
