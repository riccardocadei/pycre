import statsmodels.api as sm
import pandas as pd

def estimate_aate(R, ite):
    
    ate = sm.OLS(endog = ite, 
                 exog = pd.Series(1, index=ite.index)).fit().summary().tables[1]
    ate = pd.DataFrame(ate.data[1:], columns=ate.data[0]).set_index("")
    ate.rename(index={'const': 'ATE'}, inplace=True)
    aate = sm.OLS(endog = ite, 
                  exog = R).fit().summary().tables[1]
    aate = pd.DataFrame(aate.data[1:], columns=aate.data[0]).set_index("")
    results = pd.concat([ate, aate])
    return results