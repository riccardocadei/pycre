import statsmodels.api as sm
from scipy.stats import t
import pandas as pd

def estimate_aate(R, ite, B=1, subsample=0.7):
    """
    Estimate ATE and AATE
    
    Input
        R: pd.DataFrame Rules Matrix (N x M)
        ite: pd.Series with ITE estimates (N)
    Output
        model: pd.DataFrame with ATE and AATE estimates and 
        confidence intervals
    """

    if B==1:
        # estimate ATE
        ate = sm.OLS(endog = ite, 
                     exog = pd.Series(1, index=ite.index)).fit().summary().tables[1]
        ate = pd.DataFrame(ate.data[1:], columns=ate.data[0]).set_index("")
        ate.rename(index={'const': 'ATE'}, inplace=True)
        if R.empty:
            model = ate.astype(float)
        # estimate AATE
        else:
            aate = sm.OLS(endog = ite-float(ate.loc["ATE"].coef), 
                          exog = R).fit().summary().tables[1]
            aate = pd.DataFrame(aate.data[1:], columns=aate.data[0]).set_index("")
            model = pd.concat([ate, aate]).astype(float)
    else:
        model_list = []
        for _ in range(B):
            R_ = R.sample(frac=subsample, axis=0)
            ite_ = ite[R_.index]
            model = estimate_aate(R_, ite_, B=1)
            model_list.append(model)
        model = pd.concat(model_list).groupby(level=0).agg(['mean', 'std']).filter(like='coef').droplevel(0, axis=1).rename(columns={'mean': 'coef', 'std': 'std err'}).sort_index(ascending=False)
        model['t'] = model['coef'] / model['std err']
        model['P>|t|'] = 2 * (1 - t.cdf(abs(model['t']), len(R_) - len(model)))
        model['[0.025'] = model['coef'] - 1.96 * model['std err']
        model['0.975]'] = model['coef'] + 1.96 * model['std err']
    
    # discard not significant rules
    not_significant = model[model['P>|t|'] > 0.01].index
    if "ATE" in not_significant: 
        not_significant = not_significant.drop("ATE")
    if not not_significant.empty:
        R = R.drop(not_significant, axis=1)
        model = estimate_aate(R, ite, B, subsample)

    return model

    
