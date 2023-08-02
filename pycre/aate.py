import statsmodels.api as sm
from scipy.stats import t
import pandas as pd

def estimate_aate(R, ite, B=1, subsample=0.7):
    """
    Estimate ATE and AATE

    Parameters
    ----------
    R: pd.DataFrame 
        Rules Matrix (N x M)
    ite: pd.Series
        ITE estimates (N)
    B: int, default=1
        Number of bootstrap samples for uncertainty quantification.
    subsample: float, default=0.7
        Bootstrap ratio subsample for uncertainty quantification.

    Returns
    -------
    model: pd.DataFrame
        ATE and AATE estimates and confidence intervals
    """
    if B==1:
        # estimate ATE
        ate = sm.OLS(endog = ite, 
                     exog = pd.Series(1, index=ite.index))
        ate = ate.fit().summary().tables[1]
        ate = pd.DataFrame(ate.data[1:], 
                           columns=ate.data[0]).set_index("")
        ate.rename(index={'const': 'ATE'}, inplace=True)
        if R.empty:
            model = ate.astype(float)
        # estimate AATE
        else:
            aate = sm.OLS(endog = ite-float(ate.loc["ATE"].coef), 
                          exog = R).fit().summary().tables[1]
            aate = pd.DataFrame(aate.data[1:], 
                                columns=aate.data[0]).set_index("")
            model = pd.concat([ate, aate]).astype(float)
    else:
        model_list = []
        for _ in range(B):
            ite_ = ite.sample(frac=subsample, axis=0)
            R_ = R.loc[ite_.index,:]
            model = estimate_aate(R_, ite_, B=1)
            model_list.append(model)
        model = pd.concat(model_list).groupby(level=0)
        model = model.agg(['mean', 'std']).filter(like='coef')
        model = model.droplevel(0, axis=1)
        model = model.rename(columns={'mean': 'coef', 
                                      'std': 'std err'})
        model = model.sort_index(ascending=False)
        model['t'] = model['coef'] / model['std err']
        model['P>|t|'] = 2 * (1 - t.cdf(abs(model['t']), 
                                        len(R_) - len(model)))
        model['[0.025'] = model['coef'] - 1.96 * model['std err']
        model['0.975]'] = model['coef'] + 1.96 * model['std err']
    
    # discard not significant rules
    not_significant = model[model['P>|t|'] > 0.05].index
    if "ATE" in not_significant: 
        not_significant = not_significant.drop("ATE")
    if not not_significant.empty and B>1:
        R = R.drop(not_significant, axis=1)
        model = estimate_aate(R, ite, B, subsample)

    return model

    
