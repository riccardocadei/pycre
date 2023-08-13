from decision_rules import get_rules_matrix

def predict(X, model):
    """
    Predict ITE by Causal Rule Ensemble (CRE)
    
    Parameters
    ----------
    X: pd.DataFrame 
        Covariates matrix (N x P)
    model: pd.DataFrame 
        ATE and AATE estimates with confidence intervals

    Returns
    -------
    pd.Series
        ITE estimates (N)
    """

    R = get_rules_matrix(list(model.index)[1:], X)
    R.insert(0, 'ATE', 1)
    ite = R.mul(model['coef'].values, axis=1).sum(axis=1)

    return ite