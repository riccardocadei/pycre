from decision_rules import generate_rules, get_rules_matrix

def predict(X, model):
    """
    Predict ITE for new data
    
    Input
        X: pd.DataFrame with with Covariates ('name1', ..., 'namek')
        model: pd.DataFrame with ATE and AATE estimates and
        confidence intervals
        
    Output
        ite: pd.DataFrame with ITE estimates
    """
    R = get_rules_matrix(list(model.index)[1:], X)
    R.insert(0, 'ATE', 1)
    ite = R.mul(model['coef'].values, axis=1).sum(axis=1)

    return ite