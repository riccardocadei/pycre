import pandas as pd
import numpy as np

def dataset_generator(n = 1000, 
                      k = 10, 
                      binary_cov = True,
                      binary_out = False, 
                      effect_size = 2,
                      confounding = "no",
                      n_rules = 2):
    """
    Generate a Syntethic Dataset

    Input
        n: number of observations
        k: number of covariates 
        binary_cov: whether the outcome is binary or continuos
        binary_out: whether the outcome is binary or continuos
        effect_size: effect size magnitude
        confounding: confounding mechanism
        n_rules: number of decision rules driving the HTE

    Output
        X: pd.DataFrame with Covariates ('name1', ..., 'namek')
        y: pd.Series with Outcome ('y')
        z: pd.Series with Treatment ('z')
        ite: pd.Series with ITE ('ite')
    """

    # Covariates (no correlation)
    if binary_cov:
        X = np.random.binomial(n = 1, 
                               p = 0.5, 
                               size = (n, k)) 
    else: 
        X = np.random.uniform(low = 0, 
                              high = 1, 
                              size = (n, k))

    X_names = ["x"+str(i) for i in range(1,k+1)]
    X = pd.DataFrame(data = X, 
                     columns = X_names)

    # Treatment
    logit = -1 -X["x1"] + X["x2"] + X["x3"]
    prob = np.exp(logit) / (1 + np.exp(logit))
    z = np.random.binomial(n = 1,
                           p = prob, 
                           size = n)
    
    # Outcome
    if binary_out: 
        y0 = np.zeros(n)
        y1 = np.zeros(n)
        effect_size = 1       
    else: 
        if confounding=="no":
            mu = 0
        elif confounding=="lin":
            mu = X["x1"]+X["x3"]+X["x4"]
        elif confounding=="nonlin":
            mu = X["x1"]+np.cos(X["x3"]*X["x4"])
        else:
            raise ValueError(f"`{confounding}` confounding mechanism  doesn't exists. Please select between 'no' for no confounding,'lin' for linear confounding,'nonlin' for non-linear confounding.")
        y0 = np.random.normal(loc = mu,
                              scale = 1,
                              size = n)
        y1 = y0.copy()
    
    # apply rules
    rule_1 = (X['x1']>0.5) & (X["x2"]<=0.5)
    rule_2 = (X["x5"]>0.5) & (X["x6"]<=0.5)
    rule_3 = (X["x4"]>0.5)
    rule_4 = (X['x5']<=0.5) & (X["x7"]>0.5) & (X["x8"]<=0.5)

    if n_rules>=1:
        y0[rule_1] += effect_size
    if n_rules>=2:
        y1[rule_2] += effect_size
    if n_rules>=3:
        if binary_out:
            raise ValueError(f"Synthtic dataset with binary outcome and {n_rules} rules has not been implemented yet. Available 'n_rules' options: 1,2.")
        else:
            y0[rule_3] += (effect_size*0.5)
    if n_rules>=4:
            y1[rule_4] += (effect_size*2)
    if n_rules>=5:
        raise ValueError(f"Synthtic dataset with continuos outcome and {n_rules} rules has not been implemented yet. Available 'n_rules' options: 1,2,3,4.")
    
    y = y0 * (1-z) + y1 * z
    ite = y1 - y0

    return X, y, z, ite

def honest_splitting(X, y, z, ratio_dis = 0.5):
    """
    Honest Splitting

    Input
        X: pd.DataFrame with Covariates ('name1', ..., 'namek')
        y: pd.Series with Outcome ('y')
        z: pd.Series with Treatment ('z')
        ratio_dis: ratio of the observations used for discovery
    
    Output
        [[X_dis, y_dis, z_dis], [X_inf, y_inf, z_inf]]: 
            list of two lists with discovery and inference data
    """

    N = X.shape[0]
    N_dis = int(N*ratio_dis)
    indeces = np.random.permutation(N)

    X_dis = X.iloc[indeces[N_dis:]]
    y_dis = y[indeces[N_dis:]]
    z_dis = z[indeces[N_dis:]]

    X_inf = X.iloc[indeces[:N_dis]]
    y_inf = y[indeces[:N_dis]]
    z_inf = z[indeces[:N_dis]]

    return [[X_dis, y_dis, z_dis], [X_inf, y_inf, z_inf]]