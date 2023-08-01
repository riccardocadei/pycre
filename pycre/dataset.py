import pandas as pd
import numpy as np

def dataset_generator(N = 1000, 
                      P = 10, 
                      M = 2,
                      binary_cov = True,
                      binary_out = False, 
                      effect_size = 2,
                      confounding = "no",
                      seed = 1):
    """
    Generate a Syntethic Dataset

    Parameters
    ----------
    N: int, default=1000
        Number of observations
    P: int, default=10
        Number of covariates
    M: int, default=2
        Number of decision rules driving the HTE
    binary_cov: bool, default=True
        Whether the covariates are binary or continuos
    binary_out: bool, default=False
        Whether the outcome is binary or continuos
    effect_size: float, default=2
        Effect size magnitude
    confounding: str, default="no"
        Confounding mechanism
    seed: int, default=1
        Seed
    
    Returns
    -------
    X: pd.DataFrame 
        Cobariates Matrix (N x P)
    y: pd.Series
        Outcome Vector (N)
    z: pd.Series
        Treatment Vector (N)
    ite: pd.Series
        ITE Vector (N)
    """
    
    # set seed
    np.random.seed(seed)
    # Covariates (no correlation)
    if binary_cov:
        X = np.random.binomial(n = 1, 
                               p = 0.5, 
                               size = (N, P)) 
    else: 
        X = np.random.uniform(low = 0, 
                              high = 1, 
                              size = (N, P))

    X_names = ["x"+str(i) for i in range(1,P+1)]
    X = pd.DataFrame(data = X, 
                     columns = X_names)

    # Treatment
    logit = -1 -X["x1"] + X["x2"] + X["x3"]
    prob = np.exp(logit) / (1 + np.exp(logit))
    z = np.random.binomial(n = 1,
                           p = prob, 
                           size = N)
    
    # Outcome
    if binary_out: 
        y0 = np.zeros(N)
        y1 = np.zeros(N)
        effect_size = 1       
    else: 
        if confounding=="no":
            mu = 0
        elif confounding=="lin":
            mu = X["x1"]+X["x3"]+X["x4"]
        elif confounding=="nonlin":
            mu = X["x1"]+np.cos(X["x3"]*X["x4"])
        else:
            raise ValueError(f"`{confounding}` confounding mechanism "  
                             "doesn't exists. Please select between "
                             "'no' for no confounding,'lin' for "
                             "linear confounding,'nonlin' for "
                             "non-linear confounding.")
        y0 = np.random.normal(loc = mu,
                              scale = 1,
                              size = N)
        y1 = y0.copy()
    
    # apply rules
    rule_1 = (X['x1']>0.5) & (X["x2"]<=0.5)
    rule_2 = (X["x5"]>0.5) & (X["x6"]<=0.5)
    rule_3 = (X["x4"]>0.5)
    rule_4 = (X['x5']<=0.5) & (X["x7"]>0.5) & (X["x8"]<=0.5)

    if M>=1:
        y0[rule_1] += effect_size
    if M>=2:
        y1[rule_2] += effect_size
    if M>=3:
        if binary_out:
            raise ValueError("Synthtic dataset with binary outcome "
                             f"and {M} rules has not been implemented "
                             "yet. Available 'n_rules' options: 1,2.")
        else:
            y0[rule_3] += (effect_size*0.5)
    if M>=4:
            y1[rule_4] += (effect_size*2)
    if M>=5:
        raise ValueError("Synthtic dataset with continuos outcome "
                         f"and {M} rules has not been implemented "
                         "yet. Available 'n_rules' options: 1,2,3,4.")
    
    y = y0 * (1-z) + y1 * z
    ite = y1 - y0

    return X, y, z, ite

def honest_splitting(X, y, z, ratio_dis = 0.5):
    """
    Honest Splitting

    Parameters
    ----------
    X: pd.DataFrame 
        Covariates Matrix (N x P)
    y: pd.Series or np.ndarray
        Outcome Vector (N)
    z: pd.Series or np.ndarray
        Treatment Vector (N)
    ratio_dis: float, default=0.5
        Ratio of the observations used for discovery
    
    Returns
    -------
    list 
        list of triples [X,y,z] data for discovery and inference
    """

    N = X.shape[0]
    N_dis = int(N*ratio_dis)
    indeces = np.random.permutation(N)
    y = np.array(y)
    z = np.array(z)

    X_dis = X.iloc[indeces[N_dis:]]
    y_dis = y[indeces[N_dis:]]
    z_dis = z[indeces[N_dis:]]

    X_inf = X.iloc[indeces[:N_dis]]
    y_inf = y[indeces[:N_dis]]
    z_inf = z[indeces[:N_dis]]

    return [X_dis, y_dis, z_dis], [X_inf, y_inf, z_inf]