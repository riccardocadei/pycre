import pandas as pd
import numpy as np

def get_dataset(args):
    """
    Get Dataset

    Input
        args: Experiment arguments
    Output
        dataset: pd.DataFrame with Covariates ('name1', ..., 'namek'), 
                 Treatments ('z') and Outcome ('y')
    """
    if args.dataset_name == "syntethic":
        dataset = generate_syn_dataset(n = args.n, 
                                       k = args.k, 
                                       binary = args.binary,
                                       effect_size = args.effect_size)
    else: 
        raise ValueError("TO DO: check if path exists and read data")
    
    if args.standardize:
        raise ValueError("TO DO: standardize covariates")

    if args.subsample!=1:
        dataset = dataset.sample(frac=args.subsample)

    return dataset

def generate_syn_dataset(n = 100, 
                         k = 10, 
                         binary_cov = True,
                         binary_out = True, 
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
        dataset: pd.DataFrame with Covariates ('x1', ..., 'xk'), 
                 Treatments ('t') and Outcome ('y')
    """

    # Covariates (no correlation)
    if binary_cov:
        X = np.random.binomial(n = 1, 
                               p = 0.5, 
                               size = (100, 10)) 
    else: 
        X = np.random.uniform(low = 0, 
                              high = 1, 
                              size = (n, k))

    X_names = ["x"+str(i) for i in range(1,k+1)]
    dataset = pd.DataFrame(data = X, 
                           columns = X_names)

    # Treatment
    logit = -1 -dataset["x1"] + dataset["x2"] + dataset["x3"]
    prob = np.exp(logit) / (1 + np.exp(logit))
    t = np.random.binomial(n = 1,
                           p = prob, 
                           size = n)
    dataset['t'] = t
    
    # Outcome
    if binary_out: 
        y0 = np.zeros(n)
        y1 = np.zeros(n)
        effect_size = 1       
    else: 
        if confounding=="no":
            mu = 0
        elif confounding=="lin":
            mu = dataset["x1"]+dataset["x3"]+dataset["x4"]
        elif confounding=="nonlin":
            mu = dataset["x1"]+np.cos(dataset["x3"]*dataset["x4"])
        else:
            raise ValueError(f"`{confounding}` confounding mechanism 
                             doesn't exists. Please select between 
                             'no' for no confounding,'lin' for linear 
                             confounding,'nonlin' for non-linear 
                             confounding.")
        y0 = np.random.normal(loc = mu,
                              scale = 1,
                              size = n)
        y1 = y0
    
    # apply rules
    rule_1 = dataset['x1']>0.5 & dataset["x2"]<=0.5
    rule_2 = dataset["x5"]>0.5 & dataset["x6"]<=0.5
    rule_3 = dataset["x4"]>0.5
    rule_4 = dataset['x5']<=0.5 & dataset["x7"]>0.5 & dataset["x8"]<=0.5

    if n_rules>=1:
        y0[rule_1] += effect_size
    if n_rules>=2:
        y1[rule_2] += effect_size
    if n_rules>=3:
        if binary_out:
            raise ValueError(f"Synthtic dataset with binary outcome 
                             and {n_rules} rules has not been 
                             implemented yet. Available 'n_rules' 
                             options: 1,2.")
        else:
            y0[rule_3] += (effect_size*0.5)
    if n_rules>=4:
            y1[rule_4] += (effect_size*2)
    if n_rules>=5:
        raise ValueError(f"Synthtic dataset with continuos outcome 
                         and {n_rules} rules has not been 
                         implemented yet. Available 'n_rules' 
                         options: 1,2,3,4.")
    
    y = y0 * (1-t) + y1 * t
    dataset['y'] = y

    ite = y1 - y0
    dataset['ite'] = ite 
    return dataset