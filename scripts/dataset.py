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
                                       binary_cov = args.binary_cov,
                                       binary_out = args.binary_out,
                                       effect_size = args.effect_size,
                                       confounding = args.confounding,
                                       n_rules = args.n_rules)
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
                 Treatments ('z') and Outcome ('y')
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
    dataset = pd.DataFrame(data = X, 
                           columns = X_names)

    # Treatment
    logit = -1 -dataset["x1"] + dataset["x2"] + dataset["x3"]
    prob = np.exp(logit) / (1 + np.exp(logit))
    z = np.random.binomial(n = 1,
                           p = prob, 
                           size = n)
    dataset['z'] = z
    
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
            raise ValueError(f"`{confounding}` confounding mechanism  doesn't exists. Please select between 'no' for no confounding,'lin' for linear confounding,'nonlin' for non-linear confounding.")
        y0 = np.random.normal(loc = mu,
                              scale = 1,
                              size = n)
        y1 = y0
    
    # apply rules
    rule_1 = (dataset['x1']>0.5) & (dataset["x2"]<=0.5)
    rule_2 = (dataset["x5"]>0.5) & (dataset["x6"]<=0.5)
    rule_3 = (dataset["x4"]>0.5)
    rule_4 = (dataset['x5']<=0.5) & (dataset["x7"]>0.5) & (dataset["x8"]<=0.5)

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
    dataset['y'] = y

    ite = y1 - y0
    dataset['ite'] = ite 
    return dataset

def honest_splitting(dataset, ratio_dis = 0.5):
    """
    Honest Splitting

    Input
        dataset: pd.DataFrame with Covariates ('name1', ..., 'namek'), 
                 Treatments ('z') and Outcome ('y')
        ratio_dis: ratio of the observations used for discovery
    
    Output
        [[X_dis, y_dis, z_dis], [X_inf, y_inf, z_inf]]: 
            list of two lists with discovery and inference data
    """

    if "ite" in dataset: dataset = dataset.drop(['ite'], axis=1)
    N = dataset.shape[0]
    N_dis = int(N*ratio_dis)
    indeces = np.random.permutation(N)

    dataset_dis = dataset.iloc[indeces[N_dis:]]
    y_dis = dataset_dis["y"]
    z_dis = dataset_dis["z"]
    X_dis = dataset_dis.drop(['y', 'z'], axis=1)

    dataset_inf = dataset.iloc[indeces[:N_dis]]
    y_inf = dataset_inf["y"]
    z_inf = dataset_inf["z"]
    X_inf = dataset_inf.drop(['y', 'z'], axis=1)

    return [[X_dis, y_dis, z_dis], [X_inf, y_inf, z_inf]]