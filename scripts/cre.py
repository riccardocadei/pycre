import logging

from parser import get_parser
from dataset import get_dataset
from ite import estimate_ite_ipw
from utils import standardize
from decision_rules import generate_rules, get_rules_matrix, rules_filtering
from aate import estimate_aate

import numpy as np
import pandas as pd

def CRE(dataset, args):

    # 1. Discovery
    print(f"- Discovery Step:")
    
    # Honest Splitting
    args.n = dataset.shape[0]
    n_dis = int(args.n*args.ratio_dis)
    indeces = np.random.permutation(args.n)

    dataset_dis = dataset.iloc[indeces[n_dis:]]
    y_dis = dataset_dis["y"]
    t_dis = dataset_dis["t"]
    if "ite" in dataset_dis:
        ite_dis = dataset_dis["ite"]
        X_dis = dataset_dis.drop(['y', 't', 'ite'], axis=1)
    else:
        X_dis = dataset_dis.drop(['y', 't'], axis=1)

    dataset_inf = dataset.iloc[indeces[:n_dis]]
    y_inf = dataset_inf["y"]
    t_inf = dataset_inf["t"]
    if "ite" in dataset_dis:
        ite_inf = dataset_inf["ite"]
        X_inf = dataset_inf.drop(['y', 't', 'ite'], axis=1)
    else:
        X_inf = dataset_inf.drop(['y', 't'], axis=1)

    # Esitimate ITE
    print(f"    ITE Estimation")
    ite_dis = estimate_ite_ipw(X = X_dis, 
                               y = y_dis, 
                               t = t_dis)
    ite_dis_std = standardize(ite_dis)

    # Rules Generation
    print(f"    Rules Generation")
    rules = generate_rules(X = X_dis, 
                           ite = ite_dis_std)
    R_dis = get_rules_matrix(rules, X_dis)

    # Rules Filtering
    print(f"    Rules Filtering")
    rules = rules_filtering(R_dis)

    # Rules Selection
    print(f"    Rules Selection [TO DO]")

    # 2. Inference
    print(f"- Inference Step:")
    # Esitimate ITE
    print(f"    ITE Estimation")
    ite_inf = estimate_ite_ipw(X = X_inf, 
                               y = y_inf, 
                               t = t_inf)
    print(f"    AATE estimatation")
    R_inf = get_rules_matrix(rules, X_inf)
    #R_inf.to_csv("results/R_inf.csv")
    AATE = estimate_aate(R_inf, ite_inf)
    print(AATE.summary())
    return

def main(args):
    # reproducibility
    np.random.seed(args.seed)

    print(f"Load {args.dataset_name} dataset")
    dataset = get_dataset(args)
    
    print(f"Run CRE algorithm")
    result = CRE(dataset, args)

    return result

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
