from parser import get_parser
from dataset import get_dataset, honest_splitting
from ite import estimate_ite_ipw
from decision_rules import generate_rules, get_rules_matrix, rules_filtering, stability_selection
from aate import estimate_aate

import numpy as np

def CRE(dataset, args):
    """
    CRE algorithm
    Input
        dataset: pd.DataFrame with with Covariates ('name1', ..., 'namek'), 
                 Treatments ('z') and Outcome ('y')
        args: arguments from parser
    Output
        results: pd.DataFrame with ATE and AATE estimates and
        confidence intervals
    """

    # 0. Honest Splitting
    print("- Honest Splitting")
    dis, inf = honest_splitting(dataset, args.ratio_dis)
    X_dis, y_dis, z_dis = dis
    X_inf, y_inf, z_inf = inf

    # 1. Discovery
    print("- Discovery Step:")

    # Esitimate ITE
    print("    ITE Estimation")
    ite_dis = estimate_ite_ipw(X = X_dis, 
                               y = y_dis, 
                               z = z_dis)

    # Rules Generation
    print("    Rules Generation")
    rules = generate_rules(X = X_dis, 
                           ite = ite_dis,
                           n_trees = args.n_trees, 
                           max_depth = args.max_depth,
                           decimal = args.decimal)
    R_dis = get_rules_matrix(rules, X_dis)
    print(f"      {R_dis.shape[1]} rules generated")

    # Rules Filtering
    print("    Rules Filtering")
    R_dis = rules_filtering(R_dis)
    print(f"      {R_dis.shape[1]} rules filtered")

    # Rules Selection
    print(f"    Rules Selection")
    rules = stability_selection(R_dis, ite_dis, 
                                t_ss = args.t_ss, 
                                B = args.B,
                                alphas = args.alphas,
                                folds = args.folds)
    print(f"      {len(rules)} candidate rules selected")

    # 2. Inference
    print("- Inference Step:")
    # Esitimate ITE
    print("    ITE Estimation")
    ite_inf = estimate_ite_ipw(X = X_inf, 
                               y = y_inf, 
                               z = z_inf)
    print("    AATE estimatation")
    R_inf = get_rules_matrix(rules, X_inf)
    #R_inf.to_csv("results/R_inf.csv")
    results = estimate_aate(R_inf, ite_inf)
    results.index = results.index.str.replace("\(X\['|\)|'\]", "", regex=True)
    print(results)
    return results

def main(args):
    # set seed (reproducibility)
    np.random.seed(args.seed)

    print(f"Load {args.dataset_name} dataset")
    dataset = get_dataset(args)
    
    print("Run CRE algorithm")
    result = CRE(dataset, args)

    return result

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
