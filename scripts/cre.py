from parser import get_parser
from dataset import get_dataset, honest_splitting
from ite import estimate_ite
from decision_rules import generate_rules, get_rules_matrix, rules_filtering, stability_selection
from aate import estimate_aate
from visualize import plot

import pandas as pd

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np

def fit(dataset, args):
    """
    Fit CRE model
    Input
        dataset: pd.DataFrame with with Covariates ('name1', ..., 'namek'), 
                 Treatments ('z') and Outcome ('y')
        args: arguments from parser
    Output
        results: pd.DataFrame with ATE and AATE estimates and
        confidence intervals
    """

    # 0. Honest Splitting
    if args.verbose: print("- Honest Splitting")
    dis, inf = honest_splitting(dataset, args.ratio_dis)
    X_dis, y_dis, z_dis = dis
    X_inf, y_inf, z_inf = inf

    # 1. Discovery
    if args.verbose: print("- Discovery Step:")

    # Esitimate ITE
    if args.verbose: print("    ITE Estimation")
    ite_dis = estimate_ite(X = X_dis, 
                           y = y_dis, 
                           z = z_dis,
                           method = args.ite_estimator_dis,
                           learner_y = args.learner_y,
                           learner_ps = args.learner_ps)

    # Rules Generation
    if args.verbose: print("    Rules Generation")
    rules = generate_rules(X = X_dis, 
                           ite = ite_dis,
                           n_trees = args.n_trees, 
                           max_depth = args.max_depth,
                           decimal = args.decimal)
    R_dis = get_rules_matrix(rules, X_dis)
    if args.verbose: print(f"      {R_dis.shape[1]} rules generated")

    # Rules Filtering
    if args.verbose: print("    Rules Filtering")
    R_dis = rules_filtering(R_dis,
                            t_ext = args.t_ext, 
                            t_corr = args.t_corr,)
    if args.verbose: print(f"      {R_dis.shape[1]} rules filtered")

    # Rules Selection
    if args.verbose: print(f"    Rules Selection")
    rules = stability_selection(R_dis, ite_dis, 
                                t_ss = args.t_ss, 
                                B = args.B,
                                alphas = args.alphas,
                                folds = args.folds)
    if args.verbose: print(f"      {len(rules)} candidate rules selected")

    # 2. Inference
    if args.verbose: print("- Inference Step:")
    # Esitimate ITE
    if args.verbose: print("    ITE Estimation")
    ite_inf = estimate_ite(X = X_inf, 
                           y = y_inf, 
                           z = z_inf,
                           method = args.ite_estimator_inf,
                           learner_y = args.learner_y,
                           learner_ps = args.learner_ps)
    
    if args.verbose: print("    AATE estimatation")
    R_inf = get_rules_matrix(rules, X_inf)
    results = estimate_aate(R_inf, ite_inf)
    plot(results,
         xrange = args.xrange,
         save = args.save,
         path = args.path,
         exp_name = args.exp_name)
    if args.verbose: 
        temp = results.copy()
        temp.index = temp.index.str.replace("\(X\['|\)|'\]", "", regex=True)
        print(temp)

    return results

def main(args):
    # set seed (reproducibility)
    np.random.seed(args.seed)

    if args.verbose: print(f"Load {args.dataset_name} dataset")
    dataset = get_dataset(args)
    
    if args.verbose: print("Run CRE algorithm")
    model = CRE(args)
    model.fit(dataset)
    ite = model.eval(dataset)

    return ite

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)

def predict(X, result):
    """
    Predict ITE for new data
    
    Input
        X: pd.DataFrame with with Covariates ('name1', ..., 'namek')
        result: pd.DataFrame with ATE and AATE estimates and
        confidence intervals
        
    Output
        ite: pd.DataFrame with ITE estimates
    """
    R = get_rules_matrix(list(result.index)[1:], X)
    R.insert(0, 'ATE', 1)
    ite = R.mul(result['coef'].values, axis=1).sum(axis=1)

    return ite


class CRE:
    def __init__(self, args):
        self.args = args

    def fit(self, dataset):
        self.dataset = dataset
        self.model = fit(dataset, self.args)
        self.rules = list(self.model.index)[1:]

    def eval(self, X):
        self.ite = predict(X, self.model)
        return self.ite
