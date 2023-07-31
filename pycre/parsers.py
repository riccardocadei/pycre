from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # Reproducibility
    parser.add_argument("--seed", default=1, type=int, help="seed")

    # Discovery
    parser.add_argument("--ratio_dis", default=0.5, type=float, help="ratio of the observations used for discovery")
    parser.add_argument("--n_trees", default=20, type=int, help="number of decision trees for rules discovery")
    parser.add_argument("--max_depth", default=2, type=int, help="max depth of decision trees for rules discovery (i.e., max decision rules depth)")
    parser.add_argument("--node_size", default=10, type=int, help="min number of observations in a leaf node for rules discovery")
    parser.add_argument("--max_rules", default=50, type=int, help="max number of generated candidate decision rules")
    parser.add_argument("--criterion", default="het", type=str, help="criterion for splitting decision trees for rules discovery")
    parser.add_argument("--decimal", default=2, type=int, help="number of digits to round the rules' thresholds")
    parser.add_argument("--t_ext", default=0.02, type=float, help="threshold to discard too generic or too specific (extreme)")
    parser.add_argument("--t_corr", default=0.5, type=float, help="threshold to discard too correlated rules")
    parser.add_argument("--t_ss", default=0.8, type=float, help="threshold for stability selection in rules selection")
    parser.add_argument("--alphas", default=[0.01,0.1, 1], type=list, help="alpha values for stability selection in rules selection")
    parser.add_argument("--B", default=20, type=int, help="number of bootstrap samples for stability selection in rules selection and uncertainty quantification in estimation")
    parser.add_argument("--subsample", default=0.5, type=float, help="bootstrap ratio subsample for forest generation and stability selection in rules selection, and uncertainty quantification in estimation")

    # General
    parser.add_argument("--learner_y", default=GradientBoostingRegressor(), help="learner for outcome")
    parser.add_argument("--learner_ps", default=GradientBoostingClassifier(), help="learner for propensity score")
    parser.add_argument("--method", default="tlearner", type=str, help="ITE estimator")
    
    # Describe
    parser.add_argument("--exp_name", default="example.png", type=str, help="experiment name")
    parser.add_argument("--verbose", default=True, type=bool, help="verbose")
    parser.add_argument("--path", default="../results/", type=str, help="path to save results (i.e., plots)")
    parser.add_argument("--save", default=False, type=bool, help="save results (i.e., plots)")

    return parser