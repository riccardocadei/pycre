from sklearn.ensemble import GradientBoostingClassifier
import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # reproducibility
    parser.add_argument("--seed", default=1, type=int, help="seed")

    # dataset
    parser.add_argument("--dataset_name", default="syntethic", type=str, help="dataset name")
    parser.add_argument("--dataset_dir", default="", type=str, help="dataset directory")
    parser.add_argument("--standardize", default=False, type=bool, help="whether standardize or not the covariates")
    parser.add_argument("--subsample", default=1, type=float, help="ratio of the dataset to subsample")
    # synthetic dataset
    parser.add_argument("--n", default=1000, type=int, help="number of observations for the Syntetic Dataset")
    parser.add_argument("--k", default=10, type=int, help="number of covariates for the Syntetic Dataset")
    parser.add_argument("--binary_cov", default=True, type=bool, help="whether the covariates are binary or continuos for the Syntetic Dataset")
    parser.add_argument("--binary_out", default=True, type=bool, help="whether the outcome is binary or continuos for the Syntetic Dataset")
    parser.add_argument("--effect_size", default=2, type=float, help="effect size maginitude for the (continuos outcome) Syntetic Dataset")
    parser.add_argument("--confounding", default="no", type=str, help="confounding mechanism for the Syntetic Dataset: 'no'=no confounding, 'lin'=linear confounding, `nonlin`=non linear confounding")
    parser.add_argument("--n_rules", default=2, type=int, help="number of decision rules driving the HTE in the Syntetic Dataset")
    #splitting
    parser.add_argument("--ratio_dis", default=0.5, type=float, help="ratio of the observations used for discovery")

    # Discovery
    parser.add_argument("--n_trees", default=100, type=int, help="number of decision trees for rules discovery")
    parser.add_argument("--max_depth", default=2, type=int, help="max depth of decision trees for rules discovery (i.e., max decision rules depth)")
    parser.add_argument("--decimal", default=2, type=int, help="number of digits to round the rules' thresholds")
    parser.add_argument("--t_ext", default=0.02, type=float, help="threshold to discard too generic or too specific (extreme)")
    parser.add_argument("--t_corr", default=0.5, type=float, help="threshold to discard too correlated rules")
    parser.add_argument("--t_ss", default=0.5, type=float, help="threshold for stability selection in rules selection")
    parser.add_argument("--alphas", default=[0.1, 0.5, 1.0], type=list, help="alpha values for stability selection in rules selection")
    parser.add_argument("--B", default=50, type=int, help="number of bootstrap samples for stability selection in rules selection")
    parser.add_argument("--folds", default=5, type=int, help="number of folds for stability selection in rules selection")
    parser.add_argument("--ite_estimator_dis", default="causalforest", type=str, help="ITE estimator for discovery")
    
    # Inference
    parser.add_argument("--ite_estimator_inf", default="causalforest", type=str, help="ITE estimator for inference")

    # General
    parser.add_argument("--learner_y", default=GradientBoostingClassifier(), help="learner for outcome")
    parser.add_argument("--learner_ps", default=GradientBoostingClassifier(), help="learner for propensity score")
    
    return parser