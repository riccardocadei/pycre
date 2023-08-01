from sklearn.base import BaseEstimator

def check_args(args):

    # Reproducibility
    if args.ratio_dis < 0 or args.ratio_dis > 1:
        raise ValueError("""'ratio_dis' parameter must be between 0 
                         and 1""")
    
    # Discovery
    if args.n_trees < 1:
        raise ValueError("'n_trees' parameter must be greater than 0")
    if args.max_depth < 1:
        raise ValueError("""'max_depth' parameter must be greater than 
                         0""")
    if args.node_size < 1:
        raise ValueError("""'node_size' parameter must be greater than 
                         0""")
    if args.max_rules < 1:
        raise ValueError("""'max_rules' parameter must be greater than 
                         0""")
    if args.decimal < 0:
        raise ValueError("""'decimal' parameter must be greater than 
                         or equal to 0""")
    if args.t_ext < 0 or args.t_ext > 1:
        raise ValueError("""'t_ext' parameter must be between 0 and 
                         1""")
    if args.t_corr < 0 or args.t_corr > 1:
        raise ValueError("""'t_corr' parameter must be between 0 and 
                         1""")
    if args.t_ss < 0 or args.t_ss > 1:
        raise ValueError("""'t_ss' parameter must be between 0 and 
                         1""")
    if args.B < 1:
        raise ValueError("'B' parameter must be greater than 0")
    if args.subsample < 0 or args.subsample > 1:
        raise ValueError("""'subsample' parameter must be between 0 
                         and 1""")
    
    # General
    if not isinstance(args.learner_y, BaseEstimator):
        raise ValueError("""'learner_y' parameter must be a sklearn 
                         estimator""")
    if not isinstance(args.learner_ps, BaseEstimator):
        raise ValueError("""'learner_ps' parameter must be a sklearn 
                         estimator""")
    if args.method not in ['tlearner', 'slearner', 'xlearner', 'aipw', 
                           'drlearner', 'causalforest']:
        raise ValueError("""'method' parameter doesn't exist or it 
                         hasn't been implemented yet""")
    