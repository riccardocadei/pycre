from dataset import honest_splitting
from arguments import check_data
from ite import estimate_ite
from decision_rules import generate_rules, get_rules_matrix
from decision_rules import rules_filtering, stability_selection
from aate import estimate_aate

import warnings
warnings.filterwarnings("ignore")

def train(X, y, z, args):
    """
    Fit CRE model

    Parameters
    ----------
    X: pd.DataFrame
        Covariates Matrix (N x K)
    y: pd.Series or np.ndarray
        Outcome (N)
    z: pd.Series or np.ndarray
        Treatment (N) 
    args: argparse.Namespace
        training parameters

    Returns
    -------
    pd.DataFrame
        ATE and AATE estimates and confidence intervals
    """

    check_data(X, y, z, args.learner_y)

    # 0. Honest Splitting
    if args.verbose: print("- Honest Splitting")
    dis, inf = honest_splitting(X, y, z, args.ratio_dis)
    X_dis, y_dis, z_dis = dis
    X_inf, y_inf, z_inf = inf

    # 1. Discovery
    if args.verbose: print("- Discovery Step:")

    # Esitimate ITE
    if args.verbose: print("    ITE Estimation")
    ite_dis = estimate_ite(X = X_dis, 
                           y = y_dis, 
                           z = z_dis,
                           method = args.method,
                           learner_y = args.learner_y,
                           learner_ps = args.learner_ps)

    # Rules Generation
    if args.verbose: print("    Rules Generation")
    rules = generate_rules(X = X_dis, 
                           ite = ite_dis,
                           n_trees = args.n_trees, 
                           max_depth = args.max_depth,
                           node_size = args.node_size,
                           max_rules = args.max_rules,
                           decimal = args.decimal,
                           criterion = args.criterion,
                           subsample = args.subsample)
    R_dis = get_rules_matrix(rules, X_dis)
    if args.verbose: print(f"      {R_dis.shape[1]} rules generated")

    # Rules Filtering
    if R_dis.shape[1]>0:
        if args.verbose: print("    Rules Filtering")
        R_dis = rules_filtering(R_dis,
                                t_ext = args.t_ext, 
                                t_corr = args.t_corr,)
        if args.verbose: print(f"      {R_dis.shape[1]} rules "
                               "filtered")

    # Rules Selection
    if R_dis.shape[1]>0:
        if args.verbose: print(f"    Rules Selection")
        rules = stability_selection(R_dis, ite_dis, 
                                    t_ss = args.t_ss, 
                                    B = args.B,
                                    subsample = args.subsample,
                                    alphas = args.alphas)
        if args.verbose: 
            if len(rules) == 0:
                print("      0 candidate rules selected (No HTE "
                    "discovered with stability selection threshold "
                    f"`t_ss`={args.t_ss})")
            else:
                print(f"      {len(rules)} candidate rules selected")
    

    # 2. Inference
    if args.verbose: print("- Inference Step:")
    # Esitimate ITE
    if args.verbose: print("    ITE Estimation")
    ite_inf = estimate_ite(X = X_inf, 
                           y = y_inf, 
                           z = z_inf,
                           method = args.method,
                           learner_y = args.learner_y,
                           learner_ps = args.learner_ps)
    
    if args.verbose: print("    AATE estimatation")
    R_inf = get_rules_matrix(rules, X_inf)
    model = estimate_aate(R_inf, 
                          ite_inf, 
                          B = args.B, 
                          subsample = args.subsample)

    if args.verbose: 
        temp = model.copy()
        temp.index = temp.index.str.replace("\(X\['|\)|'\]", "", 
                                            regex=True)
        print(temp)

    return model