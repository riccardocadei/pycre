from dataset import honest_splitting
from ite import estimate_ite
from decision_rules import generate_rules, get_rules_matrix, rules_filtering, stability_selection
from aate import estimate_aate

import warnings
warnings.filterwarnings("ignore")

def train(dataset, args):
    """
    Fit CRE model
    Input
        dataset: pd.DataFrame with with Covariates ('name1', ..., 'namek'), 
                 Treatments ('z') and Outcome ('y')
        args: arguments from parser
    Output
        model: pd.DataFrame with ATE and AATE estimates and
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
    model = estimate_aate(R_inf, ite_inf)

    if args.verbose: 
        temp = model.copy()
        temp.index = temp.index.str.replace("\(X\['|\)|'\]", "", regex=True)
        print(temp)

    return model