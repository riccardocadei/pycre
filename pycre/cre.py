from parsers import get_parser
from arguments import check_args
from dataset import dataset_generator
from plot import plot_aate
from training import train
from predict import predict
import traceback

import numpy as np

class CRE:
    """
    Causal Rule Ensemble
    
    Provides a new method for interpretable heterogeneous 
    treatment effects characterization in terms of decision rules 
    via an extensive exploration of heterogeneity patterns by an 
    ensemble-of-trees approach, enforcing high stability in the 
    discovery. It relies on a two-stage pseudo-outcome regression, and 
    theoretical convergence guarantees support it.
    
    Parameters
    ----------
    **kwargs
        Arguments for CRE

        ratio_dis : float, default=0.5
            Ratio of the observations used for discovery.
        n_trees : int, default=20
            Number of decision trees for rules discovery.
        max_depth : int, default=2
            Max depth of decision trees for rules discovery (i.e., max 
            decision rules depth).
        node_size : int, default=20
            Min number of observations in a leaf node for rules 
            discovery.
        max_rules : int, default=50
            Max number of generated candidate decision rules.
        criterion : str, default="het"
            Criterion for splitting decision trees for rules 
            discovery.
        decimal : int, default=2
            Number of digits to round the rules' thresholds.
        t_ext : float, default=0.02
            Threshold to discard too generic or too specific 
            (extreme).
        t_corr : float, default=0.5
            Threshold to discard too correlated rules.
        t_ss : float, default=0.8
            Threshold for stability selection in rules selection.
        alphas : list, default=[0.01,0.1, 1]
            Alpha values for stability selection in rules selection.
        B : int, default=20
            Number of bootstrap samples for stability selection in 
            rules selection and uncertainty quantification in 
            estimation.
        subsample : float, default=0.5
            Bootstrap ratio subsample for forest generation and 
            stability selection in rules selection, and uncertainty 
            quantification in estimation.
        learner_y : sklearn learner, default=GradientBoostingClassifier()
            model for outcome estimation, 
        learner_ps : sklearn learner, default=GradientBoostingClassifier()
            model for propensity score estimation
        method : str, default="tlearner"
            Pseudo-Outcome (ITE) estimator.
        verbose : bool, default=True
            Verbose.
        seed: int, default=1
            Seed.
        path: str, default="results/"
            Path to save results.
        exp_name: str, default="example1"
            Experiment name.
        save: bool, default=False   
            Save results.
         
    Attributes
    ----------
    args : argparse.Namespace
        Arguments for CRE.
    model : pd.DataFrame
        Model with ATE and AATE estimates and confidence intervals.
    rules : list
        List of decision rules.
      
    Methods
    -------
    fit(X, y, z, W=None)
        Fit CRE model
    eval(X)
        Evaluate CRE model
    visualize()
        Visualize CRE model
    
    References
    ----------
    [1] Bargagli-Stoffi, F. J., Cadei, R., Lee, K., & Dominici, F. 
    (2023) "Causal rule ensemble: Interpretable Discovery and 
    Inference of Heterogeneous Treatment Effects" arXiv preprint 
    <arXiv:2009.09036>

    Examples
    --------
    >>> from cre import CRE
    >>> from dataset import dataset_generator
    >>> X, y, z, _ = dataset_generator()
    >>> model = CRE()
    >>> model.fit(X, y, z)
    >>> ite_pred = model.eval(X)
    >>> model.visualize()
    """

    def __init__(self, **kwargs):
        if 'args' in kwargs:
            self.args = kwargs['args']
        else:
            self.args = get_parser().parse_args(args=[])
        for kwarg, value in kwargs.items():
            setattr(self.args, kwarg, value)

        check_args(self.args)
        np.random.seed(self.args.seed)

    def fit(self, X, y, z, W = None):
        try:
            self.model = train(X, y, z, self.args, W)
            self.rules = list(self.model.index)[1:]
        except:
            traceback.print_exc()
            self.rules = []

    def eval(self, X):
        return predict(X, self.model)
    
    def visualize(self):  
        if len(self.rules) == 0: 
            raise ValueError("No rules to plot")
        else: 
            plot_aate(self.model, self.args)


def main(args):
    
    X, y, z, _ = dataset_generator()
    
    model = CRE(args = args)
    model.fit(X, y, z)
    ite_pred = model.eval(X)

    return ite_pred

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)