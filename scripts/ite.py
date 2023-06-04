#from econml.dml import CausalForestDML
#from causallib.estimation import IPW 
from sklearn.linear_model import LogisticRegression

from utils import clip

# Q: Can map outside [-1,1] even if output is binary
def estimate_ite_ipw(X, y, z,
                    learner=LogisticRegression(), clip_tr=0.01):
    """
    Estimate ITE using Inverse Propensity Weighting

    Input
        X: pd.DataFrame with Covariates ('X')
        y: pd.Series with Outcome ('y')
        z: pd.Series with Treatment ('z')
        learner: sklearn learner for propensity score estimation
        clip_tr: threshold for clipping propensity scores
   
   Output
        ite: pd.Series with ITE estimates (N)
    """
    
    ps = estimate_ps(X = X,
                     z = z,
                     learner = learner)
    ps = clip(ps, clip_tr)
    ite = ((z/ps)-(1-z)/(1-ps))*y
    return ite

def estimate_ps(X, z, learner=LogisticRegression()):
    """
    Estimate Propensity Score

    Input
        X: pd.DataFrame with Covariates ('X')
        z: pd.Series with Treatment ('z')
        learner: sklearn learner for propensity score estimation
    
    Output
        ps: pd.Series with Propensity Score estimates (N)
    """

    learner.fit(X, z)
    return learner.predict_proba(X)[:,1]