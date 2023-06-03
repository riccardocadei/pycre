from sklearn.linear_model import LassoCV
from sklearn.utils import resample
from sklearn.feature_selection import SelectFromModel

import numpy as np

def stability_selection(R, ite, 
                        t_ss = 0.6,
                        alpha = [0.1, 0.5, 1.0], 
                        B = 50, 
                        subsample_ratio = 0.7,
                        folds = 5):
    M = R.shape[1]
    ite = ite-np.mean(ite)
    stability_scores = np.zeros(M)
    for _ in range(B):
        X, y = resample(R, ite, replace=False, n_samples=int(len(R) * subsample_ratio))
        lasso = LassoCV(alphas=alpha, cv=folds).fit(X, y)
        non_zero_indices = np.where(lasso.coef_ != 0)[0]
        stability_scores[non_zero_indices] += 1
    stability_scores /= B

    rules = list(R.columns[stability_scores > t_ss])
    return rules