import matplotlib.pyplot as plt
import seaborn as sns

import os

def plot_aate(model, args):
    
    """
    Plot the ATE and AATEs with error bars.
    
    Parameters
    ----------
    model: AATEs and ATE
    args: plot arguments
        save: save plot
        path: path to save plot
        exp_name: experiment name
    
    Returns
    -------
    plot of ATE and AATEs with error bars
    """

    model = model.copy()
    model.index = model.index.str.replace("\(X\['|\)|'\]", "", regex=True)
    ATE = model.iloc[0]
    AATE = model.iloc[1:]
    AATE = AATE.reindex(AATE["coef"].sort_values(ascending=True).index)
    M = AATE.shape[0]
    xrange = max(abs(AATE["coef"])+AATE["std err"])+0.25

    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.figure(figsize=(10, M*1.5))
    plt.errorbar(x=AATE["coef"], y=AATE.index, xerr=AATE["std err"], fmt='o')
    plt.ylabel("Rules")
    plt.xlabel("AATE")
    plt.title(f"ATE = {ATE['coef']:.2f} +/- {ATE['std err']:.2f}", fontsize=20)
    plt.axvline(x=0, color='black', linestyle='--')
    plt.tight_layout()
    plt.xlim(-xrange, xrange)
    plt.ylim(-0.5, len(AATE.index)-0.5)
    
    if args.save: 
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        plt.savefig(args.path+args.exp_name)
    else:
        plt.show()

    plt.close()