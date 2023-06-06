import matplotlib.pyplot as plt
import seaborn as sns

import os

def plot(result, 
         xrange = 2,
         save = False,
         path = "../results/",
         exp_name = "example.png"):
    
    """
    Plot the ATE and AATEs with error bars.
    
    Input:
        result: AATEs and ATE
        xrange: x-axis range
        save: save plot
        path: path to save plot
        exp_name: experiment name
    
    Output:
        plot
    """

    result.index = result.index.str.replace("\(X\['|\)|'\]", "", regex=True)
    ATE = result.iloc[0]
    AATE = result.iloc[1:]
    AATE = AATE.reindex(AATE["coef"].sort_values(ascending=True).index)
    M = AATE.shape[0]

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
    
    if save: 
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path+exp_name)
    else:
        plt.show()

    plt.close()