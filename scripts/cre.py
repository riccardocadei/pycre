from parsers import get_parser
from dataset import dataset_generator
from plot import plot_aate
from training import train
from predict import predict


import numpy as np

class CRE:
    def __init__(self, args):
        self.args = args
        np.random.seed(args.seed)

    def fit(self, X, y, z):
        try:
            self.model = train(X, y, z, self.args)
            self.rules = list(self.model.index)[1:]
        except:
            self.rules = []

    def eval(self, X):
        return predict(X, self.model)
    
    def plot(self):
        if len(self.rules) == 0: 
            raise ValueError("No rules to plot")
        else: 
            plot_aate(self.model, self.args)


def main(args):
    
    X, y, z, ite = dataset_generator()
    
    model = CRE(args)
    model.fit(X, y, z)
    ite = model.eval(X)

    return ite

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)