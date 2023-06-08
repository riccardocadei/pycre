from parser import get_parser
from dataset import get_dataset
from plot import plot_aate
from training import train
from predict import predict


import numpy as np

class CRE:
    def __init__(self, args):
        self.args = args
        np.random.seed(args.seed)

    def fit(self, dataset):
        self.dataset = dataset
        self.model = train(dataset, self.args)
        self.rules = list(self.model.index)[1:]

    def plot(self):
        plot_aate(self.model, self.args)

    def eval(self, X):
        self.ite = predict(X, self.model)
        return self.ite

def main(args):
    if args.verbose: print(f"Load {args.dataset_name} dataset")
    dataset = get_dataset(args)
    
    if args.verbose: print("Run CRE algorithm")
    model = CRE(args)
    model.fit(dataset)
    ite = model.eval(dataset)

    return ite

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)