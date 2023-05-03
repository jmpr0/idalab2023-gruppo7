from sklearn.ensemble import RandomForestClassifier
from argparse import ArgumentParser

from .classifier import *


class RandomForest(Classifier):
    def __init__(self, n_jobs=-1, verbose=1, n_estimators=100, max_depth=None):
        self.model = RandomForestClassifier(
            n_jobs=n_jobs, verbose=verbose, n_estimators=n_estimators, max_depth=max_depth)

    def compile(self, **kwargs):
        pass

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)

    def predict(self, X, **kwargs):
        pass

    def _score(self, X, y):
        pass

    def print_summary(self):
        print(self.model.get_params(deep=False))
            
    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the model specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--n-estimators', default=100, type=int, required=False,
                            help='The number of trees in the forest (default=%(default)s)')
        parser.add_argument('--max-depth', default=None, type=int, required=False,
                            help='The maximum depth of the tree (default=%(default)s)')
        return parser.parse_known_args(args)
