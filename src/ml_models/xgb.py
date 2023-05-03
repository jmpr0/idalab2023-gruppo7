from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from argparse import ArgumentParser

from .classifier import *


class XGB(Classifier):
    def __init__(self, use_label_encoder=False, n_estimators=100, max_depth=3, eval_metric='mlogloss'):
        self.model = XGBClassifier(
            use_label_encoder=use_label_encoder, n_estimators=n_estimators, max_depth=max_depth, eval_metric=eval_metric)

    def compile(self, **kwargs):
        pass

    def fit(self, X, y, **kwargs):
        self.model.fit(X, LabelEncoder().fit_transform(y))

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
                            help='Number of boosting rounds (default=%(default)s)')
        parser.add_argument('--max-depth', default=3, type=int, required=False,
                            help='The maximum depth of the tree (default=%(default)s)')
        parser.add_argument('--eval-metric', default='mlogloss', type=str, required=False,
                            help='Metric used for monitoring the training result and early stopping' \
                            '(default=%(default)s)')
        return parser.parse_known_args(args)
