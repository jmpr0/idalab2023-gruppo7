class Classifier():
    def __init__(self):
        self.model = None

    def compile(self, **kwargs):
        self.model.compile(**kwargs)

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.predict_proba(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        return self.model.predict_proba(X)

    def score(self, X, **kwargs):
        return self.predict_proba(X, **kwargs)

    def print_summary(self):
        print(self.model.summary())
