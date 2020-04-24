from sklearn.metrics import mean_squared_error
import pandas as pd
import sklearn
from sklearn import linear_model


class Node:
    def __init__(self, learner, samples, labels, depth, split_by_feature=None,
                 threshold=None, children=[], nominal_value=None):
        self.samples = samples
        self.labels = labels
        self.children = children
        self.split_by_feature = split_by_feature
        self.threshold = threshold
        self.nominal_value = nominal_value
        self.depth = depth
        self.model = getattr(sklearn.linear_model, learner)().fit(pd.get_dummies(samples), labels)
        self.loss = mean_squared_error(self.model.predict(pd.get_dummies(samples)), labels) * len(samples)
