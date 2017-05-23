# -*- coding: utf-8 -*-

import numpy as np

class FeatureNames():
    def __init__(self, feature_names=None, bias_name=None, n=None):
        self.feature_names = feature_names
        self.bias_name = bias_name
        self.n_features = n or len(feature_names)

    def __repr__(self):
        return '<FeatureNames: {} feature {} bias>'.format(
            self.n_features, 'with' if self.has_bias else 'without')

    def __len__(self):
        return self.n_features + int(self.has_bias)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._slice(idx)
        if isinstance(idx, np.ndarray):
            return [self[i] for i in idx]
        if self.has_bias and idx == self.bias_idx:
            return self.bias_name
        if 0 <= idx < self.n_features:
            return self.feature_names[idx]

    @property
    def has_bias(self):
        return self.bias_name is not None

    @property
    def bias_idx(self):
        if self.has_bias:
            return self.n_features


class FeatureImportances(object):
    def __init__(self, importances, remaining):
        self.importances = importances
        self.remaining = remaining

    @classmethod
    def from_names_values(cls, names, values, **kwargs):
        params = zip(names, values)
        importance = [FeatureWeight(*x) for x in params]
        return cls(importance, **kwargs)


class FeatureWeight(object):
    def __init__(self, feature, weight, value=None):
        self.feature = feature
        self.weight = weight
        self.value = value