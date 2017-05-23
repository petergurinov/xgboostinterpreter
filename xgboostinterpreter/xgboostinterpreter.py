# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import re
import numpy as np
import scipy.sparse as spr

from functools import partial
from prettytable import PrettyTable
from classes import FeatureNames, FeatureImportances, FeatureWeight
from xgboost import XGBClassifier, XGBRegressor, DMatrix


def weight(model, vec=None, count=10):
    feature_names = FeatureNames(vec.get_feature_names(), 'BIAS')

    coefficients = _feature_important(model)

    indices = _sort_largest_positive(coefficients, count)
    names, val = feature_names[indices], coefficients[indices]

    feature_importance = FeatureImportances.from_names_values(
        names, val,
        remaining=np.count_nonzero(coefficients) - len(indices),
    )

    table = PrettyTable(['Weight', 'Feature', 'Value'])

    table.align['Weight'] = "r"
    table.align['Feature'] = "l"
    table.align['Value'] = "r"
    table.float_format = 5

    for fi in feature_importance.importances:
        f = (fi.feature).split('=')
        f_name = f[0]
        value = f[1] if f[1] else " "
        table.add_row([fi.weight, f_name, value])

    print(table)


def predict(model, X, vec=None, count=None):
    feature_names = FeatureNames(vec.get_feature_names(), bias_name='<BIAS>')

    X = vec.transform([X])
    if spr.issparse(X):
        X = X.tocsc()

    ans = model.predict(X)
    xgb_feature_names = model.booster().feature_names
    scores_weights = _feature_weights(model, X, feature_names, xgb_feature_names)

    is_multiclass = _is_multiclass(model)
    is_regr = isinstance(model, XGBRegressor)
    probability = None if is_regr else model.predict_proba(X)

    x, = spr.hstack([X, np.ones((X.shape[0], 1))]).tocsr()
    if spr.issparse(x):
        assert x.shape[0] == 1
    if spr.issparse(x) and model.missing != 0:
        val_coo = x.tocoo()
        val = x.toarray()[0]
        missing_mask = val == 0
        missing_mask[val_coo.col] = False
        val[missing_mask] = np.nan
    elif is_spr(x):
        val = x.toarray()[0]
    else:
        val = x.copy()
    if not np.isnan(model.missing):
        val[val == model.missing] = np.nan
    x = val

    def get_score_feature_weights(_label_id):
        _score, _feature_weights = scores_weights[_label_id]
        return _score, _get_features(feature_names, _feature_weights, count, x=x)

    if is_multiclass:
        print("While not support multiclass prediction interpreter!!!")
        return
    else:
        score, fw = get_score_feature_weights(0)

    print("Answer: ", ans[0]),
    if probability != None:
        print("with probability: ", probability[0][0])
    table = PrettyTable(['Contribution', 'Feature', 'Value'])
    table.align['Contribution'] = "r"
    table.align['Feature'] = "l"
    table.align['Value'] = "r"
    table.float_format = 5

    for fn in fw:
        f = (fn.feature).split('=')
        f_name = f[0]
        value = f[1] if len(f) > 1 else " "
        table.add_row([fn.weight, f_name, value])

    print(table)


# Function for weight

def _feature_important(model):
    b = model.booster()
    ftrs = b.get_score(importance_type='gain')
    all_ftrs = np.array(
        [ftrs.get(f, 0.) for f in b.feature_names], dtype=np.float32
    )
    return all_ftrs / all_ftrs.sum()


def _sort_largest_positive(x, c):
    num_positive = (x > 0).sum()
    c = num_positive if c is None else min(num_positive, c)
    return _sort_largest(x, c)


# Function for predict


def _count_targets(model):
    if isinstance(model, XGBClassifier):
        return 1 if model.n_classes_ == 2 else model.n_classes_
    elif isinstance(model, XGBRegressor):
        return 1


def _is_multiclass(model):
    if isinstance(model, XGBClassifier):
        return False if _count_targets(model) == 1 else True
    elif isinstance(model, XGBRegressor):
        return False


def _sort_largest(x, c):
    indices = np.argpartition(x, -c)[-c:]
    values = x[indices]
    return indices[np.argsort(-values)]


def _feature_weights(model, X, feature_names, xgb_feature_names):
    b = model.booster()
    leaf_id, = b.predict(DMatrix(X, missing=model.missing), pred_leaf=True)
    xgb_feature_names = {f: i for i, f in enumerate(xgb_feature_names)}
    tree_dump = b.get_dump(with_stats=True)

    target_feature_weight = partial(_target_feature_weights, feature_names=feature_names,
                                    xgb_feature_names=xgb_feature_names)
    n_targets = _count_targets(model)
    if n_targets > 1:
        scores_weights = [
            target_feature_weight(
                leaf_id[target_idx::n_targets],
                tree_dump[target_idx::n_targets],
            ) for target_idx in range(n_targets)]
    else:
        scores_weights = [target_feature_weight(leaf_id, tree_dump)]
    return scores_weights


def _target_feature_weights(leaf_ids, tree_dumps, feature_names,
                            xgb_feature_names):
    feature_weights = np.zeros(len(feature_names))
    score = 0
    for text_dump, leaf_id in zip(tree_dumps, leaf_ids):
        leaf = _indexed_leafs(_parse_tree(text_dump))[leaf_id]
        score += leaf['leaf']
        path = [leaf]
        while 'parent' in path[-1]:
            path.append(path[-1]['parent'])
        path.reverse()
        for node, child in zip(path, path[1:]):
            idx = xgb_feature_names[node['split']]
            feature_weights[idx] += child['leaf'] - node['leaf']
        feature_weights[feature_names.bias_idx] += path[0]['leaf']
    return score, feature_weights


def _indexed_leafs(parent):
    if not parent.get('children'):
        return {parent['nodeid']: parent}
    indexed = {}
    for child in parent['children']:
        child['parent'] = parent
        if 'leaf' in child:
            indexed[child['nodeid']] = child
        else:
            indexed.update(_indexed_leafs(child))
    parent['leaf'] = _parent_value(parent['children'])
    return indexed


def _parent_value(children):
    covers = np.array([child['cover'] for child in children])
    covers /= np.sum(covers)
    leafs = np.array([child['leaf'] for child in children])
    return np.sum(leafs * covers)


def _parse_tree(text_dump):
    result = None
    stack = []
    for line in text_dump.split('\n'):
        if line:
            depth, node = _parse_line(line)
            if depth == 0:
                assert not stack
                result = node
                stack.append(node)
            else:
                if depth < len(stack):
                    stack = stack[:depth]
                stack[-1].setdefault('children', []).append(node)
                stack.append(node)
    return result


def _parse_line(line):
    branch_match = re.match('^(\t*)(\d+):\[(\w+)<([^\]]+)\] yes=(\d+),no=(\d+),missing=(\d+),'
                            'gain=([^,]+),cover=(.+)$', line)
    if branch_match:
        tabs, node_id, feature, condition, yes, no, missing, gain, cover = branch_match.groups()
        depth = len(tabs)
        return depth, {'depth': depth, 'nodeid': int(node_id), 'split': feature,
            'split_condition': float(condition), 'yes': int(yes), 'no': int(no),
            'missing': int(missing), 'gain': float(gain), 'cover': float(cover)}
    leaf_match = re.match('^(\t*)(\d+):leaf=([^,]+),cover=(.+)$', line)
    if leaf_match:
        tabs, node_id, value, cover = leaf_match.groups()
        depth = len(tabs)
        return depth, {'nodeid': int(node_id), 'leaf': float(value), 'cover': float(cover)}


def _get_features(feature_names, coef, count, x=None):
    if count is None:
        count = (coef > 0).sum()
    cnt = min((np.abs(coef) > 0).sum(), count)
    ind = np.argpartition(coef, -cnt)[-cnt:]
    values = coef[ind]
    indices = ind[np.argsort(-values)]
    _feature = _features(indices, feature_names, coef, x)
    feature_weights = [fw for fw in _feature]
    return feature_weights


def _features(indices, feature_names, coef, x):
    names = mask(feature_names, indices)
    weights = mask(coef, indices)
    values = mask(x, indices)
    return [FeatureWeight(name, weight, value=value)
            for name, weight, value in zip(names, weights, values)]


def _get_value_indices(names1, names2, lookups):
    positions = {name: idx for idx, name in enumerate(names2)}
    positions.update({name: idx for idx, name in enumerate(names1)})
    return [positions[name] for name in lookups]


def mask(x, indices):
    indices_shape = (
        [len(indices)] if isinstance(indices, list) else indices.shape)
    if not indices_shape[0]:
        return np.array([])
    elif is_spr(x) and len(indices_shape) == 1:
        return x[0, indices].toarray()[0]
    else:
        return x[indices]


def is_spr(x):
    return spr.issparse(x) and len(x.shape) == 2 and x.shape[0] == 1