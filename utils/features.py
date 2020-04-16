import numpy as np
import scipy
from scipy.optimize import fmin_l_bfgs_b
import time
import inspect

t2, t1, w, i, t = 'NN', 'VB', ['preprocessing' for _ in range(200)], 100, 'NN'


class FeatureVector:
    def __init__(self):
        self.counter = 0
        self.feats = []
        
    def __call__(self, t2, t1, w, i, t, fmt='list'):
        """
        args:
            * t2 - tag at pos i-2
            * t1 - tag at pos i-1
            * w - a list of the sentence words
            * i - current index
            * t - tag
            * (optional) fmt - return format 'vec' for a np.ndarray or 'list' of bool indecies
        """
        assert fmt in ('list', 'vec', 'both'), 'fmt must be list, vec or both'
        if fmt == 'list':
            feat_list = []
            for feat in self.feats:
                i = feat(t2, t1, w, i, t)
                if i is not None:
                    feat_list.append(i)
            return feat_list
        elif fmt == 'vec':
            feat_vec = np.zeros(self.counter)
            for feat in self.feats:
                i = feat(t2, t1, w, i, t)
                if i is not None:
                    feat_vec[i] = 1
            return feat_vec
        elif fmt == 'both':
            feat_list = []
            feat_vec = np.zeros(self.counter)
            for feat in self.feats:
                i = feat(t2, t1, w, i, t)
                if i is not None:
                    feat_vec[i] = 1
                    feat_list.append(i)
            return feat_vec, feat_list
        
    def get_calls(self):
        return [feat.calls for feat in self.feats]
        
    def invert_veat(self, i):
        for feat in self.feats:
            if i in feat.index_range:
                inv_dict = dict((v, k) for k, v in feat.hash_table.items())
                return feat, inv_dict[i]
        
    def add(self, foo, hash_keys, calls_counter=False):
        self.feats.append(FeatureGroup(foo, hash_keys, self.counter, calls_counter=calls_counter))
        self.counter += len(hash_keys)

    def __len__(self):
        return self.counter


class FeatureGroup:
    def __init__(self, foo, hash_keys, counter, calls_counter=False):
        self.foo = foo
        try:
            self.feat_key = inspect.getsource(foo).split(': ')[1].strip(' ')
        except:
            self.feat_key = str(foo)
        self.index_range = range(counter, counter + len(hash_keys))
        self.hash_table = dict(zip(hash_keys, list(self.index_range)))
        self.calls_counter = calls_counter
        if self.calls_counter:
            self.hash_calls = dict.fromkeys(hash_keys, 0)
            self.calls = 0

        # unit testing
        try:
            index = self.foo(t2, t1, w, i, t)
        except Exception as e:
            index = False
        assert index, 'failed unit testing'
        assert len(self.hash_table) > 0, 'self.hash_table is empty'

    def __call__(self, t2, t1, w, i, t):
        if self.calls_counter:
            self.calls += 1
            try:
                ans = self.hash_table[self.foo(t2, t1, w, i, t)]
                self.hash_calls[ans] += 1
                return ans
            except Exception as e:
                return None
        try:
            return self.hash_table[self.foo(t2, t1, w, i, t)]
        except Exception as e:
            pass

    def __len__(self):
        return len(self.hash_table)

    def __str__(self):
        return f'FeatureGroup({self.feat_key})'

    def __repr__(self):
        return f'FeatureGroup({self.feat_key})'


def create_feature_vector(dataset, group_thresholds=None, pruning=True, get_stats=False, assertions=False, calls_counter=False):
    feature_vector = FeatureVector()
    feature_groups_dicts = {key: dict() for key in group_thresholds.keys()}

    for t2, t1, w, i, t in dataset:
        for foo in feature_groups_dicts:
            try:
                key = foo(t2, t1, w, i, t)
                if key not in feature_groups_dicts[foo]:
                    feature_groups_dicts[foo][key] = 1
                else:
                    feature_groups_dicts[foo][key] += 1
            except:
                pass

    if pruning:
        for foo in group_thresholds:
            for key in list(feature_groups_dicts[foo].keys()):
                if group_thresholds[foo] is not None and feature_groups_dicts[foo][key] <= group_thresholds[foo]:
                    del feature_groups_dicts[foo][key]

    for foo in feature_groups_dicts:
        feature_vector.add(foo, feature_groups_dicts[foo], calls_counter=calls_counter)

    if assertions:
        index_list = []
        for feat in feature_vector.feats:
            index_list.extend(list(feat.hash_table.values()))

        assert sorted(index_list) == sorted(list(set(index_list)))
    
    if get_stats:
        return feature_vector, feature_groups_dicts
    else:
        return feature_vector    


