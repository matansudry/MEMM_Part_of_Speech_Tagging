import numpy as np
import scipy
from scipy.optimize import fmin_l_bfgs_b


t2, t1, w, i, t = 'NN', 'VB', ['preprocessing' for _ in range(200)], 100, 'NN'


def prefix(word, pre):
    return word[:pre]


def suffix(word, suf):
    return word[-suf:]


def call(func, *args):
    return func(*args)


class FeatureVector:
    # global attr, counts specific features in order to sync feature_vector indecies
    counter = 0

    def __init__(self, feats):
        if isinstance(feats, dict):
            self.feats = feats.values()
        else:
            self.feats = feats
            
        # unit testing
        try:
            for feat in self.feats:
                _ = feat(t2, t1, w, i, t)
            index = True
        except Exception as e:
            index = False
        assert index
        
    def __call__(self, t2, t1, w, i, t):
        vec = np.zeros(FeatureVector.counter)
        for feat in self.feats:
            index = feat(t2, t1, w, i, t)
            if index:
                vec[index] = 1
        return vec
        
    def add(self, feat):
        self.feats.append(feat)

    def __len__(self):
        return FeatureVector.counter


class FeatureGroup:
    def __init__(self, hash_rules, hash_set, *args, **kwargs):
        self.hash_rules = hash_rules
        self._hash_dict = dict(zip(hash_set, list(range(FeatureVector.counter, FeatureVector.counter + len(hash_set)))))
        FeatureVector.counter += len(self._hash_dict)
        self.kwargs = kwargs  # unused
        self.args = args  # unused

        # unit testing
        try:
            index = tuple([eval(val) for val in self.hash_rules])
        except Exception as e:
            index = False
        assert index

    @property
    def hash_dict(self):
        return self._hash_dict
    
    def __del__(self):
        FeatureVector.counter -= len(self._hash_dict)

    def __call__(self, t2, t1, w, i, t):
        try:
            return self._hash_dict(tuple([eval(val) for val in self.hash_rules]))
        except Exception as e:
            return None

    def __len__(self):
        return len(self._hash_dict)

    def __str__(self):
        return f'FeatureGroup({str(self.hash_rules)}, dict(...))'

    def __repr__(self):
        return f'FeatureGroup({str(self.hash_rules)}, dict(...))'

class Feature:
    def __init__(self, *args):
        self.rules = args
        self.index = FeatureVector.counter
        FeatureVector.counter += 1
        
        # unit testing
        try:
            for rule in self.rules:
                if not eval(rule):
                    pass
                else:
                    pass
            index = True
        except Exception as e:
            index = False
        assert index

    def __len__(self):
        return 1

    def __del__(self):
        FeatureVector.counter -= 1

    def __call__(self, t2, t1, w, i, t):
        for rule in self.rules:
            if not eval(rule):
                return None
        return self.index

    def __str__(self):
        return f'Feature({str(self.rules)})'

    def __repr__(self):
        return f'Feature({str(self.rules)})'











def f100(word_input):
    tag = word_input[1]
    index = word_input[0][3]
    word = word_input[0][2][index-1]
    if (word == 'base' and tag == 'VT'):
        return 1
    return 0


def f101(word_input):
    tag = word_input[1]
    index = word_input[0][3]
    word = word_input[0][2][index-1]
    word_len = len(word)
    #checking if the word is less than 3
    if word_len < 3:
        return 0
    if word[word_len-3] == 'i' and word[word_len-2] == 'n' and word[word_len-1] == 'g' and tag == 'VBG':
        return 1
    return 0


def f102(word_input):
    tag = word_input[1]
    index = word_input[0][3]
    word = word_input[0][2][index-1]
    word_len = len(word)
    #checking if the word is less than 3
    if word_len < 3:
        return 0
    if word[0] == 'p' and word[1] == 'r' and word[2] == 'e' and tag == 'NN':
        return 1
    return 0


def f103(word_input):
    tag = word_input[1]
    tag_1 = word_input[0][1]
    tag_2 = word_input[0][0]
    if tag_2 == 'DT' and tag_1 == 'JJ' and tag == 'Vt':
        return 1
    return 0


def f104(word_input):
    tag = word_input[1]
    tag_1 = word_input[0][1]
    if tag_1 == 'JJ' and tag == 'Vt':
        return 1
    return 0


def f105(word_input):
    tag = word_input[1]
    if tag == 'Vt':
        return 1
    return 0


def f106(word_input):
    tag = word_input[1]
    index = word_input[0][3]
    if index < 2:
        return 0
    word = word_input[0][2][index-2]
    if word == 'the' and tag == 'Vt':
        return 1
    return 0


def f107(word_input):
    tag = word_input[1]
    index = word_input[0][3]
    number_of_words = len(word_input[0][2])
    if index + 1 > number_of_words:
        return 0
    word = word_input[0][2][index]
    if word == 'the' and tag == 'Vt':
        return 1
    return 0


#need to verify
def f108_numbers(word_input):
    index = word_input[0][3]
    word = word_input[0][2][index - 1]
    for i in word:
        if i.isdigit():
            return 1
    return 0


#need to understand what is the t here, need to check on all letters and not onlt the 1st one
def f109_capital_letters(word_input):
    index = word_input[0][3]
    word = word_input[0][2][index - 1]
    is_capital_letters = word[0].isupper()
    if is_capital_letters:
        return 1
    return 0


def f110(word_input):
    index = word_input[0][3]
    word = word_input[0][2][index - 1]
    if word == '*':
        return 1
    return 0


def f111(word_input):
    index = word_input[0][3]
    word = word_input[0][2][index - 1]
    if word == '.':
        return 1
    return 0


def f112(word_input):
    index = word_input[0][3]
    word = word_input[0][2][index - 1]
    if word == '"':
        return 1
    return 0


def f113(word_input):
    index = word_input[0][3]
    word = word_input[0][2][index - 1]
    if word == '$':
        return 1
    return 0


def f114(word_input):
    index = word_input[0][3]
    word = word_input[0][2][index - 1]
    if word == ':':
        return 1
    return 0


def f115(word_input):
    index = word_input[0][3]
    word = word_input[0][2][index - 1]
    if word == '#':
        return 1
    return 0


def features_to_vector(word_input, number_of_features):
    features_vector = np.zeros(number_of_features)
    features_vector[0] = f100(word_input)
    features_vector[1] = f101(word_input)
    features_vector[2] = f102(word_input)
    features_vector[3] = f103(word_input)
    features_vector[4] = f104(word_input)
    features_vector[5] = f105(word_input)
    features_vector[6] = f106(word_input)
    features_vector[7] = f107(word_input)
    features_vector[8] = f108_numbers(word_input)
    features_vector[9] = f109_capital_letters(word_input)
    features_vector[10] = f110(word_input)
    features_vector[11] = f111(word_input)
    features_vector[12] = f112(word_input)
    features_vector[13] = f113(word_input)
    features_vector[14] = f114(word_input)
    features_vector[15] = f115(word_input)
    return features_vector

