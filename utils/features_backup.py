import numpy as np
import scipy
from scipy.optimize import fmin_l_bfgs_b
import time


# t2, t1, w, i, t = 'NN', 'VB', ['preprocessing' for _ in range(200)], 100, 'NN'


def prefix(word, pre):
    return word[:pre]


def suffix(word, suf):
    return word[-suf:]


class FeatureVector:
    # global attr, counts specific features in order to sync feature_vector indecies
    # counter = 0

    def __init__(self):
        self.counter = 0
        self.feats = []
        self.time = 0.0
        self.calls = 0
        
    def __call__(self, t2, t1, w, i, t, format='vec'):
        """
        args:
            * t2 - tag at pos i-2
            * t1 - tag at pos i-1
            * w - a list of the sentence words
            * i - current index
            * t - tag
            * (optional) format - return format 'vec' for a np.ndarray or 'list' of bool indecies
        """
        tic = time.time()
        return_list = []
        vec = np.zeros(FeatureVector.counter)
        for feat in self.feats:
            t2, t1, w, i, t = 'NN', 'VB', ['preprocessing' for _ in range(200)], 100, 'NN'
            index = feat(t2, t1, w, i, t)
            if index:
                vec[index] = 1
                return_list.append(index)

        self.time += float(time.time() - tic)
        self.calls += 1

        if format == 'list':
            return return_list
        else:
            return vec
        
    def time(self):
        return {'avg': self.time/self.calls, 'total': self.time, 'calls': self.calls}
        
    def add_FeatureGroup(self, hash_rules, hash_set):
        t2, t1, w, i, t = 'NN', 'VB', ['preprocessing' for _ in range(200)], 100, 'NN'
        self.feats.append(FeatureGroup(hash_rules, hash_set, self.counter))
        self.counter += len(hash_set)
        try:
            for feat in self.feats:
                _ = feat(t2, t1, w, i, t)
            index = True
        except Exception as e:
            index = False
        assert index

    def add_Feature(self, *args):
        t2, t1, w, i, t = 'NN', 'VB', ['preprocessing' for _ in range(200)], 100, 'NN'
        self.feats.append(Feature(self.counter, *args))
        self.counter += 1

        try:
            for feat in self.feats:
                _ = feat(t2, t1, w, i, t)
            index = True
        except Exception as e:
            index = False
        assert index

    def __len__(self):
        return self.counter


class FeatureGroup:
    def __init__(self, hash_rules, hash_set, counter):
        self.hash_rules = hash_rules
        self.index_range = range(counter, counter + len(hash_set))
        self.hash_dict = dict(zip(hash_set, list(self.index_range)))
        self.time = 0.0
        self.calls = 0

        # unit testing
        try:
            t2, t1, w, i, t = 'NN', 'VB', ['preprocessing' for _ in range(200)], 100, 'NN'
            index = tuple([eval(val, locals()) for val in self.hash_rules])
        except Exception as e:
            index = False
        assert index, 'failed unit testing'
        assert len(self.hash_dict) > 0, 'self.hash_dict is empty'

    def __call__(self, t2, t1, w, i, t):
        tic = time.time()
        try:
            ans = self._hash_dict(tuple([eval(val) for val in self.hash_rules]))
        except Exception as e:
            ans =  None
        self.time += float(time.time() - tic)
        self.calls += 1
        return ans

    def __len__(self):
        return len(self.hash_dict)

    def __str__(self):
        return f'FeatureGroup({str(self.hash_rules)}, dict(...))'

    def __repr__(self):
        return f'FeatureGroup({str(self.hash_rules)}, dict(...))'

class Feature:
    def __init__(self, counter, *args):
        self.rules = args
        self.index = counter
        self.time = 0.0
        self.calls = 0

        # unit testing
        try:
            t2, t1, w, i, t = 'NN', 'VB', ['preprocessing' for _ in range(200)], 100, 'NN'
            for rule in self.rules:
                if not eval(rule, locals()):
                    pass
                else:
                    pass
            index = True
        except Exception as e:
            index = False
        assert index, 'failed unit testing'

    def __len__(self):
        return 1

    def __call__(self, t2, t1, w, i, t):
        tic = time.time()
        for rule in self.rules:
            if not eval(rule, locals()):
                return None
        self.time += float(time.time() - tic)
        self.calls += 1
        return self.index

    def __str__(self):
        return f'Feature({str(self.rules)})'

    def __repr__(self):
        return f'Feature({str(self.rules)})'


# def create_features(file_name, feature_threshold_dict):
    








# def f100(word_input):
    # tag = word_input[1]
    # index = word_input[0][3]
    # word = word_input[0][2][index-1]
    # if (word == 'base' and tag == 'VT'):
        # return 1
    # return 0


# def f101(word_input):
    # tag = word_input[1]
    # index = word_input[0][3]
    # word = word_input[0][2][index-1]
    # word_len = len(word)
    # #checking if the word is less than 3
    # if word_len < 3:
        # return 0
    # if word[word_len-3] == 'i' and word[word_len-2] == 'n' and word[word_len-1] == 'g' and tag == 'VBG':
        # return 1
    # return 0


# def f102(word_input):
    # tag = word_input[1]
    # index = word_input[0][3]
    # word = word_input[0][2][index-1]
    # word_len = len(word)
    # #checking if the word is less than 3
    # if word_len < 3:
        # return 0
    # if word[0] == 'p' and word[1] == 'r' and word[2] == 'e' and tag == 'NN':
        # return 1
    # return 0


# def f103(word_input):
    # tag = word_input[1]
    # tag_1 = word_input[0][1]
    # tag_2 = word_input[0][0]
    # if tag_2 == 'DT' and tag_1 == 'JJ' and tag == 'Vt':
        # return 1
    # return 0


# def f104(word_input):
    # tag = word_input[1]
    # tag_1 = word_input[0][1]
    # if tag_1 == 'JJ' and tag == 'Vt':
        # return 1
    # return 0


# def f105(word_input):
    # tag = word_input[1]
    # if tag == 'Vt':
        # return 1
    # return 0


# def f106(word_input):
    # tag = word_input[1]
    # index = word_input[0][3]
    # if index < 2:
        # return 0
    # word = word_input[0][2][index-2]
    # if word == 'the' and tag == 'Vt':
        # return 1
    # return 0


# def f107(word_input):
    # tag = word_input[1]
    # index = word_input[0][3]
    # number_of_words = len(word_input[0][2])
    # if index + 1 > number_of_words:
        # return 0
    # word = word_input[0][2][index]
    # if word == 'the' and tag == 'Vt':
        # return 1
    # return 0


# #need to verify
# def f108_numbers(word_input):
    # index = word_input[0][3]
    # word = word_input[0][2][index - 1]
    # for i in word:
        # if i.isdigit():
            # return 1
    # return 0


# #need to understand what is the t here, need to check on all letters and not onlt the 1st one
# def f109_capital_letters(word_input):
    # index = word_input[0][3]
    # word = word_input[0][2][index - 1]
    # is_capital_letters = word[0].isupper()
    # if is_capital_letters:
        # return 1
    # return 0


# def f110(word_input):
    # index = word_input[0][3]
    # word = word_input[0][2][index - 1]
    # if word == '*':
        # return 1
    # return 0


# def f111(word_input):
    # index = word_input[0][3]
    # word = word_input[0][2][index - 1]
    # if word == '.':
        # return 1
    # return 0


# def f112(word_input):
    # index = word_input[0][3]
    # word = word_input[0][2][index - 1]
    # if word == '"':
        # return 1
    # return 0


# def f113(word_input):
    # index = word_input[0][3]
    # word = word_input[0][2][index - 1]
    # if word == '$':
        # return 1
    # return 0


# def f114(word_input):
    # index = word_input[0][3]
    # word = word_input[0][2][index - 1]
    # if word == ':':
        # return 1
    # return 0


# def f115(word_input):
    # index = word_input[0][3]
    # word = word_input[0][2][index - 1]
    # if word == '#':
        # return 1
    # return 0


# def features_to_vector(word_input, number_of_features):
    # features_vector = np.zeros(number_of_features)
    # features_vector[0] = f100(word_input)
    # features_vector[1] = f101(word_input)
    # features_vector[2] = f102(word_input)
    # features_vector[3] = f103(word_input)
    # features_vector[4] = f104(word_input)
    # features_vector[5] = f105(word_input)
    # features_vector[6] = f106(word_input)
    # features_vector[7] = f107(word_input)
    # features_vector[8] = f108_numbers(word_input)
    # features_vector[9] = f109_capital_letters(word_input)
    # features_vector[10] = f110(word_input)
    # features_vector[11] = f111(word_input)
    # features_vector[12] = f112(word_input)
    # features_vector[13] = f113(word_input)
    # features_vector[14] = f114(word_input)
    # features_vector[15] = f115(word_input)
    # return features_vector

