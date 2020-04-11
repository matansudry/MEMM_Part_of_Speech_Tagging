import numpy as np
import scipy
from scipy.optimize import fmin_l_bfgs_b


class feature_statistics_class():
    def __init__(self, file, limit_common_words):
        self.TagsPerWord = {}
        self.WordCount = {}
        self.tags = ['*']
        self.words = []
        self.unseparated = []
        self.sentences_with_word_and_tag = []
        self.sentences_with_only_word = []
        self.sentences_with_only_tag = []
        self.f100_words_tags_count_dict = {}
        self.f102_prefix = {}
        self.f101_suffixes = {}
        self.f103_trigram = {}
        self.f104_bigram = {}
        self.f105_tag_common = {}
        self.f106_prev_word = {}
        self.f107_next_word = {}
        self.f108_numbers = {}
        self.f109_capital_letters = {}
        with open(file, 'r') as f:
            for line in f:
                sentence_with_word_and_tag = []
                sentence_with_only_word = []
                sentence_with_only_tag = []
                for wordtag in line.split():
                    sentence_with_word_and_tag.append(wordtag)
                    self.unseparated.append(wordtag)
                    word, tag = separate_tag_from_word(wordtag)
                    sentence_with_only_word.append(word)
                    sentence_with_only_tag.append(tag)
                    if word not in self.TagsPerWord:
                        self.TagsPerWord[word] = [tag]
                    elif tag not in self.TagsPerWord[word]:
                        self.TagsPerWord[word].append(tag)
                    if word not in self.WordCount:
                        self.WordCount[word] = 1
                    else:
                        self.WordCount[word] = self.WordCount[word] + 1
                    if tag not in self.tags:
                        self.tags.append(tag)
                    if word not in self.words:
                        self.words.append(word)
                self.sentences_with_word_and_tag.append(sentence_with_word_and_tag)
                self.sentences_with_only_word.append(sentence_with_only_word)
                self.sentences_with_only_tag.append(sentence_with_only_tag)
        self.CommonWords = list(common_words(self.WordCount, limit_common_words))
        self.tag_indices = {}
        ind = 0
        for t in self.tags:
            self.tag_indices[t] = ind
            ind = ind + 1
        self.features ={'self.f100_words_tags_count_dict'}

    def get_statistics(self):
        for sentence in self.sentences_with_word_and_tag:
            tag = None
            word = None
            next_word = None
            prev_word = None
            tag_2 = '*'
            tag_1 = '*'
            prev_word = None
            for word_with_tag in sentence:
                prev_word = word
                tag_2 = tag_1
                tag_1 = tag
                word, tag = separate_tag_from_word(word_with_tag)

                #word_tag F100
                if (word, tag) not in self.f100_words_tags_count_dict:
                    self.f100_words_tags_count_dict[(word, tag)] = 1
                else:
                    self.f100_words_tags_count_dict[(word, tag)] += 1

                #suffix F101
                suffix = get_suffix(word)
                if suffix != None:
                    if (suffix, tag) not in self.f101_suffixes:
                        self.f101_suffixes[(suffix, tag)] = 1
                    else:
                        self.f101_suffixes[(suffix, tag)] += 1

                #prefix F102
                prefix = get_prefix(word)
                if prefix != None:
                    if (prefix, tag) not in self.f102_prefix:
                        self.f102_prefix[(prefix, tag)] = 1
                    else:
                        self.f102_prefix[(prefix, tag)] += 1

                #trigram F103
                if tag_2 != None:
                    if (tag_2, tag_1, tag) not in self.f103_trigram:
                        self.f103_trigram[(tag_2, tag_1, tag)] = 1
                    else:
                        self.f103_trigram[(tag_2, tag_1, tag)] += 1

                #bigram F104
                if tag_1 != None:
                    if (tag_1, tag) not in self.f104_bigram:
                        self.f104_bigram[(tag_1, tag)] = 1
                    else:
                        self.f104_bigram[(tag_1, tag)] += 1
                #tag F105
                if tag not in self.f105_tag_common:
                    self.f105_tag_common[tag] = 1
                else:
                    self.f105_tag_common[tag] += 1

                #prev word F106
                if prev_word != None:
                    if (prev_word, tag) not in self.f106_prev_word:
                        self.f106_prev_word[(prev_word, tag)] = 1
                    else:
                        self.f106_prev_word[(prev_word, tag)] += 1

                #next word F107
                if tag_1 != None:
                    if (word, tag_1) not in self.f107_next_word:
                        self.f107_next_word[(word, tag_1)] = 1
                    else:
                        self.f107_next_word[(word, tag_1)] += 1

                # numbers F108
                if get_number(word) == True:
                    if (word, tag) not in self.f108_numbers:
                        self.f108_numbers[(word, tag)] = 1
                    else:
                        self.f108_numbers[(word, tag)] += 1

                #capital letters F109
                if get_capital_letter(word) == True:
                    if (word, tag) not in self.f109_capital_letters:
                        self.f109_capital_letters[(word, tag)] = 1
                    else:
                        self.f109_capital_letters[(word, tag)] += 1


class feature2id_class():
    def __init__(self, feature_statistics, threshold):
        self.feature_statistics = feature_statistics
        self.threshold = threshold
        self.n_total_features = 0
        self.n_tag_pairs = 0

        #features
        self.f100_words_tags_count_dict = {}
        self.f102_prefix = {}
        self.f101_suffixes = {}
        self.f103_trigram = {}
        self.f104_bigram = {}
        self.f105_tag_common = {}
        self.f106_prev_word = {}
        self.f107_next_word = {}
        self.f108_numbers = {}
        self.f109_capital_letters = {}

    def get_features(self):
        for sentence in self.feature_statistics.sentences_with_word_and_tag:
            tag = None
            word = None
            next_word = None
            prev_word = None
            tag_2 = '*'
            tag_1 = '*'
            prev_word = None
            for word_with_tag in sentence:
                prev_word = word
                tag_2 = tag_1
                tag_1 = tag
                word, tag = separate_tag_from_word(word_with_tag)

                #word_tag F100
                if (word, tag) not in self.f100_words_tags_count_dict and (self.feature_statistics.f100_words_tags_count_dict[(word, tag)] >= self.threshold):
                    self.f100_words_tags_count_dict[(word, tag)] = self.n_tag_pairs
                    self.n_tag_pairs += 1

                #suffix F101
                suffix = get_suffix(word)
                if suffix != None:
                    if (suffix, tag) not in self.f101_suffixes and (self.feature_statistics.f101_suffixes[(suffix, tag)] >= self.threshold):
                        self.f101_suffixes[(suffix, tag)] = self.n_tag_pairs
                        self.n_tag_pairs += 1

                #prefix F102
                prefix = get_prefix(word)
                if prefix != None:
                    if (prefix, tag) not in self.f102_prefix and (self.feature_statistics.f102_prefix[(prefix, tag)] >= self.threshold):
                        self.f102_prefix[(prefix, tag)] = self.n_tag_pairs
                        self.n_tag_pairs += 1

                #trigram F103
                if tag_2 != None:
                    if (tag_2, tag_1, tag) not in self.f103_trigram and (self.feature_statistics.f103_trigram[(tag_2, tag_1, tag)] >= self.threshold):
                        self.f103_trigram[(tag_2, tag_1, tag)] = self.n_tag_pairs
                        self.n_tag_pairs += 1

                #bigram F104
                if tag_1 != None:
                    if (tag_1, tag) not in self.f104_bigram and (self.feature_statistics.f104_bigram[(tag_1, tag)] >= self.threshold):
                        self.f104_bigram[(tag_1, tag)] = self.n_tag_pairs
                        self.n_tag_pairs += 1

                #tag F105
                if tag not in self.f105_tag_common and (self.feature_statistics.f105_tag_common[tag] >= self.threshold):
                    self.f105_tag_common[tag] = self.n_tag_pairs
                    self.n_tag_pairs += 1

                #prev word F106
                if prev_word != None:
                    if (prev_word, tag) not in self.f106_prev_word and (self.feature_statistics.f106_prev_word[(prev_word, tag)] >= self.threshold):
                        self.f106_prev_word[(prev_word, tag)] = self.n_tag_pairs
                        self.n_tag_pairs += 1

                #next word F107
                if tag_1 != None:
                    if (word, tag_1) not in self.f107_next_word and (self.feature_statistics.f107_next_word[(word, tag_1)] >= self.threshold):
                        self.f107_next_word[(word, tag_1)] = self.n_tag_pairs
                        self.n_tag_pairs += 1

                # numbers F108
                if get_number(word) == True:
                    if (word, tag) not in self.f108_numbers and (self.feature_statistics.f108_numbers[(word, tag)] >= self.threshold):
                        self.f108_numbers[(word, tag)] = self.n_tag_pairs
                        self.n_tag_pairs += 1

                #capital letters F109
                if get_capital_letter(word) == True:
                    if (word, tag) not in self.f109_capital_letters and (self.feature_statistics.f109_capital_letters[(word, tag)] >= self.threshold):
                        self.f109_capital_letters[(word, tag)] = self.n_tag_pairs
                        self.n_tag_pairs += 1

        self.n_total_features += self.n_tag_pairs


class Test_tagger():
    def __init__(self, file, limit_common_words):
        self.sentences_with_tag_and_word = []
        self.sentences_with_word_only = []
        self.sentences_with_tag_only = []
        with open(file, 'r') as f:
            for line in f:
                sentence_with_tag_and_word = []
                sentence_with_word_only = []
                sentence_with_tag_only = []
                for wordtag in line.split():
                    sentence_with_tag_and_word.append(wordtag)
                    word, tag = separate_tag_from_word(wordtag)
                    sentence_with_word_only.append(word)
                    sentence_with_tag_only.append(tag)
                self.sentences_with_tag_and_word.append(sentence_with_tag_and_word)
                self.sentences_with_word_only.append(sentence_with_word_only)
                self.sentences_with_tag_only.append(sentence_with_tag_only)


def separate_tag_from_word(word_with_tag):
    word = ''
    tag = ''
    count = 0
    while word_with_tag[count] != '_':
        word = word + word_with_tag[count]
        count = count + 1
    for i in range(count + 1, len(word_with_tag)):
        tag = tag + word_with_tag[i]
    return word, tag


def common_words(words_dict, limit):
    l = []
    for word in words_dict:
        if words_dict[word] > limit:
            l.append(word)
    return l


def get_prefix(word):
    word_len = len(word)
    if word_len < 3:
        return None
    else:
        return (word[0] + word[1] + word[2])


def get_suffix(word):
    word_len = len(word)
    if word_len < 3:
        return None
    else:
        matan = word[word_len - 3]
        return (word[word_len - 3] + word[word_len - 2] + word[word_len - 1])


def get_number(word):
    for i in word:
        if i.isdigit():
            return True
    return False


def get_capital_letter(word):
    if word[0].isupper():
        return True
    return False