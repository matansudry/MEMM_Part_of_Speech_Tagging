import numpy as np
import scipy
from scipy.optimize import fmin_l_bfgs_b
import random


class Dataset():
    def __init__(self, file_name, labeled=True, tags=None):
        """
        args:
            * file_name
            * labeled - is the dataset labeled?
            * tags - if dataset is not labeled, what are the tags
        """
        self.words_counter = 0
        self.sentences = []
        self.labeled = labeled
        self.tags = set()
        if not labeled and tags:
            self.tags = set(tags)
        with open(file_name, 'r') as f:
            for line in f.readlines():
                words = []
                tags = []
                for word_tag in line.split():
                    if self.labeled:
                        word, tag = word_tag.split('_')
                        self.tags.add(tag)
                    else:
                        word = word_tag
                        tag = None
                    tags.append(tag)
                    words.append(word)
                    self.words_counter += 1
                    
                self.sentences.append((words, tags))
        self.batch_loader = self.sentences.copy()
        self.batch_size = None
        self.shuffled = False
        
    def _init_loader(self, batch_size, shuffle, seed):
        self.batch_loader = self.sentences.copy()
        if shuffle:
            random.shuffle(self.batch_loader)
        self.shuffled = shuffle
        self.batch_size = batch_size

    def load_batch(self, batch_size=None, shuffle=False, seed=42):
        """
        args:
            * batch_size=None - batch_size to load, if batch_size=None -> batch_size=len(self.sentences)
            * shuffle - reshuffle loaded sentences
            * seed - set random.seed
        return:
            * generator that iterates batch_size of sentences and words yields tuples of t2, t1, w, i, t
        """
        if not batch_size:
            batch_size = len(self.sentences)
        if not self.batch_loader or self.batch_size != batch_size or self.shuffled != shuffle:
            self._init_loader(batch_size, shuffle, seed)
        sentences = self.batch_loader[:self.batch_size]
        del self.batch_loader[:self.batch_size]
        for w, tags in sentences:
            t1, t = '*', '*'
            for i in range(len(w)):
                t2, t1, t = t1, t, tags[i]
                yield t2, t1, w, i, t

    def __len__(self):
        return self.words_counter

    def __iter__(self):
        for w, tags in self.sentences:
            t1, t = '*', '*'
            for i in range(len(w)):
                t2, t1, t = t1, t, tags[i]
                yield t2, t1, w, i, t

# class feature_statistics_class():
    # def __init__(self, file_name, limit_common_words=5, lamda=0.1):
        # self.tags_per_word = {}
        # self.words_per_tag = {}
        # self.words_count = {}
        # self.tags_count = {}
        # self.tags = set(['*'])
        # self.words = set()
        # self.sentences_with_word_and_tag = []
        # self.sentences_with_only_word = []
        # self.sentences_with_only_tag = []
        # self.lamda = lamda  # unused
        
        # with open(file_name, 'r') as f:
            # for line in f.readlines():
                # sentence_with_word_and_tag = []
                # sentence_with_only_word = []
                # sentence_with_only_tag = []
                # for wordtag in line.split():
                    # sentence_with_word_and_tag.append(wordtag)
                    # word, tag = wordtag.split('_')
                    # self.tags.add(tag)
                    # self.words.add(word)
                    # sentence_with_only_word.append(word)
                    # sentence_with_only_tag.append(tag)
                    
                    # if word not in self.tags_per_word:
                        # self.tags_per_word[word] = dict()
                    # if tag not in self.tags_per_word[word]:
                        # self.tags_per_word[word][tag] = 0
                    # self.tags_per_word[word][tag] += 1
                    
                    # if tag not in self.tags_count:
                        # self.tags_count[tag] = 0
                    # self.tags_count[tag] += 1
                    
                    # if tag not in self.words_per_tag:
                        # self.words_per_tag[tag] = dict()
                    # if word not in self.words_per_tag[tag]:
                        # self.words_per_tag[tag][word] = 0
                    # self.words_per_tag[tag][word] += 1
                    
                    # if word not in self.words_count:
                        # self.words_count[word] = 0
                    # self.words_count[word] += 1

                # self.sentences_with_word_and_tag.append(sentence_with_word_and_tag)
                # self.sentences_with_only_word.append(sentence_with_only_word)
                # self.sentences_with_only_tag.append(sentence_with_only_tag)
        
        # self.CommonWords = list(common_words(self.words_count, limit_common_words))
        # self.tag_indices = {}
        # ind = 0
        # for t in self.tags:
            # self.tag_indices[t] = ind
            # ind = ind + 1
        # self.features ={'self.f100_words_tags_count_dict'}

    # def get_statistics(self):
        # for sentence in self.sentences_with_word_and_tag:
            # tag = '*'
            # word = None
            # next_word = None
            # prev_word = None
            # tag_2 = '*'
            # tag_1 = '*'
            # prev_word = None
            # for word_with_tag in sentence:
                # prev_word = word
                # tag_2 = tag_1
                # tag_1 = tag
                # word, tag = separate_tag_from_word(word_with_tag)

                # #word_tag F100
                # if (word, tag) not in self.f100_words_tags_count_dict:
                    # self.f100_words_tags_count_dict[(word, tag)] = 1
                # else:
                    # self.f100_words_tags_count_dict[(word, tag)] += 1

                # #suffix F101
                # suffix = get_suffix(word)
                # if suffix != None:
                    # if (suffix, tag) not in self.f101_suffixes:
                        # self.f101_suffixes[(suffix, tag)] = 1
                    # else:
                        # self.f101_suffixes[(suffix, tag)] += 1

                # #prefix F102
                # prefix = get_prefix(word)
                # if prefix != None:
                    # if (prefix, tag) not in self.f102_prefix:
                        # self.f102_prefix[(prefix, tag)] = 1
                    # else:
                        # self.f102_prefix[(prefix, tag)] += 1

                # #trigram F103
                # if tag_2 != None:
                    # if (tag_2, tag_1, tag) not in self.f103_trigram:
                        # self.f103_trigram[(tag_2, tag_1, tag)] = 1
                    # else:
                        # self.f103_trigram[(tag_2, tag_1, tag)] += 1

                # #bigram F104
                # if tag_1 != None:
                    # if (tag_1, tag) not in self.f104_bigram:
                        # self.f104_bigram[(tag_1, tag)] = 1
                    # else:
                        # self.f104_bigram[(tag_1, tag)] += 1
                # #tag F105
                # if tag not in self.f105_tag_common:
                    # self.f105_tag_common[tag] = 1
                # else:
                    # self.f105_tag_common[tag] += 1

                # #prev word F106
                # if prev_word != None:
                    # if (prev_word, tag) not in self.f106_prev_word:
                        # self.f106_prev_word[(prev_word, tag)] = 1
                    # else:
                        # self.f106_prev_word[(prev_word, tag)] += 1

                # #next word F107
                # if tag_1 != None:
                    # if (word, tag_1) not in self.f107_next_word:
                        # self.f107_next_word[(word, tag_1)] = 1
                    # else:
                        # self.f107_next_word[(word, tag_1)] += 1

                # # numbers F108
                # if get_number(word) == True:
                    # if (word, tag) not in self.f108_numbers:
                        # self.f108_numbers[(word, tag)] = 1
                    # else:
                        # self.f108_numbers[(word, tag)] += 1

                # #capital letters F109
                # if get_capital_letter(word) == True:
                    # if (word, tag) not in self.f109_capital_letters:
                        # self.f109_capital_letters[(word, tag)] = 1
                    # else:
                        # self.f109_capital_letters[(word, tag)] += 1


# class feature2id_class():
    # def __init__(self, feature_statistics, threshold):
        # self.feature_statistics = feature_statistics
        # self.threshold = threshold
        # self.n_total_features = 0
        # self.n_tag_pairs = 0

        # #features
        # self.f100_words_tags_count_dict = {}
        # self.f102_prefix = {}
        # self.f101_suffixes = {}
        # self.f103_trigram = {}
        # self.f104_bigram = {}
        # self.f105_tag_common = {}
        # self.f106_prev_word = {}
        # self.f107_next_word = {}
        # self.f108_numbers = {}
        # self.f109_capital_letters = {}

    # def get_features(self):
        # for sentence in self.feature_statistics.sentences_with_word_and_tag:
            # tag = '*'
            # word = None
            # next_word = None
            # prev_word = None
            # tag_2 = '*'
            # tag_1 = '*'
            # prev_word = None
            # for word_with_tag in sentence:
                # prev_word = word
                # tag_2 = tag_1
                # tag_1 = tag
                # word, tag = separate_tag_from_word(word_with_tag)

                # #word_tag F100
                # if (word, tag) not in self.f100_words_tags_count_dict and (self.feature_statistics.f100_words_tags_count_dict[(word, tag)] >= self.threshold):
                    # self.f100_words_tags_count_dict[(word, tag)] = self.n_tag_pairs
                    # self.n_tag_pairs += 1

                # #suffix F101
                # suffix = get_suffix(word)
                # if suffix != None:
                    # if (suffix, tag) not in self.f101_suffixes and (self.feature_statistics.f101_suffixes[(suffix, tag)] >= self.threshold):
                        # self.f101_suffixes[(suffix, tag)] = self.n_tag_pairs
                        # self.n_tag_pairs += 1

                # #prefix F102
                # prefix = get_prefix(word)
                # if prefix != None:
                    # if (prefix, tag) not in self.f102_prefix and (self.feature_statistics.f102_prefix[(prefix, tag)] >= self.threshold):
                        # self.f102_prefix[(prefix, tag)] = self.n_tag_pairs
                        # self.n_tag_pairs += 1

                # #trigram F103
                # if tag_2 != None:
                    # if (tag_2, tag_1, tag) not in self.f103_trigram and (self.feature_statistics.f103_trigram[(tag_2, tag_1, tag)] >= self.threshold):
                        # self.f103_trigram[(tag_2, tag_1, tag)] = self.n_tag_pairs
                        # self.n_tag_pairs += 1

                # #bigram F104
                # if tag_1 != None:
                    # if (tag_1, tag) not in self.f104_bigram and (self.feature_statistics.f104_bigram[(tag_1, tag)] >= self.threshold):
                        # self.f104_bigram[(tag_1, tag)] = self.n_tag_pairs
                        # self.n_tag_pairs += 1

                # #tag F105
                # if tag not in self.f105_tag_common and (self.feature_statistics.f105_tag_common[tag] >= self.threshold):
                    # self.f105_tag_common[tag] = self.n_tag_pairs
                    # self.n_tag_pairs += 1

                # #prev word F106
                # if prev_word != None:
                    # if (prev_word, tag) not in self.f106_prev_word and (self.feature_statistics.f106_prev_word[(prev_word, tag)] >= self.threshold):
                        # self.f106_prev_word[(prev_word, tag)] = self.n_tag_pairs
                        # self.n_tag_pairs += 1

                # #next word F107
                # if tag_1 != None:
                    # if (word, tag_1) not in self.f107_next_word and (self.feature_statistics.f107_next_word[(word, tag_1)] >= self.threshold):
                        # self.f107_next_word[(word, tag_1)] = self.n_tag_pairs
                        # self.n_tag_pairs += 1

                # # numbers F108
                # if get_number(word) == True:
                    # if (word, tag) not in self.f108_numbers and (self.feature_statistics.f108_numbers[(word, tag)] >= self.threshold):
                        # self.f108_numbers[(word, tag)] = self.n_tag_pairs
                        # self.n_tag_pairs += 1

                # #capital letters F109
                # if get_capital_letter(word) == True:
                    # if (word, tag) not in self.f109_capital_letters and (self.feature_statistics.f109_capital_letters[(word, tag)] >= self.threshold):
                        # self.f109_capital_letters[(word, tag)] = self.n_tag_pairs
                        # self.n_tag_pairs += 1

        # self.n_total_features += self.n_tag_pairs




# def common_words(words_dict, limit):
    # l = []
    # for word in words_dict:
        # if words_dict[word] > limit:
            # l.append(word)
    # return l


# def get_prefix(word):
    # return word[:3]


# def get_suffix(word):
    # return word[-3:]


# def get_number(word):
    # return any(char.isdigit() for char in word)


# def get_capital_letter(word):
    # return word == word.lower()


