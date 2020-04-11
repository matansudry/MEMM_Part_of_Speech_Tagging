import numpy as np
import scipy
from scipy.optimize import fmin_l_bfgs_b
import pickle
from utils.preprocess import separate_tag_from_word, get_suffix, get_prefix, get_number, get_capital_letter


class Optimization:
    def __init__(self, statistics, feature2id, args):
        self.statistics = statistics
        self.feature2id = feature2id
        self.args = args
        self.w_0 = np.random.rand(feature2id.n_total_features).astype(np.float32)

        self.weights = 1 # need to check it

        self.Y = statistics.tags

    def calc_objective_per_iter(self):

        likelihood = self.likelihood(self)
        grad = empirical_counts - expected_counts - regularization_grad

        return (-1) * likelihood, (-1) * grad

    def learning_using_bfgs(self):
        optimal_params = fmin_l_bfgs_b(func=self.calc_objective_per_iter, x0=self.w_0, args=self.args, maxiter=1000, iprint=50)
        self.weights = optimal_params[0]
        weights_path = '../output/trained_weights_data_i.pkl'  # i identifies which dataset this is trained on
        with open(weights_path, 'wb') as f:
            pickle.dump(optimal_params, f)

    def likelihood(self):
        weights = self.w_0
        sentence_count = 0
        sum_liner_term = 0
        sum_normalization_term = 0
        for sentence in self.statistics.sentences_with_word_and_tag:
            sentence_count += 1
            tag = '*'
            word = None
            next_word = None
            prev_word = None
            tag_2 = '*'
            tag_1 = '*'
            for index in range(len(sentence)):
                index_dict = []
                prev_word = word
                tag_2 = tag_1
                tag_1 = tag
                word, tag = separate_tag_from_word(sentence[index])

                #getting indexes from current tag
                index_dict = self.get_indexes(tag, word, next_word, prev_word, tag_2, tag_1)

                #adding the weights from the indexes we found per word
                for i in index_dict:
                    sum_liner_term += self.w_0[i]
                sum_exp = 0
                for temp_tag in self.statistics.tags:
                    temp_index_dict = []
                    temp_index_dict = self.get_indexes(temp_tag, word, next_word, prev_word, tag_2, tag_1)
                    for i in temp_index_dict:
                        sum_exp += self.w_0[i]
                sum_normalization_term = sum_normalization_term + np.log(sum_exp)
        regularization = 0.5 * self.statistics.lamda * sum(weights[i] ** 2 for i in range(len(weights)))
        return ((-1) * (sum_liner_term  - sum_normalization_term - regularization))

    def get_indexes(self, tag, word, next_word, prev_word, tag_2, tag_1):
        # getting all the indexes of the word and tag

        index_dict = []
        if (word, tag) in self.feature2id.f100_words_tags_count_dict:
            index_dict.append(self.feature2id.f100_words_tags_count_dict[(word, tag)])
        suffix = get_suffix(word)
        if suffix != None and (suffix, tag) in self.feature2id.f101_suffixes:
            index_dict.append(self.feature2id.f101_suffixes[(suffix, tag)])
        prefix = get_prefix(word)
        if prefix != None and (prefix, tag) in self.feature2id.f102_prefix:
            index_dict.append(self.feature2id.f102_prefix[(prefix, tag)])
        if tag_2 != None and (tag_2, tag_1, tag) in self.feature2id.f103_trigram:
            index_dict.append(self.feature2id.f103_trigram[(tag_2, tag_1, tag)])
        if tag_1 != None and (tag_1, tag) in self.feature2id.f104_bigram:
            index_dict.append(self.feature2id.f104_bigram[(tag_1, tag)])

        if tag in self.feature2id.f105_tag_common:
            index_dict.append(self.feature2id.f105_tag_common[tag])

        if prev_word != None and (prev_word, tag) in self.feature2id.f106_prev_word:
            index_dict.append(self.feature2id.f106_prev_word[(prev_word, tag)])
        if tag_1 != None and (word, tag_1) in self.feature2id.f107_next_word:
            index_dict.append(self.feature2id.f107_next_word[(word, tag_1)])
        if get_number(word) == True and (word, tag) in self.feature2id.f108_numbers:
            index_dict.append(self.feature2id.f108_numbers[(word, tag)])
        if get_capital_letter(word) == True and (word, tag) in self.feature2id.f109_capital_letters:
            index_dict.append(self.feature2id.f109_capital_letters[(word, tag)])
        return index_dict

    def grad(self):
        weights = self.w_0
        sentence_count = 0
        sum_liner_term = 0
        sum_normalization_term = 0
        for sentence in self.statistics.sentences_with_word_and_tag:
            sentence_count += 1
            tag = '*'
            word = None
            next_word = None
            prev_word = None
            tag_2 = '*'
            tag_1 = '*'
            for index in range(len(sentence)):
                index_dict = []
                prev_word = word
                tag_2 = tag_1
                tag_1 = tag
                word, tag = separate_tag_from_word(sentence[index])

                #getting indexes from current tag
                index_dict = self.get_indexes(tag, word, next_word, prev_word, tag_2, tag_1)

                #adding the weights from the indexes we found per word
                for i in index_dict:
                    sum_liner_term += self.w_0[i]
                sum_exp = 0
                for temp_tag in self.statistics.tags:
                    temp_index_dict = []
                    temp_index_dict = self.get_indexes(temp_tag, word, next_word, prev_word, tag_2, tag_1)
                    for i in temp_index_dict:
                        sum_exp += self.w_0[i]
                sum_normalization_term = sum_normalization_term + np.log(sum_exp)
        regularization = 0.5 * self.statistics.lamda * sum(weights[i] ** 2 for i in range(len(weights)))
        return ((-1) * (sum_liner_term  - sum_normalization_term - regularization))


















