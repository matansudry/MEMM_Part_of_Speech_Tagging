import numpy as np
import scipy
from scipy.optimize import minimize

number_of_features = 10


class Tagger():
    def __init__(self, file, limit_common_words):
        self.TagsPerWord = {}
        self.WordCount = {}
        self.tags = ['*']
        self.words = []
        self.unseparated = []
        self.sentences = []
        for line in file:
            sentence = []
            for wordtag in line.split():
                sentence.append(wordtag)
                self.unseparated.append(wordtag)
                word, tag = separate_tag_from_word(wordtag)
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
            self.sentences.append(sentence)
        self.CommonWords = list(common_words(self.WordCount, limit_common_words))
        self.tag_indices = {}
        ind = 0
        for t in self.tags:
            self.tag_indices[t] = ind
            ind = ind + 1
        self.features ={}

    # need to complete
    def adding_features(self):
        matan=1


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


def viterbi():
    matan=1
    #need to complete


def confusion_matrix():
    matan=1
    #need to complete


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


#need to complete
def f108_numbers(word_input):
    matan=1
    return matan


#need to understand what is the t here
def f109_capital_letters(word_input):
    index = word_input[0][3]
    word = word_input[0][2][index - 1]
    is_capital_letters = word[0].isupper()
    if is_capital_letters:
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
    return features_vector


def main():
    comp1 = open('data/comp1.words', 'r')
    comp2 = open('data/comp2.words', 'r')
    test1 = open('data/test1.wtag', 'r')
    train1 = open('data/train1.wtag', 'r')
    train2 = open('data/train2.wtag', 'r')

    model_1 = Tagger(train1, limit_common_words=5)
    model_2 = Tagger(train2, limit_common_words=25)

    word_input = ((1, 2, ('Matan',), 1), 'Vt')
    features_to_vector(word_input, number_of_features)


if __name__ == "__main__":
    main()