import numpy as np
import scipy
from scipy.optimize import minimize

class Train_tagger():
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


class Test_tagger():
    def __init__(self, file, limit_common_words):
        self.sentences_with_tag_and_word = []
        self.sentences_with_word_only = []
        self.sentences_with_tag_only = []
        for line in file:
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