import numpy as np
import scipy


def main():
    comp1 = open('data/comp1.words', 'r')
    comp2 = open('data/comp2.words', 'r')
    test1 = open('data/test1.wtag', 'r')
    train1 = open('data/train1.wtag', 'r')
    train2 = open('data/train2.wtag', 'r')
    for line in f:
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


    matan = 1


if __name__ == "__main__":
    main()