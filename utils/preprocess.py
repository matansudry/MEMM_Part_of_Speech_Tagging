import numpy as np


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
        self.shuffled = False
        
    def _init_loader(self, shuffle, seed, new=False):
        new_batch_loader = self.sentences.copy()
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(new_batch_loader)
        self.shuffled = shuffle
        if new:
            self.batch_loader = new_batch_loader
        else:
            self.batch_loader.extend(new_batch_loader)

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
        if self.shuffled != shuffle:
            self._init_loader(shuffle, seed, new=True)
        if len(self.batch_loader) < batch_size:
            self._init_loader(shuffle, seed)
            
        sentences = self.batch_loader[:batch_size]
        del self.batch_loader[:batch_size]
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

