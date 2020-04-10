import numpy as np
import scipy
from scipy.optimize import minimize
from utils.features import features_to_vector
from utils.decoder import Train_tagger, Test_tagger, separate_tag_from_word, common_words
from utils.score import accuracy, top_k_erros, confusion_matrix


# params, maybe need to remove that
model_name = 'model.pickle'
model_matrices = 'model_matrices.pickle'
model_preprocess = 'model_preprocess.pickle'
load_model = False
load_matrices = False
load_preprocess = False
verbose = 1

# variables
number_of_features = 10
limit_common_words = 5

# data files
comp1_path = 'data/comp1.words'
comp2_path = 'data/comp2.words'
test1_path = 'data/test1.wtag'
train1_path = 'data/train1.wtag'
train2_path = 'data/train2.wtag'







def main():
    #load files
    comp1 = open(comp1_path, 'r')
    comp2 = open(comp2_path, 'r')
    test1 = open(test1_path, 'r')
    train1 = open(train1_path, 'r')
    train2 = open(train2_path, 'r')
    train1_model = Train_tagger(train1, limit_common_words)
    train2_model = Train_tagger(train2, limit_common_words)
    test_model = Test_tagger(test1, limit_common_words)

    matan=1
    #using pretrained model

    #tutorial
    word_input = ((1, 2, ('Matan',), 1), 'Vt')
    features_to_vector(word_input, number_of_features)


if __name__ == "__main__":
    main()