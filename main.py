import numpy as np
import scipy
from scipy.optimize import fmin_l_bfgs_b
#from utils.features import features_to_vector
from utils.preprocess import feature_statistics_class, feature2id_class, Test_tagger, separate_tag_from_word, common_words
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
limit_common_words = 5
threshold = 0

# data files
comp1_path = 'data/comp1.words'
comp2_path = 'data/comp2.words'
test1_path = 'data/test1.wtag'
train1_path = 'data/train1.wtag'
train2_path = 'data/train2.wtag'


def main():

    #load files and preprocessing
    comp1 = open(comp1_path, 'r')
    comp2 = open(comp2_path, 'r')
    test1 = open(test1_path, 'r')
    train1 = open(train1_path, 'r')
    train2 = open(train2_path, 'r')
    train1_model = feature_statistics_class(train1, limit_common_words)
    train1_model.get_statistics()
    id_class = feature2id_class(train1_model, threshold)
    id_class.get_features()

    #train2_model = feature_statistics_class(train2, limit_common_words)
    #test_model = Test_tagger(test1, limit_common_words)








if __name__ == "__main__":
    main()