import numpy as np
import scipy
from scipy import optimize
from utils import preprocess
# from scipy.optimize import fmin_l_bfgs_b, minimize
# from utils.preprocess import feature_statistics_class, feature2id_class, Test_tagger, separate_tag_from_word, common_words
# from utils import classifier import Optimization
from utils import score# import accuracy, top_k_erros, confusion_matrix


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
args = None
lamda = 0.1

# data files
comp1_path = 'data/comp1.words'
comp2_path = 'data/comp2.words'
test1_path = 'data/test1.wtag'
train1_path = 'data/train1.wtag'
train2_path = 'data/train2.wtag'


def main():

    #load files and preprocessing
    statistics = preprocess.feature_statistics_class(train1_path, limit_common_words, lamda)
    statistics.get_statistics()
    feature2id = preprocess.feature2id_class(statistics, threshold)
    feature2id.get_features()

    #Optimizion
    matan = classifier.Optimization(statistics, feature2id, args)
    matan.likelihood()
    matan = 1
    #train2_model = feature_statistics_class(train2_path, limit_common_words)
    #test_model = preprocess.Test_tagger(test1_path, limit_common_words)








if __name__ == "__main__":
    main()