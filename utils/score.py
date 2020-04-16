import numpy as np
import scipy
from scipy.optimize import fmin_l_bfgs_b


def accuracy(pred_tags, true_tags):
    return (np.array(pred_tags) == np.array(true_tags)).mean()

