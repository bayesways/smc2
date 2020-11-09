import  pystan
import argparse
import numpy as np
from codebase.file_utils import (
    save_obj,
    load_obj,
    make_folder,
    path_backslash
)
from codebase.resampling_routines import multinomial
from codebase.data import get_data
from scipy.stats import bernoulli, multivariate_normal
from scipy.special import expit, logsumexp
from tqdm import tqdm
import pdb


def get_weights(y, particles):
    """
    dim(y) = k 

    For a single data point y, compute the
    likelihood of each particle as the 
    average likelihood across a sample of latent
    variables z.
    """

    weights = np.empty(particles['M'])
    for m in range(particles['M']):
        weights[m] = multivariate_normal.logpdf(
            x = y,
            mean = particles['alpha'][m].reshape(y.shape),
            cov = particles['Marg_cov'][m].reshape(y.shape[0],y.shape[0]),
            allow_singular=True
        )
    return weights
