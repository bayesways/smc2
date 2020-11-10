import  pystan
import argparse
import numpy as np
from codebase.file_utils import (
    save_obj,
    load_obj,
    make_folder,
    path_backslash
)
from codebase.ibis import exp_and_normalise
from codebase.resampling_routines import multinomial
from codebase.data import get_data
from scipy.stats import bernoulli, multivariate_normal, norm, logistic
from scipy.special import expit, logsumexp
from tqdm import tqdm
import pdb


def sample_latent_y(particles, nsim_latent, c = 1):
    J = particles['alpha'].shape[-1]  
    y_latent = np.empty((
    particles['M'],
    nsim_latent,
    J
    ))
    for m in range(particles['M']):
        a = particles['alpha'][m]
        b = particles['beta'][m]
        logistic_dstn = logistic(scale = c)        
        zz = norm.rvs(size=nsim_latent)
        y_latent[m] = a + np.outer(zz, b) 
        for j in range (J):
            y_latent[m, :, j] = y_latent[m, :, j] +\
                logistic_dstn.rvs(size=nsim_latent) 
    
    return y_latent 


def get_latent_weights(y, particles):
    """
    dim(y) = k 

    For a single data point y, compute the
    likelihood of each particle as the 
    average likelihood across a sample of latent
    variables z.
    """

    weights = np.empty((particles['M'], particles['L']))
    for m in range(particles['M']):
        w = bernoulli.logpmf(
            y,
            p=expit(particles['y_latent'][m])
        )
        weights[m] = np.sum(w, axis=1)
    return weights


def sample_latent_y_star(particles):
    for m in range(particles['M']):
        w = exp_and_normalise(particles['lv'][m])
        nw = w / np.sum(w)
        np.testing.assert_allclose(1., nw.sum())  
        sampled_index = multinomial(nw, 1)
        particles['y_latent_star'][m] = particles['y_latent'][m, sampled_index]
    return particles


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
