import  pystan
import argparse
import numpy as np
from codebase.file_utils import (
    save_obj,
    load_obj,
    make_folder,
    path_backslash
)
from codebase.ibis import (
    run_mcmc,
    essl,
    resample_particles
)
from codebase.resampling_routines import multinomial
from codebase.data import get_data
from scipy.stats import bernoulli, multivariate_normal
from scipy.special import expit, logsumexp
from tqdm import tqdm
import pdb


def jitter(data, particles, log_dir):

    m=0
    fit_run = run_mcmc(
        data = data,
        gen_model = False,
        model_num = 0,
        num_samples = 20, 
        num_warmup = 1000,
        num_chains = 1,
        log_dir = log_dir,
        initial_values = {
            'alpha' : particles['alpha'][m,0],
            'L_R': particles['L_R'][m,0]    
        },
        load_inv_metric= False, 
        adapt_engaged = True
        )

    last_position = fit_run.get_last_position()[0] # select chain 1
    mass_matrix = fit_run.get_inv_metric(as_dict=True)
    stepsize = fit_run.get_stepsize()

    particles['alpha'][m] = last_position['alpha']
    particles['L_R'][m] = last_position['L_R']
    particles['Marg_cov'][m] = last_position['Marg_cov']

    for m in range(1, particles['M']):
        fit_run = run_mcmc(
            data = data,
            gen_model = False,
            model_num = 0,
            num_samples = 20, 
            num_warmup = 0,
            num_chains = 1,
            log_dir = log_dir,
            initial_values = {
                'alpha' : particles['alpha'][m,0],
                'L_R': particles['L_R'][m,0]    
            },
            inv_metric= mass_matrix,
            adapt_engaged=False,
            stepsize = stepsize
            )
        last_position = fit_run.get_last_position()[0] # select chain 1

        particles['alpha'][m] = last_position['alpha']
        particles['L_R'][m] = last_position['L_R']
        particles['Marg_cov'][m] = last_position['Marg_cov']

    return particles



def loglklhd_z(y, z):
    """
    dim(y) = k
    dim(z) = k
    """
    a = np.log(expit(z)) * y + np.log(1 - expit(z)) * (1-y) 
    return np.sum(a)



def loglklhd_z_vector(y, mean, cov, nsim_z):
    """
    dim(y) = k
    
    Generate z samples from normal and compute the 
    mean likelihood of all z
    """

    z = multivariate_normal(
        mean,
        cov,
        ).rvs(size=nsim_z)
    loglklhds = np.empty(nsim_z)
    for i in range(nsim_z):
        loglklhds[i] = np.sum(
            loglklhd_z(y, z[i])
            )
    return np.mean(loglklhds)


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
        weights[m] = loglklhd_z_vector(
            y = y,
            mean = particles['alpha'][m].reshape(y.shape),
            cov = particles['Marg_cov'][m].reshape(y.shape[0],y.shape[0]),
            nsim_z = 10
        )
    return weights
