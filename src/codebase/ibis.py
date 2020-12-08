import  pystan
import sys, os
import argparse
import numpy as np
from codebase.file_utils import (
    save_obj,
    load_obj,
    make_folder,
    path_backslash
)
from codebase.resampling_routines import multinomial
from scipy.stats import bernoulli, multivariate_normal
from scipy.special import expit, logsumexp
from tqdm import tqdm
import pdb


def model_phonebook_path(model_num, prior):
    path_to_stan = './codebase/stancode/'

    if model_num == 0:
        if prior:
            path = 'saturated/model_0_prior.stan'
        else:
            path = 'saturated/model_0.stan'

    elif model_num == 1:
        if prior:
            path = 'CFA/model_1_prior.stan'
        else:
            path = 'CFA/model_1.stan'
    elif model_num == 2:
        if prior:
            path = 'CFA/model2_big5_prior.stan'
        else:
            path = 'CFA/model2_big5.stan'   
    elif model_num == 3:
        if prior:
            path = 'CFA/model_2_prior.stan'
        else:
            path = 'CFA/model_2.stan'
    elif model_num == 4:
        if prior:
            path = 'EFA/model_2_prior.stan'
        else:
            path = 'EFA/model_2.stan'
    else:
        print("model number not found")
        sys.exit()

    return path_to_stan+path


def model_phonebook(model_num):
    names = dict()

    if model_num == 0:
        names['param_names'] = ['Marg_cov', 'L_R', 'alpha', 'sigma']
        names['latent_names'] = []
    elif model_num == 1:
        names['param_names'] = [
            'sigma_square',
            'alpha',
            'beta_free',
            'Phi_cov',
            'Marg_cov',
            'beta',
            ]
        names['latent_names'] = []
    elif model_num == 2:
        names['param_names'] = [
            'sigma_square',
            'alpha',
            'beta_free',
            'beta_zeros',
            'Phi_cov',
            'Marg_cov',
            'beta',
            'Omega'
            ]
        names['latent_names'] = []
    elif model_num == 3:
        names['param_names'] = [
            'sigma_square',
            'alpha',
            'beta_free',
            'beta_zeros',
            'Phi_cov',
            'Marg_cov',
            'beta',
            'Omega'
            ]
        names['latent_names'] = []
    elif model_num == 4:
        names['param_names'] = [
            'sigma_square',
            'alpha',
            'Marg_cov',
            'beta',
            ]
        names['latent_names'] = []
    elif model_num == 5:
        names['param_names'] = [
            'sigma_square',
            'alpha',
            'Marg_cov',
            'beta',
            'Omega'
            ]
        names['latent_names'] = []
    else:
        print("model number not found")
        sys.exit()

    return names


def compile_model(model_num, prior, log_dir, save=True):
    
    model_bank_path = "./log/compiled_models/model%s/"%model_num
    if not os.path.exists(model_bank_path):
        os.makedirs(model_bank_path)

    with open(
        model_phonebook_path(model_num, prior),
        'r'
        ) as file:
        model_code = file.read()
    
    sm = pystan.StanModel(model_code=model_code, verbose=False)
    
    if save:
        if prior:
            save_obj(sm, 'sm_prior', log_dir)
            save_obj(sm, 'sm_prior', model_bank_path)
            file = open('%smodel_prior.txt'%model_bank_path, "w")
            file.write(model_code)
            file.close()
        else:
            save_obj(sm, 'sm', log_dir)
            save_obj(sm, 'sm', model_bank_path)
            file = open('%smodel.txt'%model_bank_path, "w")
            file.write(model_code)
            file.close()
    return sm


def sample_prior_particles(
    data,
    sm_prior,
    param_names,
    num_samples, 
    num_chains, 
    log_dir
    ):    
    fit_run = sm_prior.sampling(
        data = data,
        iter=num_samples,
        warmup=0,
        chains=num_chains,
        algorithm = 'Fixed_param',
        n_jobs=1
    )
    particles = fit_run.extract(
        permuted=False, pars=param_names)

    return particles


def get_initial_values_dict(particles, m):
    particles_dict = dict()
    for n in particles['param_names']:
        particles_dict[n] = particles[n][m,0]
    return particles_dict


def set_last_position(particles, m, last_position):
    for n in particles['param_names']:
        particles[n][m] = last_position[n]
    return particles


init_values = dict()

def set_initial_values(params):
    global init_values    # Needed to modify global copy of globvar
    init_values = params


def initf1():
    return init_values


def run_stan_model(
    data,
    compiled_model,
    num_samples, 
    num_warmup,
    num_chains,
    initial_values=None,
    inv_metric = None,
    adapt_engaged = False,
    stepsize = None
    ):

    if initial_values is not None:
        set_initial_values(initial_values)

    control={
        "metric" : "diag_e", # diag_e/dense_e
        "adapt_delta" : 0.99,
        "max_treedepth" : 14,
        "adapt_engaged" : adapt_engaged
        }

    if inv_metric is not None:
        control['inv_metric'] = inv_metric
    if stepsize is not None:
        control['stepsize'] = stepsize

    fit_run = compiled_model.sampling(
        data = data,
        iter=num_samples + num_warmup,
        warmup=num_warmup,
        chains=num_chains,
        init=initf1,
        control=control,
        n_jobs=1,
        check_hmc_diagnostics=False
    )

    return fit_run


def run_mcmc(
    data,
    sm,
    num_samples, 
    num_warmup,
    num_chains,
    log_dir,
    initial_values = None,
    inv_metric = None,
    load_inv_metric = False,
    save_inv_metric = False,
    adapt_engaged = False,
    stepsize = None
    ):

    if load_inv_metric:
        inv_metric = load_obj('inv_metric', log_dir)
        
        
    fit_run = run_stan_model(
        data,
        compiled_model = sm,
        num_samples = num_samples, 
        num_warmup = num_warmup,
        num_chains = num_chains,
        initial_values= initial_values,
        inv_metric= inv_metric,
        adapt_engaged=adapt_engaged,
        stepsize=stepsize
        )

    if save_inv_metric is not None:
        inv_metric = fit_run.get_inv_metric(as_dict=True)
        save_obj(inv_metric, 'inv_metric', log_dir)

    return fit_run


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
        initial_values = get_initial_values_dict(particles, m),
        load_inv_metric= False, 
        adapt_engaged = True
        )

    last_position = fit_run.get_last_position()[0] # select chain 1
    mass_matrix = fit_run.get_inv_metric(as_dict=True)
    stepsize = fit_run.get_stepsize()

    particles = set_last_position(particles, m, last_position)

    for m in range(1, particles['M']):
        fit_run = run_mcmc(
            data = data,
            gen_model = False,
            model_num = 0,
            num_samples = 20, 
            num_warmup = 0,
            num_chains = 1,
            log_dir = log_dir,
            initial_values = get_initial_values_dict(particles, m),
            inv_metric= mass_matrix,
            adapt_engaged=False,
            stepsize = stepsize
            )
        last_position = fit_run.get_last_position()[0] # select chain 1

        particles = set_last_position(particles, m, last_position)

    return particles


def exp_and_normalise(lw):
    """Exponentiate, then normalise (so that sum equals one).
    Arguments
    ---------
    lw: ndarray
        log weights.
    Returns
    -------
    W: ndarray of the same shape as lw
        W = exp(lw) / sum(exp(lw))
    Note
    ----
    uses the log_sum_exp trick to avoid overflow (i.e. subtract the max
    before exponentiating)
    See also
    --------
    log_sum_exp
    log_mean_exp
    """
    w = np.exp(lw - lw.max())
    return w / w.sum()


def essl(lw):
    """ESS (Effective sample size) computed from log-weights.
    Parameters
    ----------
    lw: (N,) ndarray
        log-weights
    Returns
    -------
    float
        the ESS of weights w = exp(lw), i.e. the quantity
        sum(w**2) / (sum(w))**2
    Note
    ----
    The ESS is a popular criterion to determine how *uneven* are the weights.
    Its value is in the range [1, N], it equals N when weights are constant,
    and 1 if all weights but one are zero.
    """
    w = np.exp(lw - lw.max())
    return (w.sum())**2 / np.sum(w**2)


def get_resample_index(weights, size):
    w = exp_and_normalise(weights)
    nw = w / np.sum(w)
    np.testing.assert_allclose(1., nw.sum())  
    return multinomial(nw, size)


def resample_particles(particles):
    assert  particles['M'] == len(particles['w'])

    w = exp_and_normalise(particles['w'])
    nw = w / np.sum(w)
    np.testing.assert_allclose(1., nw.sum())  
    sampled_index = multinomial(nw, particles['M'])
    for key in particles['param_names']:
            particles[key] = particles[key][sampled_index]

    return particles

