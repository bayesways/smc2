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


def compile_model(model_num, prior, log_dir):
    path_to_stan = './codebase/stancode/'

    if prior: 
        with open('%spriors/model_%s.stan'%(
        path_to_stan,
        model_num
        ), 'r') as file:
            model_code = file.read()
    else:
        with open('%smodels/model_%s.stan'%(
        path_to_stan,
        model_num
        ), 'r') as file:
            model_code = file.read()

    sm = pystan.StanModel(model_code=model_code, verbose=False)
    
    if prior:
        save_obj(sm, 'sm_prior', log_dir)
    else:
        save_obj(sm, 'sm', log_dir)
    return sm


def sample_prior_particles(
    data,
    gen_model,
    model_num,
    param_names,
    num_samples, 
    num_chains, 
    log_dir
    ):

    if gen_model:
        sm_prior = compile_model(model_num, True, log_dir)
        save_obj(sm_prior, 'sm_prior', log_dir)
    else:
        sm_prior = load_obj('sm_prior', log_dir)
    
    fit_run = sm_prior.sampling(
        # data={
        #     'N':data['N'],
        #     'J': data['J'],
        #     'y' : data['y']
        # },
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
        # data={
        #     'N':data['N'],
        #     'J': data['J'],
        #     'y' : data['y']
        # },
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
    gen_model,
    model_num,
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

    if gen_model:
        sm = compile_model(model_num, False, log_dir)
        save_obj(sm, 'sm', log_dir)
    else:
        sm = load_obj('sm', log_dir)

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


def resample_particles(particles):
    assert  particles['M'] == len(particles['w'])

    w = exp_and_normalise(particles['w'])
    nw = w / np.sum(w)
    np.testing.assert_allclose(1., nw.sum())  
    sampled_index = multinomial(nw, particles['M'])
    for key in particles['param_names']:
            particles[key] = particles[key][sampled_index]

    return particles

