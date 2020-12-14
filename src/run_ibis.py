from codebase.classes import Particles
from codebase.ibis import model_phonebook, essl
import numpy as np
from tqdm import tqdm
from codebase.file_utils import (
    save_obj,
    load_obj,
)
from scipy.special import logsumexp



def run_ibis(
    exp_data,
    model_num,
    size,
    gen_model,
    log_dir,
    degeneracy_limit = 0.5,
    name = 'ibis'
    ):

    ## setup particles
    param_names = model_phonebook(model_num)['param_names']
    latent_names = model_phonebook(model_num)['latent_names']
    particles = Particles(
        name = name,
        model_num = model_num,
        size = size,
        param_names = param_names,
        latent_names = latent_names)
    particles.set_log_dir(log_dir)
    if gen_model:
        particles.compile_prior_model()
        particles.compile_model()
    else:
        particles.load_prior_model()
        particles.load_model()

    particles.sample_prior_particles(exp_data.get_stan_data()) # sample prior particles
    particles.reset_weights() # set weights to 0
    log_lklhds = np.empty(exp_data.size)
    degeneracy_limit = 0.5
    for t in tqdm(range(exp_data.size)):
        particles.get_incremental_weights(
            exp_data.get_stan_data_at_t(t)
            )
        log_lklhds[t] =  particles.get_loglikelihood_estimate()
        particles.update_weights()
        
        if (essl(particles.weights) < degeneracy_limit * particles.size) and (t+1) < exp_data.size:
            particles.resample_particles()
            particles.jitter(exp_data.get_stan_data_upto_t(t+1))
            particles.reset_weights()
        else:
            particles.update_weights()

        save_obj(particles, 'particles', log_dir)
        save_obj(t, 't', log_dir)
    save_obj(log_lklhds, 'log_lklhds', log_dir)

    print('\n\n')
    marg_lklhd = np.exp(logsumexp(log_lklhds))
    print('Marginal Likelihood %.5f'%marg_lklhd)
    save_obj(marg_lklhd, 'marg_lklhd', log_dir)

    output = dict()
    output['particles'] = particles
    output['log_lklhds'] = log_lklhds
    output['marg_lklhd'] = marg_lklhd
    return output
