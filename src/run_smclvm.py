from codebase.classes_smclvm import PariclesSMCLVM
from codebase.ibis import model_phonebook, essl, corrcoef_2D
from codebase.mcmc_tlk_latent import (
    gen_latent_weights_master
)
import numpy as np
from tqdm import tqdm
from codebase.file_utils import (
    save_obj,
    load_obj,
)
from scipy.special import logsumexp
from pdb import set_trace

def run_smclvm(
    exp_data,
    model_num,
    size,
    gen_model,
    log_dir,
    degeneracy_limit=0.5,
    name="smc2",
):

    param_names = model_phonebook(model_num)["param_names"]
    latent_names = model_phonebook(model_num)["latent_names"]
    jitter_corrs = dict()
    for t in range(exp_data.size):
        jitter_corrs[t] = dict()
    particles = PariclesSMCLVM(
        name="ibis_lvm",
        model_num=model_num,
        size=size,
        param_names=param_names,
        latent_names=latent_names,
        latent_model_num=1,
        hmc_adapt_nsim = 200,
        hmc_post_adapt_nsim = 5,
    )
    particles.set_log_dir(log_dir)
    if gen_model:
        particles.compile_prior_model()
        particles.compile_model()
    else:
        particles.load_prior_model()
        particles.load_model()

    log_lklhds = np.empty(exp_data.size)
    degeneracy_limit = 0.5

    particles.set_latentvars_shape(exp_data.get_stan_data())
    particles.sample_prior_particles(exp_data.get_stan_data_at_t2(0))  # sample prior particles
    # particles.initialize_latentvars(exp_data.get_stan_data())
    particles.reset_weights()  # set weights to 0
    particles.initialize_counter(exp_data.get_stan_data())


    for t in tqdm(range(exp_data.size)):
        particles.sample_latent_variables(exp_data.get_stan_data_at_t(t), t)
        particles.get_theta_incremental_weights(exp_data.get_stan_data_at_t(t), t)
        log_lklhds[t] = particles.get_loglikelihood_estimate()
        particles.update_weights()
        
        if (essl(particles.weights) < degeneracy_limit * particles.size) and (
            t + 1
        ) < exp_data.size:
            particles.add_ess(t)
            particles.resample_particles()
            
            particles.gather_latent_variables_up_to_t(
                t+1, 
                exp_data.get_stan_data_upto_t(t+1)
            )
            # add corr of param before jitter
            pre_jitter = dict()
            for p in param_names:
                if p not in ['zz', 'yy']:
                    pre_jitter[p] = particles.particles[p]
            ###
            particles.jitter(exp_data.get_stan_data_upto_t(t + 1))

            # add corr of param
            for p in param_names:
                if p not in ['zz', 'yy']:
                    jitter_corrs[t][p] = corrcoef_2D(
                        pre_jitter[p], particles.particles[p]
                    )
            ###
            
            particles.check_particles_are_distinct()

            particles.reset_weights()
        else:
            pass

        save_obj(t, "t", log_dir)
        save_obj(particles, "particles", log_dir)
        save_obj(jitter_corrs, "jitter_corrs", log_dir)
        save_obj(log_lklhds, "log_lklhds", log_dir)

    print("\n\n")
    marg_lklhd = np.exp(logsumexp(log_lklhds))
    print("Marginal Likelihood %.5f" % marg_lklhd)
    save_obj(marg_lklhd, "marg_lklhd", log_dir)

    output = dict()
    output["particles"] = particles
    output["log_lklhds"] = log_lklhds
    output["marg_lklhd"] = marg_lklhd
    output["jitter_corrs"] = jitter_corrs
    return output
