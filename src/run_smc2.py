from codebase.classes_smc2 import ParticlesLVM
from codebase.ibis import model_phonebook, essl
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


def run_smc2(
    exp_data,
    model_num,
    size,
    bundle_size,
    gen_model,
    log_dir,
    degeneracy_limit=0.5,
    name="smc2",
):

    param_names = model_phonebook(model_num)["param_names"]
    latent_names = model_phonebook(model_num)["latent_names"]
    jitter_corrs = dict()
    for p in param_names:
        jitter_corrs[p] = np.zeros(exp_data.size)
    particles = ParticlesLVM(
        name="ibis_lvm",
        model_num=model_num,
        size=size,
        bundle_size=bundle_size,
        param_names=param_names,
        latent_names=latent_names,
        latent_model_num=1,
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

    particles.initialize_particles()
    particles.initialize_bundles(exp_data.get_stan_data())
    particles.initialize_latent_var_given_theta(exp_data.get_stan_data())
    particles.sample_prior_particles(exp_data.get_stan_data())  # sample prior particles
    particles.reset_weights()  # set weights to 0
    particles.initialize_counter(exp_data.get_stan_data())

    for t in tqdm(range(exp_data.size)):
        particles.sample_latent_bundle_at_t(t, exp_data.get_stan_data_at_t2(t))
        particles.get_theta_incremental_weights_at_t(exp_data.get_stan_data_at_t(t))
        log_lklhds[t] = particles.get_loglikelihood_estimate()
        particles.update_weights()

        
        if (essl(particles.weights) < degeneracy_limit * particles.size) and (
            t + 1
        ) < exp_data.size:
            particles.add_ess(t)
            particles.resample_particles_bundles()

            particles.gather_latent_variables_up_to_t(
                t+1, 
                exp_data.get_stan_data_upto_t(t+1)
            )

            particles.jitter_bundles_and_pick_one(exp_data.get_stan_data_upto_t(t + 1))
            particles.check_latent_particles_are_distinct()
            
            # add corr of param before jitter
            pre_jitter = dict()
            for p in param_names:
                pre_jitter[p] = particles.extract_particles_in_numpy_array(p).flatten()
            ###
            particles.jitter(exp_data.get_stan_data_upto_t(t + 1))

            # add corr of param
            for p in param_names:
                jitter_corrs[p][t] = np.corrcoef(
                    pre_jitter[p], particles.extract_particles_in_numpy_array(p).flatten()
                )[0, 1]
            ###
            

            particles.reset_weights()
            particles.check_particles_are_distinct()
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