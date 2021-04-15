from codebase.classes_mcmc import MCMC
import numpy as np
from codebase.file_utils import save_obj, load_obj, make_folder, path_backslash
from tqdm import tqdm
from pdb import set_trace


def run_mcmc_from_start(
    stan_data,
    mcmc_nsim,
    mcmc_adapt_nsim,
    model_num,
    bundle_size,
    gen_model,
    param_names,
    latent_names,
    log_dir,
    hmc_adapt_nsim,
    hmc_post_adapt_nsim,
    name="mcmc",
):

    particles = MCMC(
        name=name,
        model_num=model_num,
        param_names=param_names,
        latent_names=latent_names,
        bundle_size=bundle_size,
        latent_model_num=1,
        hmc_adapt_nsim=hmc_adapt_nsim,
        hmc_post_adapt_nsim=hmc_post_adapt_nsim
    )

    particles.set_log_dir(log_dir)

    if gen_model:
        particles.compile_prior_model()
        particles.compile_model()
    else:
        particles.load_prior_model()
        particles.load_model()

    particles.sample_prior_particles(stan_data)
    betas = np.empty((mcmc_nsim, 6, 1))
    alphas = np.empty((mcmc_nsim, 6))
    zs = np.empty((mcmc_nsim, stan_data["N"], 1))
    ys = np.empty((mcmc_nsim, stan_data["N"], 6))

    particles.sample_latent_variables(stan_data)

    for i in tqdm(range(mcmc_nsim)):
        particles.get_bundle_weights(stan_data)
        particles.sample_latent_particles_star(stan_data)
        particles.sample_latent_var_given_theta(stan_data)

        zs[i] = particles.latent_mcmc_sample["z"]
        ys[i] = particles.latent_mcmc_sample["y"]

        if i < mcmc_adapt_nsim:
            particles.sample_theta_given_z_and_save_mcmc_parms(stan_data)
        else:
            particles.sample_theta_given_z_with_used_mcmc_params(stan_data)
        alphas[i] = particles.particles["alpha"]
        betas[i] = particles.particles["beta"]
    ps = dict()
    ps["alpha"] = alphas
    ps["beta"] = betas
    ps["z"] = zs
    ps["y"] = ys
    ps["accs"] = particles.acceptance
    return ps

def run_mcmc_jitter(
    stan_data,
    mcmc_nsim,
    mcmc_adapt_nsim,
    model_num,
    bundle_size,
    gen_model,
    param_names,
    latent_names,
    initial_values,
    log_dir,
    hmc_adapt_nsim,
    hmc_post_adapt_nsim,
    name="mcmc"
):

    particles = MCMC(
        name=name,
        model_num=model_num,
        param_names=param_names,
        latent_names=latent_names,
        bundle_size=bundle_size,
        latent_model_num=1,
        hmc_adapt_nsim=hmc_adapt_nsim,
        hmc_post_adapt_nsim=hmc_post_adapt_nsim
    )

    particles.set_log_dir(log_dir)

    particles.load_prior_model()
    particles.load_model()

    particles.acceptance = np.zeros(stan_data["N"])
    particles.particles = initial_values.copy()
    particles.sample_latent_variables(stan_data)
    
    for i in range(mcmc_nsim):
        particles.get_bundle_weights(stan_data)
        particles.sample_latent_particles_star(stan_data)
        particles.sample_latent_var_given_theta(stan_data)

        if i < mcmc_adapt_nsim:
            particles.sample_theta_given_z_and_save_mcmc_parms(stan_data)
        else:
            particles.sample_theta_given_z_with_used_mcmc_params(stan_data)
        # alphas[i] = particles.particles["alpha"]
        # betas[i] = particles.particles["beta"]
    
    # ps = dict()
    # ps["alpha"] = particles.particles["alpha"]
    # ps["beta"] = particles.particles["beta"]
    # ps["z"] = particles.latent_particles["z"]
    # ps["y"] = particles.latent_particles["z"]
    # ps["accs"] = particles.acceptance
    return particles


def run_mcmc_jitter_with_used_params(
    stan_data,
    mcmc_nsim,
    model_num,
    bundle_size,
    gen_model,
    param_names,
    latent_names,
    initial_values,
    log_dir,
    hmc_post_adapt_nsim,
    mass_matrix,
    stepsize,
    name="mcmc"
):

    particles = MCMC(
        name=name,
        model_num=model_num,
        param_names=param_names,
        latent_names=latent_names,
        bundle_size=bundle_size,
        latent_model_num=1,
        hmc_adapt_nsim=0,
        hmc_post_adapt_nsim=hmc_post_adapt_nsim
    )

    particles.set_log_dir(log_dir)

    particles.load_prior_model()
    particles.load_model()
    particles.mass_matrix = mass_matrix
    particles.stepsize = stepsize

    particles.acceptance = np.zeros(stan_data["N"])
    particles.particles = initial_values.copy()
    particles.sample_latent_variables(stan_data)

    for i in range(mcmc_nsim):
        particles.get_bundle_weights(stan_data)
        particles.sample_latent_particles_star(stan_data)
        particles.sample_latent_var_given_theta(stan_data)
        particles.sample_theta_given_z_with_used_mcmc_params(stan_data)
        
        # alphas[i] = particles.particles["alpha"]
        # betas[i] = particles.particles["beta"]
    
    # ps = dict()
    # ps["alpha"] = particles.particles["alpha"]
    # ps["beta"] = particles.particles["beta"]
    # ps["z"] = particles.latent_particles["z"]
    # ps["y"] = particles.latent_particles["z"]
    # ps["accs"] = particles.acceptance
    return particles