from codebase.classes_mcmc import MCMC
import numpy as np
from codebase.file_utils import save_obj, load_obj, make_folder, path_backslash
from tqdm import tqdm


def run_mcmc(
    stan_data,
    nsim_mcmc,
    num_warmup,
    model_num,
    bundle_size,
    gen_model,
    param_names,
    latent_names,
    log_dir,
    name="mcmc",
):

    particles = MCMC(
        name=name,
        model_num=model_num,
        nsim=1,
        param_names=param_names,
        latent_names=latent_names,
        bundle_size=bundle_size,
        latent_model_num=1,
    )

    particles.set_log_dir(log_dir)

    if gen_model:
        particles.compile_prior_model()
        particles.compile_model()
    else:
        particles.load_prior_model()
        particles.load_model()

    particles.sample_prior_particles(stan_data)

    betas = np.empty((nsim_mcmc, 6, 1))
    alphas = np.empty((nsim_mcmc, 6))
    zs = np.empty((nsim_mcmc, stan_data["N"], 1))
    ys = np.empty((nsim_mcmc, stan_data["N"], 6))

    particles.sample_latent_variables(stan_data)

    for i in tqdm(range(nsim_mcmc)):
        particles.get_bundle_weights(stan_data)
        particles.sample_latent_particles_star(stan_data)
        particles.sample_latent_var_given_theta(stan_data)

        zs[i] = particles.latent_mcmc_sample["z"]
        ys[i] = particles.latent_mcmc_sample["y_latent"]

        if i < num_warmup:
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
