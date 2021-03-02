import numpy as np
from codebase.data import gen_data_master
from codebase.ibis import (
    compile_model,
    sample_prior_particles,
    get_resample_index,
    run_mcmc,
    get_initial_values_dict,
    exp_and_normalise,
)
from codebase.mcmc_tlk_latent import (
    gen_latent_weights_master,
    generate_latent_variables_bundle,
)
from codebase.file_utils import save_obj, load_obj, make_folder, path_backslash
from codebase.resampling_routines import multinomial
from scipy.special import logsumexp
from scipy.stats import norm
from pdb import set_trace

class MCMC:
    def __init__(
        self,
        name,
        model_num,
        nsim,
        param_names,
        latent_names,
        bundle_size,
        latent_model_num,
        adapt_nsim = 100,
        post_adapt_nsim = 5
    ):
        self.name = name
        self.model_num = model_num
        self.size = nsim
        self.param_names = param_names
        self.latent_names = latent_names
        self.bundle_size = bundle_size
        self.latent_model_num = latent_model_num
        self.adapt_nsim = adapt_nsim
        self.post_adapt_nsim = post_adapt_nsim

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir

    def load_prior_model(self):
        self.compiled_prior_model = load_obj(
            "sm_prior", "./log/compiled_models/model%s/" % self.model_num
        )

    def load_model(self):
        self.compiled_model = load_obj(
            "sm", "./log/compiled_models/model%s/" % self.model_num
        )

    def compile_prior_model(self):
        self.compiled_prior_model = compile_model(
            model_num=self.model_num, prior=True, log_dir=self.log_dir, save=True
        )

    def compile_model(self):
        self.compiled_model = compile_model(
            model_num=self.model_num, prior=False, log_dir=self.log_dir, save=True
        )

    def sample_prior_particles(self, data):
        self.acceptance = np.zeros(data["N"])
        self.particles = sample_prior_particles(
            data=data,
            sm_prior=self.compiled_prior_model,
            param_names=self.param_names,
            num_samples=self.size,
            num_chains=1,
            log_dir=self.log_dir,
        )

    def sample_latent_variables(self, data):
        latent_vars = generate_latent_variables_bundle(
            self.bundle_size,  # don't need
            data["N"],
            data["J"],
            1,  # number of factors
            self.particles["alpha"],
            self.particles["beta"],
        )
        self.latent_particles = latent_vars.copy()
    
    def sample_latent_variables2(self, data):
        latent_vars = generate_latent_variables_bundle(
            self.bundle_size,  # don't need
            data["N"],
            data["J"],
            1,  # number of factors
            self.particles["alpha"],
            self.particles["beta"],
        )
        return latent_vars

    def get_bundle_weights(self, data):
        self.weights = gen_latent_weights_master(
            self.latent_model_num,
            data,
            self.latent_particles["y"],
            self.bundle_size,
        )

    def sample_latent_particles_star(self, data):
        latent_var_star = generate_latent_variables_bundle(
            self.bundle_size,
            data["N"],
            data["J"],
            data["K"],  # number of factors
            self.particles["alpha"],
            self.particles["beta"],
        )
        weights_star = gen_latent_weights_master(
            self.latent_model_num, data, latent_var_star["y"], self.bundle_size
        )
        ## Accept/Reject Step
        logdiff = weights_star.mean(axis=0) - self.weights.mean(axis=0)
        for t in range(data["N"]):
            u = np.random.uniform()
            if np.log(u) <= logdiff[t]:
                self.acceptance[t] += 1
                self.latent_particles["z"][:, t] = latent_var_star["z"][:, t].copy()
                self.latent_particles["y"][:, t] = latent_var_star["y"][
                    :, t
                ].copy()
                self.weights[:, t] = weights_star[:, t].copy()

    def reset_weights(self):
        self.weights = np.zeros(self.bundle_size)

    def sample_latent_var_given_theta(self, data):
        resample_index = np.empty(data["N"], dtype=int)
        for t in range(data["N"]):
            resample_index[t] = get_resample_index(self.weights[:, t], 1).astype(int)
        samples = dict()
        for name in self.latent_names:
            samples[name] = self.latent_particles[name][
                resample_index, np.arange(data["N"])
            ].copy()
        self.latent_mcmc_sample = samples

    def sample_theta_given_z(self, data):
        mcmc_data = data.copy()
        mcmc_data["zz"] = np.copy(self.latent_mcmc_sample["z"])
        fit_run = run_mcmc(
            data=mcmc_data,
            sm=self.compiled_model,
            num_samples=10,
            num_warmup=500,
            num_chains=1,
            log_dir=self.log_dir,
            adapt_engaged=True,
        )
        last_position = fit_run.get_last_position()[0]  # select chain 1
        for name in self.param_names:
            self.particles[name] = last_position[name]

    def sample_theta_given_z_and_save_mcmc_parms(self, data):
        mcmc_data = data.copy()
        mcmc_data["zz"] = np.copy(self.latent_mcmc_sample["z"])
        fit_run = run_mcmc(
            data=mcmc_data,
            sm=self.compiled_model,
            num_samples=self.post_adapt_nsim,
            num_warmup=self.adapt_nsim,
            num_chains=1,
            initial_values = self.particles,
            log_dir=self.log_dir,
            adapt_engaged=True,
        )
        self.mass_matrix = fit_run.get_inv_metric(as_dict=True)
        self.stepsize = fit_run.get_stepsize()
        last_position = fit_run.get_last_position()[0]  # select chain 1
        for name in self.param_names:
            self.particles[name] = last_position[name]

    def sample_theta_given_z_and_save_mcmc_parms2(self, data):
        mcmc_data = data.copy()
        mcmc_data["zz"] = np.copy(self.latent_mcmc_sample["z"])
        # values_dict = dict()
        # values_dict['alpha'] = np.copy(self.particles['alpha']).reshape(6,)
        # values_dict['beta'] = np.copy(self.particles['beta']).reshape(6,1)
        fit_run = run_mcmc(
            data=mcmc_data,
            sm=self.compiled_model,
            num_samples=5,
            num_warmup=100,
            num_chains=1,
            initial_values = self.particles,
            log_dir=self.log_dir,
            adapt_engaged=True,
        )
        save_obj(fit_run.get_inv_metric(as_dict=True), 'mass_matrix', self.log_dir)
        save_obj(fit_run.get_stepsize(), 'stepsize', self.log_dir)
        last_position = fit_run.get_last_position()[0]  # select chain 1
        for name in self.param_names:
            self.particles[name] = last_position[name]

    def sample_theta_given_z_with_used_mcmc_params(self, data):
        mcmc_data = data.copy()
        mcmc_data["zz"] = self.latent_mcmc_sample["z"].copy()
        fit_run = run_mcmc(
            data=mcmc_data,
            sm=self.compiled_model,
            num_samples=self.post_adapt_nsim,
            num_warmup=0,
            num_chains=1,
            initial_values = self.particles,
            log_dir=self.log_dir,
            inv_metric=self.mass_matrix,
            adapt_engaged=False,
            stepsize=self.stepsize,
        )
        last_position = fit_run.get_last_position()[0]  # select chain 1
        for name in self.param_names:
            self.particles[name] = last_position[name]

    def sample_theta_given_z_with_used_mcmc_params2(self, data):
        mcmc_data = data.copy()
        mcmc_data["zz"] = np.copy(self.latent_mcmc_sample["z"])
        # values_dict = dict()
        # values_dict['alpha'] = np.copy(self.particles['alpha']).reshape(6,)
        # values_dict['beta'] = np.copy(self.particles['beta']).reshape(6,1)
        fit_run = run_mcmc(
            data=mcmc_data,
            sm=self.compiled_model,
            num_samples=5,
            num_warmup=0,
            num_chains=1,
            initial_values = self.particles, 
            log_dir=self.log_dir,
            inv_metric=load_obj('mass_matrix', self.log_dir),
            adapt_engaged=False,
            stepsize=load_obj('stepsize', self.log_dir)
        )
        last_position = fit_run.get_last_position()[0]  # select chain 1
        for name in self.param_names:
            self.particles[name] = last_position[name]
