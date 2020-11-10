import numpy as np
from codebase.data import gen_data_master
from codebase.ibis import (
    compile_model,
    sample_prior_particles, 
    get_resample_index,
    run_mcmc,
    get_initial_values_dict
)
from codebase.ibis_tlk import gen_weights_master
from codebase.file_utils import (
    save_obj,
    load_obj,
    make_folder,
    path_backslash
)
from scipy.special import logsumexp

class Particles:


    def __init__(
        self,
        name,
        model_num,
        nsim,
        param_names,
        latent_names
        ):
        self.name = name
        self.model_num = model_num
        self.size = nsim
        self.param_names = param_names
        self.latent_names = latent_names


    def set_log_dir(self, log_dir):
        self.log_dir = log_dir


    def load_prior_model(self):
        self.compiled_prior_model = load_obj(
            'sm_prior',
            self.log_dir
            )


    def load_model(self):
        self.compiled_model = load_obj(
            'sm',
            self.log_dir
            )

    def compile_prior_model(self):
        self.compiled_prior_model = compile_model(
            model_num=self.model_num,
            prior = True,
            log_dir = self.log_dir,
            save = True
            )


    def compile_model(self):
            self.compiled_model = compile_model(
                model_num=self.model_num,
                prior = False,
                log_dir = self.log_dir,
                save = True
                )


    def sample_prior_particles(self, data):
        self.particles = sample_prior_particles(
            data = data,
            gen_model = False,
            model_num = self.model_num,
            param_names = self.param_names,
            num_samples = self.size, 
            num_chains = 1, 
            log_dir = self.log_dir
            )


    def sample_prior_latent_variables(self):
        pass
        # self.latent_variables = 


    def reset_weights(self):
        self.weights = np.zeros(self.size)


    def get_incremental_weights(self, data):
        self.incremental_weights = gen_weights_master(
            self.model_num,
            data,
            self.particles,
            self.size
            )


    def update_weights(self):
        self.weights = self.weights + self.incremental_weights
    

    def resample_particles(self):
        resample_index = get_resample_index(self.weights, self.size)
        # print(resample_index)
        for name in self.param_names:
            samples = self.particles[name][resample_index].copy()
            self.particles[name] = samples


    def get_particles_at_position_m(self, m):
        values_dict = dict()
        for name in self.param_names:
            values_dict[name] = np.squeeze(
                self.particles[name][m].copy()
            )
        return values_dict


    def jitter_and_save_mcmc_parms(self, data, m=0):
        fit_run = run_mcmc(
            data = data,
            gen_model = False,
            model_num = self.model_num,
            num_samples = 20, 
            num_warmup = 1000,
            num_chains = 1,
            log_dir = self.log_dir,
            initial_values = self.get_particles_at_position_m(m),
            load_inv_metric= False, 
            adapt_engaged = True
            )

        self.mass_matrix = fit_run.get_inv_metric(as_dict=True)
        self.stepsize = fit_run.get_stepsize()
        last_position = fit_run.get_last_position()[0] # select chain 1
        for name in self.param_names:
            self.particles[name][m] = last_position[name]


    def jitter_with_used_mcmc_params(self, data, m):        
        fit_run = run_mcmc(
            data = data,
            gen_model = False,
            model_num = self.model_num,
            num_samples = 20, 
            num_warmup = 0,
            num_chains = 1,
            log_dir = self.log_dir,
            initial_values = self.get_particles_at_position_m(m),
            inv_metric= self.mass_matrix,
            adapt_engaged=False,
            stepsize = self.stepsize
            )
        last_position = fit_run.get_last_position()[0] # select chain 1

        for name in self.param_names:
            self.particles[name][m] = last_position[name]


    def jitter(self, data):
        self.jitter_and_save_mcmc_parms(data, 0)
        for m in range(1, self.size):
            self.jitter_with_used_mcmc_params(data, m)


    def get_loglikelihood_estimate(self):
        return logsumexp(
            self.incremental_weights + self.weights
            ) - logsumexp(self.weights)


class Data:
    def __init__(self, name, model_num, nsim, random_seed=None):
        self.name = name
        self.model_num = model_num
        self.size = nsim 
        self.random_seed = random_seed

    def generate(self):
        self.raw_data = gen_data_master(
            self.model_num,
            self.size
            )

    def get_stan_data(self):
        stan_data = dict()
        for name in self.raw_data['stan_constants']:
            stan_data[name] = self.raw_data[name]
        for name in self.raw_data['stan_data']:
            stan_data[name] = self.raw_data[name]
        return stan_data

    def get_stan_data_at_t(self, t):
        stan_data = dict()
        for name in self.raw_data['stan_constants']:
            stan_data[name] = self.raw_data[name]
        stan_data['N'] = 1
        for name in self.raw_data['stan_data']:
            stan_data[name] = self.raw_data[name][t]
        return stan_data

    def get_stan_data_upto_t(self, t):
        stan_data = dict()
        for name in self.raw_data['stan_constants']:
            stan_data[name] = self.raw_data[name]
        stan_data['N'] = t
        for name in self.raw_data['stan_data']:
            stan_data[name] = self.raw_data[name][:t]
        return stan_data