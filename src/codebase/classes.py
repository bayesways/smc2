import numpy as np
from codebase.data import gen_data_master
from codebase.ibis import (
    compile_model,
    sample_prior_particles, 
    get_resample_index,
    run_mcmc,
    get_initial_values_dict,
    exp_and_normalise
)
from codebase.ibis_tlk import gen_weights_master
from codebase.ibis_tlk_latent import gen_latent_weights_master, sample_latent_master
from codebase.file_utils import (
    save_obj,
    load_obj,
    make_folder,
    path_backslash
)
from codebase.resampling_routines import multinomial
from scipy.special import logsumexp

class Particles:
    def __init__(
        self,
        name,
        model_num,
        size,
        param_names,
        latent_names
        ):
        self.name = name
        self.model_num = model_num
        self.size = size
        self.param_names = param_names
        self.latent_names = latent_names


    def set_log_dir(self, log_dir):
        self.log_dir = log_dir


    def load_prior_model(self):
        self.compiled_prior_model = load_obj(
            'sm_prior',
            "./log/compiled_models/model%s/"%self.model_num
            )


    def load_model(self):
        self.compiled_model = load_obj(
            'sm',
            "./log/compiled_models/model%s/"%self.model_num
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
            sm_prior = self.compiled_prior_model,
            param_names = self.param_names,
            num_samples = self.size, 
            num_chains = 1, 
            log_dir = self.log_dir
            )


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
            sm = self.compiled_model,
            num_samples = 20, 
            num_warmup = 1000,
            num_chains = 1, # don't change
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
            sm = self.compiled_model,
            num_samples = 20, 
            num_warmup = 0,
            num_chains = 1, # don't change
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
        if self.model_num != 8:
            self.jitter_and_save_mcmc_parms(data, 0)
            for m in range(1, self.size):
                self.jitter_with_used_mcmc_params(data, m)
        else:
            pass


    def get_loglikelihood_estimate(self):
        return logsumexp(
            self.incremental_weights + self.weights
            ) - logsumexp(self.weights)


class ParticlesLatent(Particles):
    def __init__(
        self,
        name,
        model_num,
        nsim,
        param_names,
        latent_names,
        cloud_size,
        latent_model_num
        ):
        super().__init__(
            name,
            model_num,
            nsim,
            param_names,
            latent_names
            )
        self.cloud_size = cloud_size
        self.latent_model_num = latent_model_num


    def sample_prior_latent_variables(self, data):
            self.latent_particles = sample_latent_master(
                self.latent_model_num, 
                self.particles,
                self.size,
                self.cloud_size, 
                data
            )


    def reset_latent_weights(self):
        self.latent_weights = np.zeros(self.cloud_size)


    def get_latent_weights(self, data):
        self.latent_weights = gen_latent_weights_master(
            self.latent_model_num,
            data,
            self.latent_particles,
            self.size,
            self.cloud_size
            )


    def resample_particles(self):
        # get the index of the resampled particles
        resample_index = get_resample_index(self.weights, self.size)
        # then apply the same index to parameters 
        for name in self.param_names:
            samples = self.particles[name][resample_index].copy()
            self.particles[name] = samples
        # and latent variables 
        for name in self.latent_names:
            samples = self.latent_particles[name][resample_index].copy()
            self.latent_particles[name] = samples


    def sample_latent_y_star(self):
        y_latent_star = np.empty((
            self.size, self.latent_particles['y_latent'].shape[-1]
            ))
        z_star = np.empty(self.size)
        latent_particles_star = dict()
        for m in range(self.size):
            w = exp_and_normalise(self.latent_weights[m])
            nw = w / np.sum(w)
            np.testing.assert_allclose(1., nw.sum())  
            sampled_index = multinomial(nw, 1)
            y_latent_star[m] = self.latent_particles['y_latent'][m, sampled_index]
            z_star[m] = self.latent_particles['z'][m, sampled_index]
        latent_particles_star['y_latent'] = y_latent_star
        latent_particles_star['z'] = z_star
        self.latent_particles_star = latent_particles_star


class Data:
    def __init__(self, name, model_num, size, random_seed=None):
        self.name = name
        self.model_num = model_num
        self.size = size 
        self.random_seed = random_seed

    def generate(self):
        self.raw_data = gen_data_master(
            model_num = self.model_num,
            nsim_data = self.size,
            random_seed = self.random_seed
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