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
from codebase.mcmc_tlk_latent import gen_latent_weights_master, sample_latent_master
from codebase.file_utils import (
    save_obj,
    load_obj,
    make_folder,
    path_backslash
)
from codebase.resampling_routines import multinomial
from scipy.special import logsumexp
from scipy.stats import norm
import pdb

class MCMC:
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
        self.name = name
        self.model_num = model_num
        self.size = nsim
        self.param_names = param_names
        self.latent_names = latent_names
        self.cloud_size = cloud_size
        self.latent_model_num = latent_model_num


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


    def sample_latent_variables(self):
        latent_vars = dict()

        # zz = np.empty(self.cloud_size)
        # y_latent = np.empty((self.cloud_size, data['J']))
        # for m in range(self.cloud_size):
        zz = norm.rvs(size = self.cloud_size)
        y_latent = np.squeeze(
            self.particles['alpha']
            ) + np.outer(zz, np.squeeze( self.particles['beta']))
        
        latent_vars['z'] = zz
        latent_vars['y_latent'] = y_latent
        self.latent_particles = latent_vars


    def get_latent_weights(self, data):
        self.weights = gen_latent_weights_master(
            self.latent_model_num,
            data,
            self.latent_particles,
            self.cloud_size
            ) 


    def reset_weights(self):
        self.weights = np.zeros(self.cloud_size)
    

    def resample_particles(self, data):
        resample_index = np.empty(data['N'], dtype=int)
        for t in range(data['N']):
            resample_index[t] = get_resample_index(self.weights[t], 1).astype(int)
        samples=dict()
        for name in self.latent_names:
            samples[name] = self.latent_particles[name][resample_index].copy()
        self.latent_particles = samples


    def sample_theta_given_z(self, data):
        mcmc_data = data.copy()
        mcmc_data['z'] = self.latent_particles['z'].copy()

        fit_run = run_mcmc(
            data = mcmc_data,
            gen_model = False,
            model_num = self.model_num,
            num_samples = 10, 
            num_warmup = 1000,
            num_chains = 1,
            log_dir = self.log_dir,
            adapt_engaged=True
            )
        last_position = fit_run.get_last_position()[0] # select chain 1
        for name in self.param_names:
            self.particles[name] = last_position[name]
    

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