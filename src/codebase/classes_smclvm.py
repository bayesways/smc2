import numpy as np
from codebase.ibis import (
    compile_model,
    sample_prior_particles,
    get_resample_index,
    run_mcmc,
    get_initial_values_dict,
    exp_and_normalise,
)
from scipy.stats import bernoulli
from scipy.special import expit, logsumexp
from codebase.classes_mcmc import MCMC
from run_mcmc import run_mcmc_jitter, run_mcmc_jitter_with_used_params
from codebase.ibis_tlk_latent import (
    get_weight_matrix_for_particle,
    initialize_latentvars,
    get_bundle_weights,
    generate_latent_pair,
    get_weight_matrix_at_datapoint,
)
from codebase.classes import Particles
from codebase.file_utils import save_obj, load_obj, make_folder, path_backslash
from codebase.resampling_routines import multinomial
import copy 
from pdb import set_trace


class PariclesSMCLVM(Particles):
    def __init__(
        self,
        name,
        model_num,
        size,
        param_names,
        latent_names,
        latent_model_num,
        hmc_adapt_nsim,
        hmc_post_adapt_nsim,
    ):
        super().__init__(
            name, model_num, size, param_names, latent_names,
            hmc_adapt_nsim,
            hmc_post_adapt_nsim)
        self.particles = np.empty(size, dtype=MCMC)

    def initialize_counter(self, data):
        self.ess = np.zeros((data["N"], data["N"]))
        self.acceptances = np.zeros((self.size, data["N"]))
        self.counts = np.zeros((self.size, data["N"]))

    def initialize_particles(self):
        for m in range(self.size):
            theta_m = MCMC(
                name=self.name,
                model_num=self.model_num,
                param_names=self.param_names,
                latent_names=self.latent_names
            )
            self.particles[m] = theta_m
            self.particles[m].set_log_dir(self.log_dir)
            self.particles[m].compiled_model = self.compiled_model
            self.particles[m].compiled_prior_model = self.compiled_prior_model

    # def initialize_latentvars(self, data):
    #     latentvars = dict()
    #     latentvars["z"] = np.empty((self.size, data["N"], data["K"]))
    #     latentvars["y"] = np.empty((self.size, data["N"], data["J"]))
    #     self.particles.latentvars = latentvars

    def add_ess(self, t):
        self.ess[t, :t+1] += 1

    def get_acceptance_rate_for_particle_m(self, m):
        trials = self.mcmc_nsim
        accs = self.particles[m].acceptance
        return (accs/trials)

    def get_threshold_ess_indicator(self):
        return self.ess.sum(axis=1)>1

    def set_latentvars_shape(self, data): 
        self.particles['zz'] = np.empty((self.size, data["N"], data["K"]))
        self.particles['yy'] = np.empty((self.size, data["N"], data["J"]))

        
    # def extract_particles_in_numpy_array(self, name):
    #     if name in self.param_names:
    #         dim = list(self.particles[name].shape)
    #     elif name in self.latent_names:
    #         dim = list(self.particles.latentvars[name].shape)
    #     else:
    #         exit
        
    #     dim.insert(0, self.size)
    #     dim = tuple(dim)

    #     extract_array = np.empty(dim)
    #     if name in self.param_names:
    #         for m in range(self.size):
    #             extract_array = self.particles[m].particles[name]
    #     elif name in self.latent_names:
    #         for m in range(self.size):
    #             extract_array[m] = self.particles[m].latent_particles[name]
    #     else:
    #         exit
    #     return extract_array

    def check_particles_are_distinct(self):
        for name in self.param_names:
            ext_part = self.particles[name]
            dim  = ext_part.shape
            uniq_dim = np.unique(ext_part, axis=0).shape
            assert dim == uniq_dim
        # for name in self.latent_names:
        #     set_trace()
        #     ext_part = self.particles.latentvars[name]
        #     dim  = ext_part.shape
        #     uniq_dim = np.unique(ext_part, axis=0).shape
        #     assert dim == uniq_dim 
    
    def check_latent_particles_are_distinct(self):
        for name in self.latent_names:
            ext_part = self.extract_particles_in_numpy_array(name)
            dim  = ext_part.shape
            uniq_dim = np.unique(ext_part, axis=0).shape
            assert dim == uniq_dim

    def compute_weights_at_point(self, yy, size, datapoint):
        weights = np.empty(size)
        for m in range(size):
            weights[m] = bernoulli.logpmf(
                datapoint,
                p=expit(yy[m].mean(axis=0))
            ).sum()
        return weights

    def get_theta_incremental_weights(self, data):
        self.incremental_weights = self.compute_weights_at_point(
            self.particles['yy'],
            self.size,
            data['D']
            )

    def jitter_and_save_mcmc_parms(self, data, m=0):
        initial_values = dict()
        initial_values['beta'] = self.get_particles_at_position_m(m)['beta']
        initial_values['alpha'] = self.get_particles_at_position_m(m)['alpha']
        fit_run = run_mcmc(
            data = data,
            sm = self.compiled_model,
            num_samples = self.hmc_post_adapt_nsim, #normally 20
            num_warmup = self.hmc_adapt_nsim, #normally 1000
            num_chains = 1, # don't change
            log_dir = self.log_dir,
            initial_values = initial_values,
            load_inv_metric= False, 
            adapt_engaged = True
            )
        self.mass_matrix = fit_run.get_inv_metric(as_dict=True)
        self.stepsize = fit_run.get_stepsize()
        last_position = fit_run.get_last_position()[0] # select chain 1
        for name in self.param_names:
            self.particles[name][m] = last_position[name]


    def jitter_with_used_mcmc_params(self, data, m):
        initial_values = dict()
        initial_values['beta'] = self.get_particles_at_position_m(m)['beta']
        initial_values['alpha'] = self.get_particles_at_position_m(m)['alpha']
        fit_run = run_mcmc(
            data = data,
            sm = self.compiled_model,
            num_samples = self.hmc_adapt_nsim, # normally 20 
            num_warmup = 0,
            num_chains = 1, # don't change
            log_dir = self.log_dir,
            initial_values = initial_values,
            inv_metric= self.mass_matrix,
            adapt_engaged=False,
            stepsize = self.stepsize
            )
        last_position = fit_run.get_last_position()[0] # select chain 1
        for name in self.param_names:
            self.particles[name][m] = last_position[name]

    def jitter(self, data):
        self.set_latentvars_shape(data)
        self.jitter_and_save_mcmc_parms(data, 0)
        for m in range(1, self.size):
            self.jitter_with_used_mcmc_params(data, m)