import numpy as np
from codebase.ibis import (
    compile_model,
    sample_prior_particles,
    get_resample_index,
    run_mcmc,
    get_initial_values_dict,
    exp_and_normalise,
)
from codebase.classes_mcmc import MCMC
from run_mcmc import run_mcmc_jitter, run_mcmc_jitter_with_used_params
from codebase.ibis_tlk_latent import (
    get_weight_matrix_for_particle,
    initialize_bundles,
    get_bundle_weights,
    generate_latent_pair,
    get_weight_matrix_at_datapoint,
)
from codebase.classes import Particles
from codebase.file_utils import save_obj, load_obj, make_folder, path_backslash
from codebase.resampling_routines import multinomial
from scipy.special import logsumexp
import copy 
from pdb import set_trace


class ParticlesSMC2(Particles):
    def __init__(
        self,
        name,
        model_num,
        size,
        param_names,
        latent_names,
        bundle_size,
        latent_model_num,
        mcmc_nsim,
        mcmc_adapt_nsim,
        hmc_adapt_nsim,
        hmc_post_adapt_nsim,
    ):
        super().__init__(
            name, model_num, size, param_names, latent_names, hmc_adapt_nsim,
            hmc_post_adapt_nsim,)
        self.bundle_size = bundle_size
        self.latent_model_num = latent_model_num
        self.particles = np.empty(size, dtype=MCMC)
        self.mcmc_nsim = mcmc_nsim
        self.mcmc_adapt_nsim = mcmc_adapt_nsim
        
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
                latent_names=self.latent_names,
                bundle_size=self.bundle_size,
                latent_model_num=self.latent_model_num,
            )
            self.particles[m] = theta_m
            self.particles[m].set_log_dir(self.log_dir)
            self.particles[m].compiled_model = self.compiled_model
            self.particles[m].compiled_prior_model = self.compiled_prior_model

    def initialize_bundles(self, data):
        for m in range(self.size):
            bundles = dict()
            bundles["z"] = np.empty((self.bundle_size, data["N"], data["K"]))
            bundles["y"] = np.empty((self.bundle_size, data["N"], data["J"]))
            self.particles[m].bundles = bundles

    def add_ess(self, t):
        self.ess[t, :t+1] += 1

    def get_acceptance_rate_for_particle_m(self, m):
        trials = self.mcmc_nsim
        accs = self.particles[m].acceptance
        return (accs/trials)

    def get_threshold_ess_indicator(self):
        return self.ess.sum(axis=1)>1

    def extract_particles_in_numpy_array(self, name):
        if name in self.param_names:
            dim = list(self.particles[0].particles[name].shape)
        elif name in self.latent_names:
            dim = list(self.particles[0].latent_particles[name].shape)
        else:
            exit
        
        dim.insert(0, self.size)
        dim = tuple(dim)

        extract_array = np.empty(dim)
        if name in self.param_names:
            for m in range(self.size):
                extract_array[m] = self.particles[m].particles[name]
        elif name in self.latent_names:
            for m in range(self.size):
                extract_array[m] = self.particles[m].latent_particles[name]
        else:
            exit
        return extract_array

    def check_particles_are_distinct(self):
        for name in self.param_names:
            ext_part = self.extract_particles_in_numpy_array(name)
            dim  = ext_part.shape
            uniq_dim = np.unique(ext_part, axis=0).shape
            assert dim == uniq_dim
        for name in self.latent_names:
            ext_part = self.extract_particles_in_numpy_array(name)
            dim  = ext_part.shape
            uniq_dim = np.unique(ext_part, axis=0).shape
            assert dim == uniq_dim 
    
    def check_latent_particles_are_distinct(self):
        for name in self.latent_names:
            ext_part = self.extract_particles_in_numpy_array(name)
            dim  = ext_part.shape
            uniq_dim = np.unique(ext_part, axis=0).shape
            assert dim == uniq_dim

    def sample_prior_particles(self, data):
        for m in range(self.size):
            self.particles[m].sample_prior_particles(data)

    def sample_latent_bundle_at_t(self, t, data_t):
        for m in range(self.size):
            # use sample_latent_variables2 function
            latent_vars = self.particles[m].sample_latent_variables2(data_t)

            # so we need to initialize particles if we do it this way 
            latent_particles = dict()
            latent_particles['z'] = np.empty((self.bundle_size, 1, 1))
            latent_particles['y'] = np.empty((self.bundle_size, 1, 6))
            self.particles[m].latent_particles = latent_particles
            
            self.particles[m].latent_particles["z"] = latent_vars['z']
            self.particles[m].latent_particles["y"] = latent_vars['y']

            self.particles[m].bundles["z"][:,t] = latent_vars['z'][:,0]
            self.particles[m].bundles["y"][:,t] = latent_vars["y"][:, 0] 

    def get_theta_incremental_weights_at_t(self, data):
        weights = np.empty(self.size)
        for m in range(self.size):
            self.particles[m].get_bundle_weights(data)
            weights[m] = self.particles[m].weights.mean()
        self.incremental_weights = weights

    def run_jitter_and_learn(self, data, initial_values):
        ps = run_mcmc_jitter(
            stan_data = data,
            mcmc_nsim = self.mcmc_nsim,
            mcmc_adapt_nsim = self.mcmc_adapt_nsim,
            model_num = self.model_num,
            bundle_size = self.bundle_size,
            gen_model = False,
            param_names = self.param_names,
            latent_names = self.latent_names,
            initial_values = initial_values,
            log_dir = self.log_dir,
            hmc_adapt_nsim = self.hmc_adapt_nsim,
            hmc_post_adapt_nsim = self.hmc_post_adapt_nsim,
            name="mcmc",
        )
        return ps

    def run_jitter_from_params(self, data, initial_values, mass_matrix, stepsize):
        ps = run_mcmc_jitter_with_used_params(
            stan_data = data,
            mcmc_nsim = self.mcmc_nsim,
            model_num = self.model_num,
            bundle_size = self.bundle_size,
            gen_model = False,
            param_names = self.param_names,
            latent_names = self.latent_names,
            initial_values = initial_values,
            log_dir = self.log_dir,
            hmc_post_adapt_nsim = self.hmc_post_adapt_nsim,
            mass_matrix=mass_matrix,
            stepsize=stepsize,
            name="mcmc",
        )
        return ps

    def jitter(self, data, t):
    
        jitter_particles = self.run_jitter_and_learn(
            data,
            self.particles[0].particles
            )
        self.particles[0].acceptance = jitter_particles.acceptance.copy()
        self.particles[0].particles = jitter_particles.particles.copy()
        self.particles[0].bundles['z'][:,:t] = np.copy(jitter_particles.latent_particles["z"])
        self.particles[0].bundles['y'][:,:t] = np.copy(jitter_particles.latent_particles["y"])
        self.mass_matrix = jitter_particles.mass_matrix.copy()
        self.stepsize = jitter_particles.stepsize.copy()
        for m in range(1, self.size):
            jitter_particles = self.run_jitter_from_params(
                data,
                self.particles[m].particles,
                self.mass_matrix,
                self.stepsize
                )
            self.particles[m].acceptance = jitter_particles.acceptance.copy()
            self.particles[m].particles = jitter_particles.particles.copy()
            self.particles[m].bundles['z'][:,:t] = np.copy(jitter_particles.latent_particles["z"])
            self.particles[m].bundles['y'][:,:t] = np.copy(jitter_particles.latent_particles["y"])
        # for m in range(self.size):
        #     self.particles[m].sample_theta_given_z_and_save_mcmc_parms2(data)
        # self.particles[0].sample_theta_given_z_and_save_mcmc_parms2(data)
        # for m in range(1, self.size):
        #     self.particles[m].sample_theta_given_z_with_used_mcmc_params2(data)


    def resample_particles_bundles(self):
        resample_index = get_resample_index(self.weights, self.size)
        for i in range(self.size):
            self.particles[i].particles = self.particles[resample_index[i]].particles.copy()
            self.particles[i].latent_particles = self.particles[resample_index[i]].latent_particles.copy()

    def gather_latent_variables_up_to_t(self, t, data):
        for m in range(self.size):
            self.particles[m].latent_particles["z"] = (
                np.copy(self.particles[m].bundles['z'][:,:t])
            )
            self.particles[m].latent_particles["y"] = (
                np.copy(self.particles[m].bundles["y"][:,:t])
            )
            self.particles[m].get_bundle_weights(data)
            #produces self.particles[m].weights with dim (bundlesize, dataN)
