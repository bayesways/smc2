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


class ParticlesLVM(Particles):
    def __init__(
        self,
        name,
        model_num,
        size,
        param_names,
        latent_names,
        bundle_size,
        latent_model_num,
    ):
        super().__init__(name, model_num, size, param_names, latent_names)
        self.bundle_size = bundle_size
        self.latent_model_num = latent_model_num
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
                nsim=1,
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

    def initialize_latent_var_given_theta(self, data):
        for m in range(self.size):
            self.particles[m].latent_var_given_theta = np.empty(
                (self.bundle_size, data["N"], data["K"])
            )

    def add_ess(self, t):
        self.ess[t, :t+1] += 1

    def get_acceptance_rate_for_particle_m(self, m):
        trials = self.ess.sum(axis=0)
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
            if dim != uniq_dim :
                return False
            else:
                pass

    # def get_bundles_at_t(self, t):
    #     # returns a pointer to current values
    #     bundles_at_t = dict()
    #     for name in self.latent_names:
    #         bundles_at_t[name] = self.bundles[name][:,:,t]
    #     return bundles_at_t

    # def get_bundles_upto_t(self, t):
    #     # returns a pointer to current values
    #     bundles_at_t = dict()
    #     for name in self.latent_names:
    #         bundles_at_t[name] = self.bundles[name][:,:,:t]
    #     return bundles_at_t

    def sample_prior_particles(self, data):
        for m in range(self.size):
            self.particles[m].sample_prior_particles(data)

    def generate_latent_bundle(self, alpha, beta, data_t):
        bundle = dict()
        zz = np.empty((self.bundle_size, data_t["K"]))
        y_latent = np.empty((self.bundle_size, data_t["J"]))
        for l in range(self.bundle_size):
            bundle_vars = generate_latent_pair(data_t["J"], data_t["K"], alpha, beta)
            zz[l] = bundle_vars["z"]
            y_latent[l] = bundle_vars["y"]
        bundle["z"] = zz
        bundle["y"] = y_latent
        return bundle

    def sample_latent_bundle_at_t(self, t, data_t):
        for m in range(self.size):
            self.particles[m].sample_latent_variables(data_t)
            self.particles[m].bundles["z"][:,t] = np.copy(
                self.particles[m].latent_particles["z"][:, 0]
            )
            self.particles[m].bundles["y"][:,t] = np.copy(
                self.particles[m].latent_particles["y"][:, 0]
            )
            if m>1 and (
            (
                self.particles[0].latent_particles['z'][0,0]
            ) == (
                self.particles[m].latent_particles['z'][0,0]
                )
            ):
                print('got it')
                set_trace()

    def get_theta_incremental_weights_at_t(self, data):
        weights = np.empty(self.size)
        for m in range(self.size):
            self.particles[m].get_bundle_weights(data)
            weights[m] = self.particles[m].weights.mean()
        self.incremental_weights = weights

    def jitter(self, data):
        self.particles[0].sample_theta_given_z_and_save_mcmc_parms2(data)
        for m in range(1, self.size):
            self.particles[m].sample_theta_given_z_with_used_mcmc_params2(data)

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
            
        
    def jitter_bundles_and_pick_one(self, data):
        for m in range(self.size):
            self.particles[m].sample_latent_particles_star(data)
            self.particles[m].sample_latent_var_given_theta(data)
