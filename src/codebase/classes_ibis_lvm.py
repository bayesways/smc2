import numpy as np
from codebase.ibis import (
    compile_model,
    sample_prior_particles, 
    get_resample_index,
    run_mcmc,
    get_initial_values_dict,
    exp_and_normalise
)
from codebase.ibis_tlk_latent import(
    get_weight_matrix_for_particle,
    initialize_bundles,
    get_bundle_weights,
    generate_latent_pair,
    get_weight_matrix_at_datapoint
)
from codebase.classes import Particles
from codebase.file_utils import (
    save_obj,
    load_obj,
    make_folder,
    path_backslash
)
from codebase.resampling_routines import multinomial
from scipy.special import logsumexp
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
        latent_model_num
        ):
        super().__init__(
            name,
            model_num,
            size,
            param_names,
            latent_names
            )
        self.bundle_size = bundle_size
        self.latent_model_num = latent_model_num


    def initialize_bundles(self, data):        
        self.bundles = initialize_bundles(
            self.size,
            self.bundle_size,
            data)


    def initialize_latent_var_given_theta(self, data):        
        self.latent_var_given_theta = np.empty((self.size,
            data['N'],
            data['K']
            )
            )


    def initialize_counter(self, data):
        self.ess = np.zeros(data['N'])
        self.acceptances = np.zeros((
            self.size,
            data['N']
        ))
        self.counts = np.zeros((
            self.size,
            data['N']
        ))


    def add_ess(self, t):
        self.ess[t] = 1


    def get_bundles_at_t(self, t):
        # returns a pointer to current values
        bundles_at_t = dict()
        for name in self.latent_names:
            bundles_at_t[name] = self.bundles[name][:,:,t]
        return bundles_at_t

    
    def get_bundles_upto_t(self, t):
        # returns a pointer to current values
        bundles_at_t = dict()
        for name in self.latent_names:
            bundles_at_t[name] = self.bundles[name][:,:,:t]
        return bundles_at_t


    def generate_latent_bundle(
        self,
        alpha,
        beta,
        data_t):
        bundle = dict()
        zz = np.empty((
            self.bundle_size,
            data_t['K']))
        y_latent = np.empty((
            self.bundle_size,
            data_t['J']))
        for l in range(self.bundle_size):
            bundle_vars = generate_latent_pair(        
                data_t['J'],
                data_t['K'],
                alpha,
                beta)
            zz[l] = bundle_vars['z']
            y_latent[l] = bundle_vars['y']
        bundle['z'] = zz
        bundle['y'] = y_latent
        return bundle


    def sample_latent_bundle_at_t(self, t, data_t):
        for m in range(self.size):
            new_bundle = self.generate_latent_bundle(
                self.particles['alpha'][m],
                self.particles['beta'][m],
                data_t
                )
            for name in self.latent_names:
                self.bundles[name][m, :, t] = new_bundle[name]


    def get_theta_incremental_weights_at_t(self, t, data):
        weights = get_weight_matrix_at_datapoint(
                self.size,
                self.bundle_size,
                data,
                self.get_bundles_at_t(t)['y'])
        self.incremental_weights = weights.mean(axis=1)


    def jitter(self, t, data):
        mcmc_data = data.copy()
        mcmc_data['zz'] = self.latent_var_given_theta[0,:t]
        self.jitter_and_save_mcmc_parms(mcmc_data, 0)
        for m in range(1, self.size):
            mcmc_data['zz'] = self.latent_var_given_theta[m,:t]
            self.jitter_with_used_mcmc_params(mcmc_data, m)


    def resample_particles_bundles(self):
        resample_index = get_resample_index(self.weights, self.size)
        for name in self.param_names:
            self.particles[name] = self.particles[name][resample_index].copy()
        for name in self.latent_names:
            self.bundles[name] = self.bundles[name][resample_index].copy()
        

    def jitter_bundles_and_pick_one(self, data):
        for t in range(data['N']):
            data_t = dict()
            data_t['N'] = 1
            data_t['K'] = data['K']
            data_t['J'] = data['J']
            data_t['D'] = data['D'][t]
            for m in range(self.size):
                self.counts[m,t] += 1
                bundle_star = self.generate_latent_bundle(
                    self.particles['alpha'][m],
                    self.particles['beta'][m],
                    data_t
                    )
                weights_star = get_bundle_weights(
                    self.bundle_size,
                    data_t,
                    bundle_star['y'])
                existing_weights = get_bundle_weights(
                    self.bundle_size,
                    data_t,
                    self.get_bundles_at_t(t)['y'][m])
                ## Accept/Reject Step 
                logdiff = weights_star.mean() - existing_weights.mean()
                u=np.random.uniform()
                if (np.log(u) <= logdiff):
                    self.acceptances[m,t] += 1
                    for name in self.latent_names:
                        self.bundles[name][m, :, t] = bundle_star[name].copy()
                    # pick bundle
                    resample_index = get_resample_index(weights_star, 1)[0].astype(int)
                    self.latent_var_given_theta[m,t] = bundle_star['z'][resample_index].copy()
                else:
                    # pick bundle
                    resample_index = get_resample_index(existing_weights, 1)[0].astype(int)
                    self.latent_var_given_theta[m,t] = self.get_bundles_at_t(t)['z'][m][resample_index].copy()

