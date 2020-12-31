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
    gen_latent_weights_master,
    generate_latent_variables_bundle,
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


    def sample_latent_variables(self, data):
        latent_bundles_z = np.empty((
            self.size,
            self.bundle_size, 
            data['N'],
            data['K'] 
            ))
        latent_bundles_y = np.empty((
            self.size,
            self.bundle_size, 
            data['N'],
            data['J']
            ))
        for m in range(self.size):
            latent_vars = generate_latent_variables_bundle(
                self.bundle_size,
                data['N'],
                data['J'],
                data['K'], # number of factors
                self.particles['alpha'][m],
                self.particles['beta'][m]
            )
            latent_bundles_z[m] = latent_vars['z']
            latent_bundles_y[m] = latent_vars['y_latent']
        self.latent_bundles_z = latent_bundles_z
        self.latent_bundles_y = latent_bundles_y


    def get_bundle_weights(self, data):
        latent_weights = np.empty((
            self.size, 
            self.bundle_size,
            data['N']
        )
        )
        for m in range(self.size):
            latent_weights[m] = gen_latent_weights_master(
                self.latent_model_num,
                data,
                self.latent_bundles_y[m],
                self.bundle_size
                ) 
        self.latent_weights = latent_weights


    def jitter(self, data):
        mcmc_data = data.copy()
        mcmc_data['zz'] = self.latent_bundles_z_mcmc_sample[0].copy() 
        self.jitter_and_save_mcmc_parms(mcmc_data, 0)
        for m in range(1, self.size):
            mcmc_data['zz'] = self.latent_bundles_z_mcmc_sample[m].copy()
            self.jitter_with_used_mcmc_params(mcmc_data, m)


    def sample_latent_particles_star(self, data):
        for m in range(self.size):
            latent_var_star = generate_latent_variables_bundle(
                self.bundle_size,
                data['N'],
                data['J'],
                data['K'], # number of factors
                self.particles['alpha'][m],
                self.particles['beta'][m])
            weights_star = gen_latent_weights_master(
                self.latent_model_num,
                data,
                latent_var_star['y_latent'],
                self.bundle_size
            )

            ## Accept/Reject Step 
            logdiff = weights_star.mean(axis=0)-self.latent_weights[m].mean(axis=0)
            for t in range(data['N']):
                u=np.random.uniform()
                if (np.log(u) <= logdiff[t]):
                    # self.acceptance[t] += 1
                    self.latent_bundles_z[m, :,t] = latent_var_star['z'][:,t].copy()
                    self.latent_bundles_y[m, :,t] = latent_var_star['y_latent'][:,t].copy()
                    self.latent_weights[m, :, t] = weights_star[:, t].copy()


    def sample_latent_var_given_theta(self, data):
        self.latent_bundles_z_mcmc_sample = self.latent_bundles_z[:,0,:].copy()
        self.latent_bundles_y_mcmc_sample = self.latent_bundles_y[:,0,:].copy()
        for m in range(self.size):
            resample_index = np.empty(data['N'], dtype=int)
            for t in range(data['N']):
                resample_index[t] = get_resample_index(self.latent_weights[m, :,t], 1).astype(int)
            self.latent_bundles_z_mcmc_sample[m] = self.latent_bundles_z[m, resample_index, np.arange(data['N'])].copy()
            self.latent_bundles_y_mcmc_sample[m] = self.latent_bundles_y[m, resample_index, np.arange(data['N'])].copy()
            