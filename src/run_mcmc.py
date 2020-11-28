from codebase.classesmcmc import Data, MCMC
import argparse
import numpy as np
from codebase.file_utils import (
    save_obj,
    load_obj,
    make_folder,
    path_backslash
)
from tqdm.notebook import tqdm
from pdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument("-th", "--task_handle",
                    help="hande for task", type=str, default="_")
parser.add_argument("-xdir", "--existing_directory",
                    help="refit compiled model in existing directory",
                    type=str, default=None)
parser.add_argument("-gm", "--gen_model",
                    help="generate model", type=bool, default=0)
parser.add_argument("-model", "--model_num",
                    help="model number", type=int, default=0)
args = parser.parse_args()


if args.existing_directory is None:
    log_dir = make_folder(args.task_handle)  
    print("\n\nCreating new directory: %s" % log_dir)

else:
    log_dir = args.existing_directory
    log_dir = path_backslash(log_dir)
    print("\n\nReading from existing directory: %s" % log_dir)

# generate data
data_sim = 100
exp_data = Data("1factorlogit", 1, 100, 4)
exp_data.generate()
save_obj(exp_data, 'data', log_dir)



param_names = ['beta', 'alpha']
latent_names = ['z', 'y_latent']
particles = MCMC('1factorlogit', 7, 1,  param_names, latent_names, 50, 1)

particles.set_log_dir(log_dir)

if args.gen_model:
    particles.compile_prior_model()
    set_trace()
    particles.compile_model()
else:
    particles.load_prior_model()
    particles.load_model()

particles.sample_prior_particles(exp_data.get_stan_data())

nsim_mcmc = 100
betas = np.empty((nsim_mcmc, 6, 1))
alphas = np.empty((nsim_mcmc, 6))
zs = np.empty((nsim_mcmc, data_sim, 1))
ys = np.empty((nsim_mcmc, data_sim, 6))


for i in tqdm(range(nsim_mcmc)):
    particles.sample_latent_variables(exp_data.get_stan_data())
    particles.get_bundle_weights(exp_data.get_stan_data())
    particles.sample_latent_particles_star(exp_data.get_stan_data())
    particles.sample_latent_var_given_theta(exp_data.get_stan_data())    
    
    zs[i] = particles.latent_mcmc_sample['z']
    ys[i] = particles.latent_mcmc_sample['y_latent']
    
    particles.sample_theta_given_z(exp_data.get_stan_data())
    alphas[i] = particles.particles['alpha']
    betas[i] = particles.particles['beta']
    # set_trace()
    
ps = dict()
ps['alpha'] = alphas
ps['beta'] = betas
ps['z'] = zs
ps['y'] = ys
save_obj(ps, 'mcmc_post_samples', log_dir)
