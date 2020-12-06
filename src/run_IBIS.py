from codebase.classes import Data, Particles
import argparse
import numpy as np
from codebase.file_utils import (
    save_obj,
    load_obj,
    make_folder,
    path_backslash
)
from codebase.ibis import essl, exp_and_normalise, model_phonebook
from tqdm import tqdm
from scipy.special import logsumexp


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
exp_data = Data(
    name = args.task_handle, 
    model_num = 4, 
    size = 50,
    random_seed = 0
    )
    
exp_data.generate()

model_num = 1
## setup particles
param_names = model_phonebook(model_num)['param_names']
latent_names = model_phonebook(model_num)['latent_names']
particles = Particles(
    name = 'normal',
    model_num = model_num,
    size = 10,
    param_names = param_names,
    latent_names = latent_names)
particles.set_log_dir(log_dir)
if args.gen_model:
    particles.compile_prior_model()
    particles.compile_model()
else:
    particles.load_prior_model()
    particles.load_model()

particles.sample_prior_particles(exp_data.get_stan_data()) # sample prior particles
particles.reset_weights() # set weights to 0
log_lklhds = np.empty(exp_data.size)
degeneracy_limit = 0.5
for t in tqdm(range(exp_data.size)):
    particles.get_incremental_weights(
        exp_data.get_stan_data_at_t(t)
        )
    log_lklhds[t] =  particles.get_loglikelihood_estimate()
    particles.update_weights()
    
    if (essl(particles.weights) < degeneracy_limit * particles.size) and (t+1) < exp_data.size:
        particles.resample_particles()
        particles.jitter(exp_data.get_stan_data_upto_t(t+1))
        particles.reset_weights()
    else:
        particles.update_weights()

save_obj(log_lklhds, 'log_lklhds', log_dir)

print('\n\n')
marg_lklhd = np.exp(logsumexp(log_lklhds))
print('Marginal Likelihood %.5f'%marg_lklhd)

for name in ['alpha', 'Marg_cov']:
    samples = np.squeeze(particles.particles[name])
    w = exp_and_normalise(particles.weights)
    print('\n\nEstimate')
    print(np.round(np.average(samples,axis=0, weights=w),2))
    print('\nRead Data')
    print(np.round(exp_data.raw_data[name],2))
