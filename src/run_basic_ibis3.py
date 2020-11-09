import  pystan
import argparse
import numpy as np
from codebase.file_utils import (
    save_obj,
    load_obj,
    make_folder,
    path_backslash
)
from tqdm import tqdm
from codebase.data import gen_factor_data
from codebase.ibis import (
    compile_model,
    run_mcmc,
    sample_prior_particles,
    essl,
    jitter,
    resample_particles
)
from codebase.ibis_tlk2 import (
    get_weights
)
from scipy.special import logsumexp
import pdb
    
parser = argparse.ArgumentParser()
parser.add_argument("-th", "--task_handle",
                    help="hande for task", type=str, default="_")
parser.add_argument("-xdir", "--existing_directory", help="refit compiled model in existing directory",
                    type=str, default=None)
parser.add_argument("-gm", "--gen_model",
                    help="generate model", type=bool, default=0)
parser.add_argument("-model", "--model_num",
                    help="model number", type=int, default=0)
args = parser.parse_args()

############################################################
###### Create Directory or Open existing ##########
if args.existing_directory is None:
    log_dir = make_folder(args.task_handle)  
    print("\n\nCreating new directory: %s" % log_dir)

else:
    log_dir = args.existing_directory
    log_dir = path_backslash(log_dir)
    print("\n\nReading from existing directory: %s" % log_dir)

nsim_data = 20
data = gen_factor_data(nsim_data)
save_obj(data, 'data', log_dir)

param_names = ['Marg_cov', 'beta', 'alpha', 'sigma', 'Theta', 'Phi_cov']
nsim_particles = 100


if args.gen_model:
    compile_model(
        model_num=args.model_num,
        prior = False,
        log_dir = log_dir
        )


particles = sample_prior_particles(
    data = data,
    gen_model = args.gen_model,
    model_num = args.model_num,
    param_names = param_names,
    num_samples = nsim_particles, 
    num_chains = 1, 
    log_dir = log_dir
    )


particles['M'] = nsim_particles
particles['param_names'] = param_names
particles['latent_var_names'] = []
particles['names'] = particles['latent_var_names'] + particles['param_names']
particles['w'] = np.zeros(nsim_particles)
log_weights = particles['w']


log_lklhds = np.empty(nsim_data)
degeneracy_limit = 0.5
for t in tqdm(range(nsim_data)):
    log_incr_weights = get_weights(data['y'][t], particles)
    log_lklhds[t] =  logsumexp(log_incr_weights + log_weights) - logsumexp(log_weights)
    log_weights = log_incr_weights + log_weights
    
    if (essl(log_weights) < degeneracy_limit * particles['M']) and (t+1) < data['N']:
        data_temp = dict()
        data_temp['J'] = data['J']
        data_temp['y'] = data['y'][:(t+1)].copy()
        data_temp['N'] = t+1
        print("Deg %d"%(t))
        particles = resample_particles(particles)
        
        particles = jitter(data_temp, particles, log_dir)
        particles['w'] = np.zeros(particles['M'])
    else:
        particles['w'] = log_weights
    save_obj(particles, 'particles%s'%(t+1), log_dir)
    
save_obj(log_lklhds, 'log_lklhds', log_dir)

marg_lklhd = np.exp(logsumexp(log_lklhds))
print('Marginal Likelihood %.5f'%marg_lklhd)
