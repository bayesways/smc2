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
from codebase.data import get_data
from codebase.ibis import (
    compile_model,
    run_mcmc,
    sample_prior_particles,
    get_weights,
    ESS,
    jitter,
    multinomial_sample_particles
)
import pdb
    
parser = argparse.ArgumentParser()
parser.add_argument("-th", "--task_handle",
                    help="hande for task", type=str, default="_")
parser.add_argument("-xdir", "--existing_directory", help="refit compiled model in existing directory",
                    type=str, default=None)
parser.add_argument("-gm", "--gen_model",
                    help="generate model", type=bool, default=0)
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
data = get_data(nsim_data, 6, 1)
save_obj(data, 'data', log_dir)

param_names = ['alpha', 'Marg_cov', 'L_R']
nsim_particles = 100
num_warmup = 0
num_chains = 1



if args.gen_model:
    compile_model(
        model_num=0,
        prior = False,
        log_dir = log_dir
        )

particles = sample_prior_particles(
    data = data,
    gen_model = args.gen_model,
    model_num = 0,
    param_names = param_names,
    num_samples = nsim_particles, 
    num_chains = 1, 
    log_dir = log_dir
    )

particles['M'] = nsim_particles
particles['param_names'] = ['alpha', 'Marg_cov', 'L_R']
particles['latent_var_names'] = ['z']
particles['names'] = particles['latent_var_names'] + particles['param_names']
particles['w'] = np.zeros(nsim_particles)
log_weights = particles['w']


degeneracy_limit = 0.5
for t in tqdm(range(nsim_data)):
    log_incr_weights = get_weights(data['y'][t], particles)
    log_weights = log_incr_weights + log_weights
    
    if (ESS(log_weights) < degeneracy_limit * particles['M']) and (t+1) < data['N']:
        data_temp = dict()
        data_temp['J'] = data['J']
        data_temp['y'] = data['y'][:(t+1)].copy()
        data_temp['N'] = t+1
        print("Deg %d"%(t))
        particles = multinomial_sample_particles(particles, np.exp(log_weights))
        
        particles = jitter(data_temp, particles, log_dir)
        particles['w'] = np.zeros(particles['M'])
    else:
        particles['w'] = log_weights

    save_obj(particles, 'particles%s'%(t+1), log_dir)

