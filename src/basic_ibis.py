import  pystan
import argparse
import numpy as np
from codebase.file_utils import (
    save_obj,
    load_obj,
    make_folder,
    path_backslash
)
from codebase.data import get_data


def compile_model(model_num, prior, log_dir):
    path_to_stan = './codebase/stancode/'

    if prior: 
        with open('%slogit_%s_prior.stan'%(
        path_to_stan,
        model_num
        ), 'r') as file:
            model_code = file.read()
    else:
        with open('%slogit_%s.stan'%(
        path_to_stan,
        model_num
        ), 'r') as file:
            model_code = file.read()

    sm = pystan.StanModel(model_code=model_code, verbose=False)
    
    if prior:
        save_obj(sm, 'sm_prior', log_dir)
    else:
        save_obj(sm, 'sm', log_dir)
    return sm


def sample_prior_particles(
    data,
    gen_model,
    model_num,
    param_names,
    num_samples, 
    num_warmup,
    num_chains, 
    log_dir
    ):

    if gen_model:
        sm_prior = compile_model(model_num, True, log_dir)
        save_obj(sm_prior, 'sm_prior', log_dir)
    else:
        sm_prior = load_obj('sm_prior', log_dir)
    
    param_names = ['z', 'alpha', 'Marg_cov']

    fit_run = sm_prior.sampling(
        data={
            'N':data['N'],
            'J': data['J'],
            'y' : data['y']
        },
        iter=num_samples + num_warmup,
        warmup=num_warmup,
        chains=num_chains,
        algorithm = 'Fixed_param',
        n_jobs=1
    )
    particles = fit_run.extract(
        permuted=False, pars=param_names)

    save_obj(particles, 'particles', log_dir)

    return particles


def run_mcmc(
    data,
    gen_model,
    model_num,
    param_names,
    num_samples, 
    num_warmup,
    num_chains, 
    log_dir
    ):

    if gen_model:
        sm = compile_model(model_num, False, log_dir)
        save_obj(sm, 'sm', log_dir)
    else:
        sm = load_obj('sm', log_dir)
    
    param_names = ['z', 'alpha', 'Marg_cov']

    fit_run = sm.sampling(
        data={
            'N':data['N'],
            'J': data['J'],
            'y' : data['y']
        },
        iter=num_samples + num_warmup,
        warmup=num_warmup,
        chains=num_chains,
        n_jobs=1
    )
    particles = fit_run.extract(
        permuted=False, pars=param_names)

    save_obj(particles, 'particles', log_dir)

    return particles


parser = argparse.ArgumentParser()
parser.add_argument("-th", "--task_handle",
                    help="hande for task", type=str, default="_")
parser.add_argument("-xdir", "--existing_directory", help="refit compiled model in existing directory",
                    type=str, default=None)
parser.add_argument("-gm", "--gen_model",
                    help="generate model", type=bool, default=False)
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

data = get_data(50, 6, 1)

param_names = ['z', 'alpha', 'Marg_cov']
num_samples = 100
num_warmup = 0
num_chains = 1
gm = args.gen_model

# particles = sample_prior_particles(
#     data,
#     args.gen_model,
#     0,
#     param_names,
#     100, 
#     0, 
#     1,
#     log_dir
#     )



data_temp = data.copy()
data_temp['y'] = data['y'][:1]
data_temp['N'] = 1

import pdb; pdb.set_trace()


particles = run_mcmc(
    data_temp,
    args.gen_model,
    0,
    param_names,
    100, 
    100, 
    1,
    log_dir
    )

