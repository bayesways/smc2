import  pystan
import argparse
import numpy as np
from codebase.file_utils import (
    save_obj,
    load_obj,
    make_folder,
    path_backslash
)

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


path_to_stan = './codebase/stancode/demofiles/'

with open(path_to_stan+'test.stan', 'r') as file:
    model_code = file.read()

param_names = ['alpha', 'beta']

if args.gen_model:
    sm = pystan.StanModel(model_code=model_code, verbose=False)
    save_obj(sm, 'sm', log_dir)

else:
    sm = load_obj('sm', log_dir)
    param_names = ['alpha', 'beta']

num_samples = 10
num_warmup = 10
num_chains = 1
fit_run = sm.sampling(
    data={'N':10, 'x':np.ones(10)},
    iter=num_samples + num_warmup,
    warmup=num_warmup,
    chains=num_chains,
    algorithm = 'Fixed_param',
    n_jobs=1
)
samples = fit_run.extract(
    permuted=False, pars=param_names)

save_obj(samples, 'ps', log_dir)
