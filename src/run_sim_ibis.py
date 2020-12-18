from codebase.classes import Data
import argparse
import numpy as np
from codebase.file_utils import (
    save_obj,
    load_obj,
    make_folder,
    path_backslash
)
from codebase.ibis import exp_and_normalise
from run_ibis import run_ibis
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
    size = 10,
    random_seed = 0
    )
    
exp_data.generate()

model_num = 1

ibis = run_ibis(
    exp_data,
    model_num,
    10,
    args.gen_model,
    log_dir
    )

for name in ['alpha', 'Marg_cov']:
    samples = np.squeeze(ibis['particles'].particles[name])
    w = exp_and_normalise(ibis['particles'].weights)
    print('\n\nEstimate')
    print(np.round(np.average(samples,axis=0, weights=w),2))
    # print('\nRead Data')
    # print(np.round(exp_data.raw_data[name],2))
