from codebase.classes_data import Data
import argparse
import numpy as np
from codebase.file_utils import (
    save_obj,
    load_obj,
    make_folder,
    path_backslash
)
from codebase.ibis import exp_and_normalise
from run_ibis_batch import run_ibis_batch1, run_ibis_batch2
from pdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument("-th", "--task_handle",
                    help="hande for task", type=str, default="_")
parser.add_argument("-xdir", "--existing_directory",
                    help="refit compiled model in existing directory",
                    type=str)                
parser.add_argument("-p", "--part_num",
                    help="part number", type=str, default="1")                    
parser.add_argument("-gm", "--gen_model",
                    help="generate model", type=bool, default=0)
parser.add_argument("-model", "--model_num",
                    help="model number", type=int, default=0)
args = parser.parse_args()


if args.existing_directory is None:
    log_dir = make_folder(args.task_handle+'_m'+str(args.model_num))  
    print("\n\nCreating new directory: %s" % log_dir)
else:
    log_dir = args.existing_directory
    log_dir = path_backslash(log_dir)
    print("\n\nReading from existing directory: %s" % log_dir)

data_model_num='big5_batch'
# generate data
exp_data = Data(
    name = args.task_handle, 
    model_num = data_model_num,  
    size = 677,
    random_seed = 0    
    )
    
exp_data.generate()
save_obj(exp_data, 'data', log_dir)

if args.part_num == '1':
    ibis = run_ibis_batch1(
        exp_data,
        args.model_num,
        500,
        args.gen_model,
        log_dir
        )

elif args.part_num == '2':
    ibis = run_ibis_batch2(
        exp_data = exp_data,  
        model_num = args.model_num,
        data_end = 400,
        log_dir = log_dir,
        )

elif args.part_num == '3':
    ibis = run_ibis_batch2(
        exp_data = exp_data,  
        model_num = args.model_num,
        data_end = exp_data.size,
        log_dir = log_dir,
        )



for name in ['alpha', 'Marg_cov']:
    samples = np.squeeze(ibis['particles'].particles[name])
    w = exp_and_normalise(ibis['particles'].weights)
    print('\n\nEstimate')
    print(np.round(np.average(samples,axis=0, weights=w),2))
    # print('\nRead Data')
    # print(np.round(exp_data.raw_data[name],2))
