from codebase.classes_data import Data
import argparse
import numpy as np
from codebase.file_utils import save_obj, load_obj, make_folder, path_backslash
from codebase.ibis import exp_and_normalise
from run_smclvm import run_smclvm
from scipy.special import logsumexp
from pdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument(
    "-th", "--task_handle", help="hande for task", type=str, default="_"
)
parser.add_argument(
    "-xdir",
    "--existing_directory",
    help="refit compiled model in existing directory",
    type=str,
    default=None,
)
parser.add_argument("-gm", "--gen_model", help="generate model", type=bool, default=0)
parser.add_argument("-model", "--model_num", help="model number", type=int, default=0)
args = parser.parse_args()


if args.existing_directory is None:
    log_dir = make_folder(args.task_handle)
    print("\n\nCreating new directory: %s" % log_dir)

else:
    log_dir = args.existing_directory
    log_dir = path_backslash(log_dir)
    print("\n\nReading from existing directory: %s" % log_dir)

# generate data
exp_data = Data(name=args.task_handle, model_num=1, size=50, random_seed=2)

exp_data.generate()
save_obj(exp_data, "data", log_dir)
model_num = 1

smclvm = run_smclvm(exp_data, model_num, 12, args.gen_model, log_dir)
