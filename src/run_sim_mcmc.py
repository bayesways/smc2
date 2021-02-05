from codebase.classes_data import Data
from run_mcmc import run_mcmc
import argparse
import numpy as np
from codebase.file_utils import save_obj, load_obj, make_folder, path_backslash
from tqdm.notebook import tqdm
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
exp_data = Data(name=args.task_handle, model_num=1, size=100, random_seed=0)

exp_data.generate()
save_obj(exp_data, "data", log_dir)

param_names = ["beta", "alpha"]
latent_names = ["z", "y"]
ps = run_mcmc(
    stan_data=exp_data.get_stan_data(),
    nsim_mcmc=100,
    num_warmup=10,
    model_num=2,
    bundle_size=10,
    gen_model=args.gen_model,
    param_names=param_names,
    latent_names=latent_names,
    log_dir=log_dir,
    adapt_nsim=100
)

save_obj(ps, "mcmc_post_samples", log_dir)
