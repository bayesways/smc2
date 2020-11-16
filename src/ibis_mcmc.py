from codebase.classesmcmc import Data, MCMC
import  pystan
import argparse
import numpy as np
from codebase.file_utils import (
    save_obj,
    load_obj,
    make_folder,
    path_backslash
)
from codebase.ibis import essl, exp_and_normalise, run_mcmc
from scipy.special import expit
from tqdm.notebook import tqdm
data_sim = 100
exp_data = Data("1factor", 1, data_sim)
exp_data.generate()


log_dir = './log/debug/'

param_names = ['beta', 'alpha']
latent_names = ['z', 'y_latent']
particles = MCMC('1factorlogit', 7, 1,  param_names, latent_names, 1000, 1)

particles.set_log_dir(log_dir)


# particles.compile_prior_model()
# particles.compile_model()
particles.load_prior_model()
particles.load_model()

particles.sample_prior_particles(exp_data.get_stan_data())

nsim_mcmc = 500
betas = np.empty((nsim_mcmc, 6))
alphas = np.empty((nsim_mcmc, 6))
zs = np.empty((nsim_mcmc, data_sim))
ys = np.empty((nsim_mcmc, data_sim, 6))

for i in tqdm(range(nsim_mcmc)):
    particles.sample_latent_variables()
    particles.get_latent_weights(exp_data.get_stan_data())
    particles.resample_particles(exp_data.get_stan_data())
    zs[i] = particles.latent_particles['z']
    ys[i] = particles.latent_particles['y_latent']
    
    particles.sample_theta_given_z(exp_data.get_stan_data())
    alphas[i] = particles.particles['alpha']
    betas[i] = particles.particles['beta']
    
ps = dict()
ps['alpha'] = alphas
ps['beta'] = betas
ps['z'] = zs
ps['y'] = ys
save_obj(ps, 'mcmc_post_samples', log_dir)



log_dir = './log/mcmc_hmc_test/'
ps = load_obj('mcmc_post_samples', log_dir)


import altair as alt
import pandas as pd
alt.data_transformers.disable_max_rows()

plot_data = pd.DataFrame(ps['alpha']) 
plot_data['id'] = np.arange(len(plot_data))
plot_data = plot_data.melt(id_vars=['id'], var_name=['col'])

mcmc_chart = alt.Chart(plot_data).mark_line(
    opacity = 1,
    strokeWidth = 1,
).encode(
    alt.Y('value', title=None),
    alt.X('id:O',
          title=None
         )
).properties(width=200, height=100)

(mcmc_chart).facet(
    alt.Facet('col'),
    columns=3
)


# ## Run full HMC Algorithm

exp_data = Data("1factor", 1, 200)
exp_data.generate()


log_dir = 'log/debugmcmc/'
fit_run = run_mcmc(
    data = exp_data.get_stan_data(),
    gen_model = False,
    model_num = 0,
    num_samples = 1000, 
    num_warmup = 1000,
    num_chains = 1,
    log_dir = log_dir,
    adapt_engaged=True
    )

param_names = ['beta', 'alpha', 'yy']
particles = fit_run.extract(
        permuted=False, pars=param_names)
save_obj(particles, 'ps', log_dir)



log_dir = 'log/debugmcmc/'
particles = load_obj('ps', log_dir)


np.round(np.mean(exp_data.raw_data['y'], 0), 2)



samples = np.squeeze(particles['yy'])
np.round(np.mean(np.mean(samples,axis=0),axis=0), 2)

for name in ['beta']:
    samples = np.squeeze(particles[name])
    print('\n\nEstimate %s'%name)
    print(np.round(np.mean(expit(samples),axis=0),2))
    print('\nRead Data')
    print(np.round(expit(exp_data.raw_data[name]), 2))


plot_data2 = pd.DataFrame(np.squeeze(particles['alpha'])) 
plot_data2['id'] = np.arange(len(plot_data2))
plot_data2 = plot_data2.melt(id_vars=['id'], var_name=['col'])

hmc_chart = alt.Chart(plot_data2).mark_line(
    opacity = 1,
    strokeWidth = 1,
    color='red'
).encode(
    alt.Y('value', title=None),
    alt.X('id:O',
          title=None
         )
).properties(width=200, height=100)

(hmc_chart).facet(
    alt.Facet('col'),
    columns=3
)


plot_data['method'] = 'mcmc1'
plot_data2['method'] = 'hmc2'
merge_plot_data = pd.concat([plot_data, plot_data2])


merge_chart = alt.Chart(merge_plot_data).mark_line(
    opacity = 0.6,
    strokeWidth = 1,
    color='red'
).encode(
    alt.Y('value', title=None),
    alt.X('id:O',
          title=None
         ), 
    alt.Color('method')
).properties(width=200, height=100)

(merge_chart).facet(
    alt.Facet('col'),
    columns=3
)
