# smc2
Sequential Monte Carlo Methods for Factor Analysis

1) HMC code for binary logit models. Notebooks `1.1` and `1.2` contain the running code and the results.  This is done in two different parametrizations: Method1 fixes the leading beta elements to 1; Method 2 fixes the variance of z to 1, so all elements of z get a prior N(0,1).
2) MCMC for binary data that uses the pseudo marginal to learn the latent variable `z`. We ca run MCMC for 1 factor model that will be used for IBIS with 
`run_sim_mcmc.py`, results plots in `2.1 MCMC with pseudomarginal - 1 factor model.ipynb`. It includes plotting functions for comparing to real data
3) IBIS for continuous data. This is a general code that works for any Stan model that does not contain latent variables. For example we can run cotinuous factor models using the marginal representation. To run use `run_sim_ibis.py`. The main function is in `run_ibis.py`. There is a notebook to compare directly the results of this IBIS and the standard HMC when applied to simulated continuous factor data (see `3.1 IBIS and HMC  2 factor sim EZ.ipynb`)
