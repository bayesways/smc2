# smc2
Sequential Monte Carlo Methods for Factor Analysis

1) HMC code for binary logit models. Notebooks `1.1` and `1.2` contain the running code and the results.  This is done in two different parametrizations: Method1 fixes the leading beta elements to 1; Method 2 fixes the variance of z to 1, so all elements of z get a prior N(0,1).
2) MCMC for binary data that uses the pseudo marginal to learn the latent variable `z`. We ca run MCMC for 1 factor model that will be used for IBIS with 
`run_sim_mcmc.py`, results plots in `2.1 MCMC with pseudomarginal.ipynb`. It includes plotting functions for comparing to real data
3) IBIS for continuous data. This is a general code that works for any Stan model that does not contain latent variables. For example we can run cotinuous factor models using the marginal representation. To run use `run_sim_ibis.py`. The main function is in `run_sim_ibis.py`. There is a notebook to compare directly the results of this IBIS and the standard HMC when applied to simulated continuous factor data (see `3.1 IBIS and HMC  2 factor sim EZ.ipynb`)
4) SMC2 : IBIS with Pseudomarginal for binary data. The main function is in `run_sim_smc2.py` 
5) IBIS-LVM : IBIS with Latent Variables as part of the paremeter vector in `run_sim_smclvm.py` 


### Current results


#### Continuous Data Simulations

Correctness is shown in notebooks 3.3 for the EZ data scenario.
Model comparisons are in notebooks 3.4.2 for EZ and CL data scenario. The saved data are on MAC laptop under `/log/fabian_sim/EZ` and `/log/fabian_sim/CL`.

MISC: There are additional model choice experiments in notebooks  and notebooks 8.1,8.2


#### Binary Data Simulations

Correctness is established in notebook 7.1.  
Comparison of different inference algorithms is done in notebooks 5.2.1-5.2.3. The saved data is on LINUX pc, `/log/IBIS experiments/`.

#### Sensitivity on Priors

Two notebooks on checking the impact of prior choices for the continuous data models. 
The prior chosen in the papers is dependent on data (to avoid Heywood problems). We tested the effect of using a prior
that is not dependent on the data. We ran both the EZ model and the AZ models for the EZ data, both for MCMC and for the sequential methodologies. 

Notebooks `9.1 *.ipynb` contain the results. There are also two notebooks which show the results for MCMC only, that are meant to be used for Bayes SEM paper, see notebooks `9.0 * .ipynb`.

**NOTE** there is a mistake in 9.1 EZ notebooks. I believe the sequential run number 2 (with the independent prior) did not
use the right mode. Needs to re-run. 



### On Fabian

Use the environment `pystan-dev`. To update the environment run `conda env update --file env_fabian.yml`.
The environment works with the anaconda3/5.3.0. If it's not loaded by default you need to:

1. Activate `apps/anaconda3/5.3.0` 
2. Add new compiler `compilers/gcc/10.2.0`
3. Activate env `pystan2`    


    ```
    module add apps/anaconda3/5.3.0
    module add compilers/gcc/10.2.0
    source activate pystan2
    ```

Also remember to activate the environment before submiting jobs to `qsub`.

Submit jobs with constraings of memory (e.g. 50G) and run time(e.g. 120 hours) as follows:

    ```
    qsub-python -l h_vmem=50G,h_rt=120:0:0 run_sim_smc2.py -th test
    ``` 
