import numpy as np
from numpy.linalg import inv, cholesky
from scipy.stats import bernoulli, multivariate_normal, norm
from scipy.special import expit, logsumexp
from scipy.optimize import minimize
# import theano.tensor as tt
# import pymc3 as pm
# import theano
from pdb import set_trace

def check_posdef(S):
    """
    Check that matrix is positive definite
    Inputs
    ============
    - matrix
    """
    try:
        cholesky(S)
        return 1
    except NameError:
        print("\n Error: could not compute cholesky factor of S")
        raise
        return 0

def initialize_bundles(
    size, 
    bundle_size,
    data
    ):
    latent_bundles = dict()
    latent_bundles['z'] = np.zeros((
        size,
        bundle_size,
        data['N'],
        data['K']
        )
    )
    latent_bundles['y'] = np.zeros((
        size,
        bundle_size,
        data['N'],
        data['J']
        )
    )
    return latent_bundles

def initialize_latentvars(
    size, 
    data
    ):
    latentvars = dict()
    latentvars['z'] = np.zeros((
        size,
        data['N'],
        data['K']
        )
    )
    latentvars['y'] = np.zeros((
        size,
        data['N'],
        data['J']
        )
    )
    return latentvars


def get_weight(
    data,
    yy
    ):
    assert data['N'] == 1
    w = bernoulli.logpmf(
            data['D'],
            p=expit(yy)
            ).sum()
    return w 


def get_bundle_weights(
    bundle_size,
    data,
    y_bundle,
    ):
    bundle_w = np.empty(bundle_size)
    for l in range(bundle_size):
        bundle_w[l] = get_weight(data, y_bundle[l])
    return bundle_w


def get_weight_matrix_at_datapoint(
    size,
    bundle_size,
    data,
    yy):
    weights = np.empty((size,bundle_size))
    for m in range(size):
        weights[m] =  get_bundle_weights(
            bundle_size,
            data,
            yy[m])
    return weights


def get_weight_matrix_for_particle(
    bundle_size,
    data,
    yy):
    weights = np.empty(bundle_size)
    for l in range(bundle_size):
        weights[l] =  bernoulli.logpmf(
            data['D'],
            p=expit(yy[l])
            ).sum()
    return weights

def generate_latent_pair(
    data_J,
    data_K,
    alpha,
    beta):
    pair = dict()
    zz = norm.rvs(size = data_K)
    yy = alpha + zz @ beta.T
    pair['z'] = zz
    pair['y'] = yy
    return pair

def generate_latent_pair_laplace(
    data_y,
    alpha,
    beta):
    lapldist =  get_laplace_approx(
        data_y,
        {
            'alpha':alpha,
            'beta':beta,
        }
        )
    pair = dict()
    zz = lapldist.rvs(size = 1).reshape((1,))
    yy = alpha + zz @ beta.T
    pair['z'] = zz
    pair['y'] = yy
    return pair


def generate_latent_pair_vb(
    data_y,
    alpha,
    beta):
    vbdist =  get_vb_approx(
        data_y,
        {
            'alpha':alpha,
            'beta':beta,
        }
        )
    pair = dict()
    zz = vbdist.rvs(size = 1).reshape((1,))
    yy = alpha + zz @ beta.T
    pair['z'] = zz
    pair['y'] = yy
    return pair

## Laplace Approximation functions

def get_pi_z(z, theta):
    exp_eta = np.exp(theta['alpha'] +  z @ theta['beta'].T)
    return exp_eta/(1+exp_eta)

def get_log_likelihood(z,y,theta):
    pi_z = get_pi_z(z, theta)
    s1 = (y*np.log(pi_z))+((1.-y)*(np.log(1. - pi_z)))
    return np.sum(s1)

def get_neg_posterior(z,y,theta):
    return -1.*(get_log_likelihood(z,y,theta)+norm.logpdf(z))

def get_grad_pi_z(z, theta):
    exp_eta = np.exp(theta['alpha'] +  z @ theta['beta'].T)
    return (exp_eta *  theta['beta'].T)/(1+exp_eta)**2

def get_fisher_information(z, y, theta):
    pi_z = get_pi_z(z, theta)
    grad_pi_z = get_grad_pi_z(z, theta)
    r1 =grad_pi_z**2
    r2 =pi_z*(1.-pi_z)
    return 1. + np.sum(r1/r2)

def get_laplace_approx(y, theta):
    res = minimize(get_neg_posterior, np.array([[1]]), args=(y, theta), method='BFGS')
    cov_matrix = get_fisher_information(res.x, y, theta).reshape((1,1))
    if check_posdef(cov_matrix) == 0:
        cov_matrix = np.eye(theta['beta'].shape[1])
    return multivariate_normal(mean = res.x, cov = inv(cov_matrix))

## VB approximation functions

def get_vb_params(y, theta):
    # vb_method = 'fullrank_advi'
    vb_method = 'advi'
    with pm.Model() as model:
        z = pm.Normal('z', mu=0,  sigma=1, shape=(1,))
        p = pm.invlogit(theta['alpha'] + z @ theta['beta'].T)
        obs = pm.Bernoulli('obs', p=p, observed=y)
        advi = pm.fit(method=vb_method, n=10000, progressbar=False)
    vb_mean = advi.mean.eval()
    vb_cov = advi.cov.eval()
    return vb_mean, vb_cov

def get_vb_approx(y, theta):
    vb_mean, vb_cov = get_vb_params(y, theta)
    if check_posdef(vb_cov) == 0:
        vb_cov = np.eye(theta['beta'].shape[1])
    return multivariate_normal(mean=vb_mean, cov=vb_cov)