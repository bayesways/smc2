import numpy as np
from scipy.stats import bernoulli, multivariate_normal, logistic, norm
from scipy.special import expit, logsumexp
from pdb import set_trace
from scipy.optimize import brentq, minimize


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
        })
    pair = dict()
    zz = lapldist.rvs(size = 1).reshape((1,))
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
    s1 = np.sum((y*np.log(pi_z))+((1.-y)*(1.-np.log(pi_z))))
    s2 = -.5 * np.sum(z**2)
    return s1+s2

def get_neg_log_likelihood(z,y,theta):
    return -get_log_likelihood(z,y,theta)

def get_neg_posterior(z,y,theta):
    return - get_log_likelihood(z,y,theta) - norm.logpdf(z)

def get_grad_pi_z(z, theta):
    exp_eta = np.exp(theta['alpha'] +  z @ theta['beta'].T)
    return (exp_eta *  theta['beta'].T)/(1+exp_eta)**2

def get_hessian(z, y, theta):
    pi_z = get_pi_z(z, theta)
    grad_pi_z = get_grad_pi_z(z, theta)
    r1 =grad_pi_z**2
    r2 =pi_z*(1.-pi_z)
    return 1. + np.sum(r1/r2)

def get_laplace_approx(y, theta):
    res = minimize(get_neg_posterior, np.array([[0]]), args=(y, theta), method='BFGS')
    fisher_info_matrix = get_hessian(res.x, y, theta)
    return multivariate_normal(mean = res.x, cov = fisher_info_matrix**(-1))
