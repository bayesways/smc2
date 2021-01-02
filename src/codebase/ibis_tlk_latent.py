import numpy as np
from scipy.stats import bernoulli, multivariate_normal, logistic, norm
from scipy.special import expit, logsumexp
from pdb import set_trace


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
