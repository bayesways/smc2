import numpy as np
from scipy.stats import bernoulli, multivariate_normal, logistic, norm
from scipy.special import expit, logsumexp
from pdb import set_trace


def gen_latent_weights_master(
    latent_model_num,
    data,
    particles,
    bundle_size
    ):
    if latent_model_num == 1:
        return get_latent_weights_1(
            data,
            particles,
            bundle_size
            )


def get_latent_weights_1(
    data,
    latent_particles,
    bundle_size
    ):
    """
    data in NxJ
    weights are of size NxK 
    where K is the size of each bundle. 

    For each data point D_t we compute
    weights p(D_t | Z_t^k) for k:1...K

    And return the weight of each bundle
    as the average across k 
    """
    weights = np.empty((bundle_size, data['N']))
    for m in range(bundle_size):
        for t in range(data['N']):
            weights[m,t] = bernoulli.logpmf(
                data['D'][t],
                p=expit(latent_particles['y_latent'][m])
            ).sum()
    # return np.mean(weights,axis=1)
    return weights


# def sample_latent_master(
#     latent_model_num,
#     particles,
#     particle_size,
#     bundle_size,
#     data
#     ):
#     if latent_model_num == 1:
#         return sample_latent_1(
#             particles,
#             particle_size,
#             bundle_size,
#             data
#             )



# def sample_latent_1(particles, particles_size, bundle_size, data, c = 1):
#     y_latent = np.empty((
#         particles_size,
#         bundle_size,
#         data['J']))
#     zz = np.empty((
#         particles_size,
#         bundle_size))
#     for m in range(particles_size):
#         a = particles['alpha'][m]
#         b = particles['beta'][m]
#         logistic_dstn = logistic(scale = c)        
#         zz[m] = norm.rvs(size=bundle_size)
#         y_latent[m] = a + np.outer(zz[m], b) 
#         for j in range (data['J']):
#             y_latent[m, :, j] = y_latent[m, :, j] +\
#                 logistic_dstn.rvs(size=bundle_size) 
    
#     samples = dict()
#     samples['z'] = zz
#     samples['y_latent'] = y_latent
#     return samples


def generate_latent_variables(
    data_N,
    data_J,
    data_K,
    alpha,
    beta):
    latent_vars = dict()
    zz = norm.rvs(
        size = data_N*data_K
        ).reshape((data_N,data_K))
    y_latent = np.empty((
        data_N,
        data_J))
    y_latent = np.squeeze(alpha) +\
        zz @ np.squeeze(beta).reshape(data_J, data_K).T
    latent_vars['z'] = zz
    latent_vars['y_latent'] = y_latent
    return latent_vars



def generate_latent_variables_bundle(
    bundle_size,
    data_N,
    data_J,
    data_K,
    alpha,
    beta):
    latent_vars = dict()
    zz = np.empty((
        bundle_size,
        data_N,
        data_K))
    y_latent = np.empty((
        bundle_size,
        data_N,
        data_J))
    for m in range(bundle_size):
        bundle_vars = generate_latent_variables(        
            data_N,
            data_J,
            data_K,
            alpha,
            beta)
        zz[m] = bundle_vars['z']
        y_latent[m] = bundle_vars['y_latent']
    latent_vars['z'] = zz
    latent_vars['y_latent'] = y_latent
    return latent_vars
