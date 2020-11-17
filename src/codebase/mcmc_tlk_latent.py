import numpy as np
from scipy.stats import bernoulli, multivariate_normal, logistic, norm
from scipy.special import expit, logsumexp


def gen_latent_weights_master(
    latent_model_num,
    data,
    particles,
    cloud_size
    ):
    if latent_model_num == 1:
        return get_latent_weights_1(
            data,
            particles,
            cloud_size
            )


def get_latent_weights_1(
    data,
    latent_particles,
    cloud_size
    ):
    """
    data in NxJ
    weights are of size NxK 
    where K is the size of each cloud. 

    For each data point D_t we compute
    weights p(D_t | Z_t^k) for k:1...K

    And return the weight of each cloud
    as the average across k 
    """
    weights = np.empty((cloud_size, data['N']))
    for m in range(cloud_size):
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
#     cloud_size,
#     data
#     ):
#     if latent_model_num == 1:
#         return sample_latent_1(
#             particles,
#             particle_size,
#             cloud_size,
#             data
#             )



# def sample_latent_1(particles, particles_size, cloud_size, data, c = 1):
#     y_latent = np.empty((
#         particles_size,
#         cloud_size,
#         data['J']))
#     zz = np.empty((
#         particles_size,
#         cloud_size))
#     for m in range(particles_size):
#         a = particles['alpha'][m]
#         b = particles['beta'][m]
#         logistic_dstn = logistic(scale = c)        
#         zz[m] = norm.rvs(size=cloud_size)
#         y_latent[m] = a + np.outer(zz[m], b) 
#         for j in range (data['J']):
#             y_latent[m, :, j] = y_latent[m, :, j] +\
#                 logistic_dstn.rvs(size=cloud_size) 
    
#     samples = dict()
#     samples['z'] = zz
#     samples['y_latent'] = y_latent
#     return samples


def sample_zcloud(d1, d2):
        return norm.rvs(
            size = d1*d2).reshape((
                d1,
                d2,
                )
            )


def generate_latent_variables(
    cloud_size,
    data_N,
    data_J,
    alpha,
    beta):
    latent_vars = dict()
    zz = sample_zcloud(
        cloud_size,
        data_N)
    y_latent = np.empty((
        cloud_size,
        data_N,
        data_J))
    for m in range(cloud_size):
        y_latent[m] = np.squeeze(alpha) +\
            np.outer(zz[m], np.squeeze(beta))
    
    latent_vars['z'] = zz
    latent_vars['y_latent'] = y_latent
    return latent_vars
