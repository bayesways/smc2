import numpy as np
from scipy.stats import bernoulli, multivariate_normal, logistic, norm
from scipy.special import expit, logsumexp


def gen_latent_weights_master(
    latent_model_num,
    data,
    particles,
    particle_size,
    cloud_size
    ):
    if latent_model_num == 1:
        return get_latent_weights_1(
            data,
            particles,
            particle_size,
            cloud_size
            )


def get_latent_weights_1(
    data,
    latent_particles,
    particle_size,
    cloud_size
    ):
    weights = np.empty((particle_size, cloud_size))
    for m in range(particle_size):
        w = bernoulli.logpmf(
            data['D'],
            p=expit(latent_particles['y_latent'][m])
        )
        weights[m] = np.sum(w, axis=1)
    return weights



def sample_latent_master(
    latent_model_num,
    particles,
    particle_size,
    cloud_size,
    data
    ):
    if latent_model_num == 1:
        return sample_latent_1(
            particles,
            particle_size,
            cloud_size,
            data
            )



def sample_latent_1(particles, particles_size, cloud_size, data, c = 1):
    y_latent = np.empty((
        particles_size,
        cloud_size,
        data['J']))
    zz = np.empty((
        particles_size,
        cloud_size))
    for m in range(particles_size):
        a = particles['alpha'][m]
        b = particles['beta'][m]
        logistic_dstn = logistic(scale = c)        
        zz[m] = norm.rvs(size=cloud_size)
        y_latent[m] = a + np.outer(zz[m], b) 
        for j in range (data['J']):
            y_latent[m, :, j] = y_latent[m, :, j] +\
                logistic_dstn.rvs(size=cloud_size) 
    
    samples = dict()
    samples['z'] = zz
    samples['y_latent'] = y_latent
    return samples
