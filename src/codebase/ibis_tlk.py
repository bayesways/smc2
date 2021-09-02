import numpy as np
from scipy.stats import bernoulli, multivariate_normal
from scipy.special import expit, logsumexp


def gen_weights_master(
    model_num,
    data,
    particles,
    M
    ):
    if model_num in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
        return get_weights_0(
            data,
            particles,
            M
            )
    else:
        print('No weights method found for this model number')
        pass
    

def get_weights_0(data, particles, M):
    """
    dim(y) = k 

    For a single data point y, compute the
    likelihood of each particle as the 
    average likelihood across a sample of latent
    variables z.
    """

    weights = np.empty(M)
    for m in range(M):
        weights[m] = multivariate_normal.logpdf(
            x = data['y'],
            mean = particles['alpha'][m].reshape(
                data['y'].shape
                ),
            cov = particles['Marg_cov'][m].reshape(
                data['y'].shape[0],
                data['y'].shape[0]
                ),
            allow_singular=True
        )
    return weights
