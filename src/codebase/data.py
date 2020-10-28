import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm, bernoulli
from numpy.linalg import inv,cholesky
from scipy.special import expit, logit


def gen_cov_matrix(dim, scale = 1., random_seed = None):
    """
    Return covariance matrix with values scaled according
    to the input scale.
    Inputs
    ============
    - dim
    - scale

    Output
    ============
    - np. array of shape (dim, dim)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    A = np.tril(uniform(-scale,scale,size = (dim,dim)))
    C = A @ A.T
    return C


def flatten_corr_matrix_samples(Rs, offset = 0, colnames=None):
    """
    Flatten a [N, K, K ] array of correlation
    matrix samples to a [N,M] array where
    M is the number of of elements below the
    diagonal for a K by K matrix.

    For each sample correlation matrix we care only
    for these M parameters
    Inputs
    ============
    - Rs : samples to flattent out, should be
        of dimension [N,K,K]
    - offset : set 1 to exclude diagonal
    - colnames
    Output
    ============
    -  a dataframe of size [N,M]
    """
    N,K = Rs.shape[0], Rs.shape[1]
    if colnames is None:
        colnames = [str(x) for x in range(K)]

    assert len(colnames) == K, 'colnames should as long as the columns of R'
    cnames = corr_headers(colnames, offset = offset)

    M = len(cnames)
    fRs = np.empty((N,M))
    for i in range (N):
        fRs[i,:] = flatten_corr(Rs[i,:,:], offset = offset)

    fRs = pd.DataFrame(fRs)
    fRs.columns=cnames


    return fRs



def C_to_R(M):
    """
    Send a covariance matrix M to the corresponding
    correlation matrix R
    Inputs
    ============
    - M : covariance matrix
    Output
    ============
    - correlation matrix
    """
    d = np.asarray(M.diagonal())
    d2 = np.diag(d**(-.5))
    R = d2 @ M @ d2
    return R


def thin(x, rate = 10):
    """
    Thin an array of numbers by choosing every
    nth sample
    """
    return x[::rate]


def check_posdef(R):
    """
    Check that matrix is positive definite
    Inputs
    ============
    - matrix
    """
    try:
        cholesky(R)
    except NameError:
        print("\n Error: could not compute cholesky factor of R")
        raise
    return 0    


def get_data(
    nsim_data,
    J=6,
    random_seed=None,
    ):
    if random_seed is not None:
        np.random.seed(random_seed)

    alpha = np.zeros(J)

    Phi_cov = np.eye(J)
    zz = multivariate_normal.rvs(
        mean=alpha,
        cov=Phi_cov,
        size=nsim_data
    )
    
    pp = expit(zz)
    yy = bernoulli.rvs(p=pp)  
    
    data = dict()
    data['random_seed'] = random_seed
    data['N'] = nsim_data
    data['J'] = J
    data['alpha'] = alpha
    data['Phi_cov'] = Phi_cov
    data['z'] = zz
    data['y'] = yy
    data['p'] = pp
    return data