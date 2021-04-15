import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm, bernoulli, uniform
from numpy.linalg import inv,cholesky
from scipy.special import expit, logit


def flatten_matrix(a, include_diag = True):
    """
    Flatten a [K, K ] correlation
    matrix to [M,] array where
    M is the number of of elements above the
    diagonal for a K by K matrix.
    Inputs
    """
    if include_diag:
        offset = 0
    else:
        offset = 1
    return a[np.triu_indices(a.shape[0], k=offset)]


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


def gen_data_master(
    model_num,
    nsim_data,
    J=6,
    K=1,
    c=1,
    random_seed=None
    ):
    if random_seed is None: 
        random_seed = 0
    if model_num == 1: #1 factor model for binary data
        return gen_data_1(
            nsim_data,
            J=6,
            random_seed=random_seed
            )
    elif model_num == 2:
        return gen_data_2(
            nsim_data,
            J=6,
            random_seed=random_seed
            )
    elif model_num == 4:
        return gen_data_4(
            nsim_data,
            J=6,
            K=2,
            random_seed=random_seed
            )
    elif model_num == 6: #2 factor model for binary data
        return gen_data_6(
            nsim_data,
            random_seed=random_seed
            )
    elif model_num == 'big5':
        return get_big5()
        
    

def get_big5():
    standardize = 1
    gender = 'women'
    
    print("\n\nReading data for %s"%gender)
    df = pd.read_csv("../dat/muthen_"+gender+".csv")
    df = df.replace(-9, np.nan).astype(float)
    df.dropna(inplace=True)
    df = df.astype(int)
    data = dict()
    data['N'] = df.shape[0]
    data['K'] = 5
    data['J'] = df.shape[1]
    if standardize:
        from sklearn import preprocessing
        data['y'] = preprocessing.scale(df.values)
    else:
        data['y'] = df.values
    print("\n\nN = %d, J= %d, K =%d"%(data['N'],data['J'], data['K'] ))
    data['sigma_prior'] = np.diag(np.linalg.inv(np.cov(data['y'], rowvar=False)))
    data['stan_constants'] = ['N','J', 'K', 'sigma_prior']
    data['stan_data'] = ['y']
    return data



def gen_data_1(
    nsim_data,
    J=6,
    random_seed=None
    ):
    """
    1 factor model for binary data
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    alpha = np.array([-0.53,  0.35, -1.4 , -1.4 , -0.96, -2.33])
    # beta = np.array([1, 0.7, .8, .5, .9, .6])
    beta = np.ones(6)

    zz = norm.rvs(size=nsim_data)
    yy = alpha + np.outer(zz, beta)
    
    DD = bernoulli.rvs(p=expit(yy))

    
    data = dict()
    data['random_seed'] = random_seed
    data['N'] = nsim_data
    data['J'] = J
    data['K'] = 1
    data['alpha'] = alpha
    data['beta'] = beta
    data['z'] = zz
    data['y'] = yy
    data['D'] = DD
    data['stan_constants'] = ['N', 'J', 'K']
    data['stan_data'] = ['D']
    return data


def gen_data_2(
    nsim_data,
    J=6,
    random_seed=None,
    ):
    if random_seed is not None:
        np.random.seed(random_seed)

    # alpha = np.zeros(J)
    alpha = np.array([1,2,-1,-2,0,4]).astype(float)
    sigma = np.array([1,2,3,1,2,3]).astype(float)
    Marg_cov = np.diag(sigma) @ np.eye(J) @ np.diag(sigma)
    yy = multivariate_normal.rvs(
        mean=alpha,
        cov=Marg_cov,
        size=nsim_data
    )
    
    data = dict()
    data['random_seed'] = random_seed
    data['N'] = nsim_data
    data['J'] = J
    data['alpha'] = alpha
    data['sigma'] = sigma
    data['Marg_cov'] = Marg_cov
    data['y'] = yy
    data['stan_constants'] = ['N','J']
    data['stan_data'] = ['y']
    return data


def gen_data_3(
    nsim_data,
    J=6,
    random_seed=None,
    ):
    if random_seed is not None:
        np.random.seed(random_seed)

    # alpha = np.zeros(J)
    alpha = [1,2,-1,-2,0,0.6]

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


def gen_data_4(
    nsim_data,
    J=6,
    K=2,
    rho=0.2,
    c=0.65,
    b=0.8,
    off_diag_residual = False,
    off_diag_corr = 0.6,
    random_seed=None,
    ):
    if random_seed is not None:
        np.random.seed(random_seed)

    alpha = np.zeros(J)
    beta = np.array([[1, 0],
                        [b, 0],
                        [b, 0],
                        [0, 1],
                        [0, b],
                        [0, b]], dtype=float)

    sigma_z = np.repeat(np.sqrt(c), K)
    Phi_corr = np.eye(K)
    Phi_corr[0, 1] = rho
    Phi_corr[1, 0] = rho
    Phi_cov = np.diag(sigma_z) @ Phi_corr @  np.diag(sigma_z)

    sigma_sq = 1 - np.diag(beta @ Phi_cov @ beta.T)
    sigma = np.sqrt(sigma_sq)

    if off_diag_residual:
        Theta_corr = np.eye(J)
#         Theta = np.diag(sigma_sq)
        for i in [1, 2, 5]:
            for j in [3, 4]:
                #                 Theta[i,j] = off_diag_corr*sigma[i]*sigma[j]
                #                 Theta[j,i] = off_diag_corr*sigma[i]*sigma[j]
                Theta_corr[i, j] = off_diag_corr
                Theta_corr[j, i] = off_diag_corr
        Theta = np.diag(np.sqrt(sigma_sq)) @ Theta_corr @  np.diag(
            np.sqrt(sigma_sq))
    else:
        Theta = np.diag(sigma_sq)

    Marg_cov = beta @ Phi_cov @ beta.T + Theta
    yy = multivariate_normal.rvs(mean=alpha, cov=Marg_cov, size=nsim_data)

    sigma_prior = np.diag(np.linalg.inv(np.cov(yy, rowvar=False)))

    data = dict()
    data['random_seed'] = random_seed
    data['N'] = nsim_data
    data['K'] = K
    data['J'] = J
    data['alpha'] = alpha
    data['beta'] = beta
    data['sigma_z'] = sigma_z
    data['Phi_corr'] = Phi_corr
    data['Phi_cov'] = Phi_cov
    data['Marg_cov'] = Marg_cov
    data['Theta'] = Theta
    data['sigma'] = sigma
    data['y'] = yy
    data['off_diag_residual'] = off_diag_residual
    data['sigma_prior'] = sigma_prior
    data['stan_constants'] = ['N','J', 'K', 'sigma_prior']
    data['stan_data'] = ['y']
    return(data)



def gen_data_6(
    nsim_data,
    J=6,
    K=2,
    rho=0.2,
    b=0.8,
    rho2=0.1,
    c=1,
    random_seed=None
    ):
    if random_seed is not None:
        np.random.seed(random_seed)
    beta = np.array(
        [[1, 0],
        [b, 0],
        [b, 0],
        [0, 1],
        [0, b],
        [0, b]], dtype=float
        )
    alpha = np.zeros(J)
    Phi_corr = np.eye(K)
    Phi_corr[0, 1] = rho
    Phi_corr[1, 0] = rho
    Phi_cov = Phi_corr
    assert check_posdef(Phi_cov) == 0
    zz = multivariate_normal.rvs(
        mean=np.zeros(K),
        cov=Phi_cov,
        size=nsim_data
        )

    yy = alpha + zz @ beta.T
    DD = bernoulli.rvs(p=expit(yy))

    data = dict()
    data['random_seed'] = random_seed
    data['N'] = nsim_data
    data['K'] = K
    data['J'] = J
    data['alpha'] = alpha
    data['beta'] = beta
    data['Phi_cov'] = Phi_cov
    data['z'] = zz
    data['y'] = yy
    data['D'] = DD
    data['stan_constants'] = ['N','J','K']
    data['stan_data'] = ['D']
    
    return(data)