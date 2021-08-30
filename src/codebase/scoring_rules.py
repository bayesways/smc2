import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, bernoulli
from scipy.special import expit
from pdb import set_trace

def variogram_score(y_pred, y_obs, p = 0.5):
    mcmc_lenght = y_pred.shape[0]
    y_obs_length = y_obs.shape[0]
    p = 0.5

    s = 0.
    for i in range(y_obs_length):
        for j in range(y_obs_length):
            s1 = np.abs(y_obs[i] - y_obs[j])**p
            s2 = (-1./mcmc_lenght) * np.array(
                [
                    np.abs(y_pred[k,i] - y_pred[k,j])**p
                    for k in range(mcmc_lenght)
                ]).sum()
            s_ij = (s1 + s2)**2
            s = s + s_ij
    return s


def get_variogram_score(ps, data):
    mcmc_length = ps["alpha"].shape[0]
    dim_J = ps['alpha'].shape[1]
    post_y = np.empty((mcmc_length, dim_J), dtype = float)
    
    for m in range(mcmc_length):
        mean =  ps['alpha'][m]
        Cov = ps['Marg_cov'][m]
        post_y[m] = multivariate_normal.rvs(mean=mean, cov = Cov)

    scores_variogram = variogram_score(
        y_pred=post_y,
        y_obs=data)
    return scores_variogram


def to_str_pattern(y0):
    if np.ndim(y0) == 1:
        return "".join(y0.astype(str))
    if np.ndim(y0) == 2:
        y = pd.DataFrame(y0)
        yresp = y.apply(lambda x: "".join(x.astype(str)), axis=1)
        return yresp


def to_nparray_data(yresp):
    if type(yresp) == str:
        return np.array(list(yresp)).astype(int)
    else:
        J = len(yresp[0])
        N = yresp.shape[0]
        res = np.empty((N, J))
        for i in range(N):
            res[i] = np.array(list(yresp[i])).astype(int)
        return res


def get_response_probs(data_ptrn, prob):
    distinct_patterns = np.unique(data_ptrn)
    ## compute E_y(theta) for a specific pattern y
    Ey = dict()
    for ptrn in distinct_patterns:
        prob_matrix = bernoulli.logpmf(k=to_nparray_data(ptrn), p=prob)
        Ey[ptrn] = np.mean(np.exp(np.sum(prob_matrix, 1)), 0)
    return Ey

def compute_log_score(probs_dict, data):
    n = data.shape[0]
    score = 0.0
    for i in range(n):
        score_individual = -np.log(probs_dict[data[i]])
        score = score + score_individual
    return score


def get_logscore1(data_ptrn, post_y):
    E_prob = get_response_probs(data_ptrn, expit(post_y))
    return compute_log_score(E_prob, data_ptrn)


def get_method2(ps, dim_J, dim_K, nsim):
    post_y = np.empty((nsim, dim_J))
    for m in range(nsim):
        m_alpha = ps["alpha"][m]
        if "Marg_cov" in ps.keys():
            m_Marg_cov = ps["Marg_cov"][m]
            post_y_sample = multivariate_normal.rvs(
                mean=m_alpha, cov=m_Marg_cov, size=1
            )
        else:
            m_beta = ps["beta"][m]
            if "Phi_cov" in ps.keys():
                m_Phi_cov = ps["Phi_cov"][m]
            else:
                m_Phi_cov = np.eye(dim_K)
            zz_from_prior = multivariate_normal.rvs(
                mean=np.zeros(dim_K), cov=m_Phi_cov, size=1
            ).reshape(dim_K,)
            post_y_sample = m_alpha + zz_from_prior @ m_beta.T
        post_y[m] = post_y_sample
    return post_y


def get_logscore(ps, data):
    mcmc_length = ps["alpha"].shape[0]
    dim_J = ps['alpha'].shape[1]
    dim_K = ps["beta"].shape[-1]
    data_ptrn = to_str_pattern(data)

    # fix use whole distribution
    post_y = get_method2(ps, dim_J, dim_K, mcmc_length)
    lgscr1 = get_logscore1(data_ptrn, post_y)
    
    return lgscr1
