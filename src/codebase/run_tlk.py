import sys, os

def model_phonebook_path(model_num, prior, data_type='cont'):
    if data_type == 'cont':
        path_to_stan = './codebase/stancode/cont/'
    else:
        pass

    if model_num == 0:
        if prior:
            path = 'saturated/model_0_prior.stan'
        else:
            path = 'saturated/model_0.stan'

    elif model_num == 1:
        if prior:
            path = 'CFA/model_1_prior.stan'
        else:
            path = 'CFA/model_1.stan'
    elif model_num == 2:
        if prior:
            path = 'CFA/model2_big5_prior.stan'
        else:
            path = 'CFA/model2_big5.stan'   
    elif model_num == 3:
        if prior:
            path = 'CFA/model_2_prior.stan'
        else:
            path = 'CFA/model_2.stan'
    elif model_num == 4:
        if prior:
            path = 'EFA/model_1_prior.stan'
        else:
            path = 'EFA/model_1.stan'
    elif model_num == 5:
        if prior:
            path = 'EFA/model_2_prior.stan'
        else:
            path = 'EFA/model_2.stan'
    else:
        print("model number not found")
        sys.exit()

    return path_to_stan+path


def model_phonebook(model_num):
    names = dict()

    if model_num == 0:
        names['param_names'] = ['Marg_cov', 'L_R', 'alpha', 'sigma']
        names['latent_names'] = []
    elif model_num == 1:
        names['param_names'] = [
            'sigma_square',
            'alpha',
            'beta_free',
            'Phi_cov',
            'Marg_cov',
            'beta',
            ]
        names['latent_names'] = []
    elif model_num == 2:
        names['param_names'] = [
            'sigma_square',
            'alpha',
            'beta_free',
            'beta_zeros',
            'Phi_cov',
            'Marg_cov',
            'beta',
            'Omega'
            ]
        names['latent_names'] = []
    elif model_num == 3:
        names['param_names'] = [
            'sigma_square',
            'alpha',
            'beta_free',
            'beta_zeros',
            'Phi_cov',
            'Marg_cov',
            'beta',
            'Omega'
            ]
        names['latent_names'] = []
    elif model_num == 4:
        names['param_names'] = [
            'sigma_square',
            'alpha',
            'Marg_cov',
            'beta'
            ]
        names['latent_names'] = []
    elif model_num == 5:
        names['param_names'] = [
            'sigma_square',
            'alpha',
            'Marg_cov',
            'beta',
            'Omega'
            ]
        names['latent_names'] = []
    else:
        print("model number not found")
        sys.exit()

    return names