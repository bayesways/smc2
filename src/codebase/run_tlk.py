import sys, os

def model_phonebook_path(model_num, prior, data_type='cont'):
    if model_num in range(0,6):
        path_to_stan = './codebase/stancode/cont/'
    elif model_num in [7]:
        path_to_stan = './codebase/stancode/models/'
    elif model_num in [8]:
        path_to_stan = './codebase/stancode/disc/'
    else:
        pass

    if model_num == 0:
        if prior:
            path = 'saturated/model_0_prior.stan'
        else:
            path = 'saturated/model_0.stan'
    elif model_num == 1:
        if prior:
            path = 'CFA/EZ/model_1_prior.stan'
        else:
            path = 'CFA/EZ/model_1.stan'
    elif model_num == 2:
        if prior:
            path = 'CFA/AZ/model2_big5_prior.stan'
        else:
            path = 'CFA/AZ/model2_big5.stan'   
    elif model_num == 3:
        if prior:
            path = 'CFA/AZ/model_2_prior.stan'
        else:
            path = 'CFA/AZ/model_2.stan'
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
    elif model_num == 7:
        if prior:
            path = 'ibis/model_7_prior.stan'
        else:
            path = 'ibis/model_7.stan'
    elif model_num == 8:
        if prior:
            path = 'CFA/EZ/model_1_prior.stan'
        else:
            path = 'CFA/EZ/model_1.stan'
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
    elif model_num == 7:
        names['param_names'] = ['beta', 'alpha']
        names['latent_names'] = ['z', 'y_latent']
    elif model_num == 8:
        names['param_names'] = ['beta', 'alpha']
        names['latent_names'] = ['z', 'y_latent']
    else:
        print("model number not found")
        sys.exit()

    return names