import sys, os

def model_phonebook_path(model_num, prior, data_type='cont'):
    if model_num in range(0,4):
        path_to_stan = './codebase/stancode/disc/'
    elif model_num in range(5,15):
        path_to_stan = './codebase/stancode/cont/'
    else:
        pass
    if model_num == 1:
        if prior:
            path = 'CFA/EZ/model_1_prior.stan'
        else:
            path = 'CFA/EZ/model_1.stan'
    elif model_num == 2:
        if prior:
            path = 'smc2/model_1_1factor_prior.stan'
        else:
            path = 'smc2/model_1_1factor.stan'
    elif model_num == 3:
        if prior:
            path = 'smc2/model_1_1factor_prior2.stan'
        else:
            path = 'smc2/model_1_1factor2.stan'
    elif model_num == 4:
        if prior:
            path = 'smc2/model_1_1factor_prior3.stan'
        else:
            path = 'smc2/model_1_1factor3.stan'
    elif model_num == 5:
        if prior:
            path = 'CFA/AZ/model_2_prior.stan'
        else:
            path = 'CFA/AZ/model_2.stan' 
    elif model_num == 6:
        if prior:
            path = 'CFA/AZ/model2_big5_prior.stan'
        else:
            path = 'CFA/AZ/model2_big5.stan' 
    elif model_num == 7:
        if prior:
            path = 'CFA/EZ/model_1_prior.stan'
        else:
            path = 'CFA/EZ/model_1.stan' 
    elif model_num == 8:
        if prior:
            path = 'CFA/EZ/model_1B_prior.stan'
        else:
            path = 'CFA/EZ/model_1B.stan' 
    elif model_num == 9:
        if prior:
            path = 'EFA/model_1_prior.stan'
        else:
            path = 'EFA/model_1.stan' 
    elif model_num == 10:
        if prior:
            path = 'EFA/model_2_prior.stan'
        else:
            path = 'EFA/model_2.stan' 
    elif model_num == 11:
        if prior:
            path = 'CFA/EZ/model1_big5_prior.stan'
        else:
            path = 'CFA/EZ/model1_big5.stan'
    elif model_num == 12:
        if prior:
            path = 'saturated/model_0_prior.stan'
        else:
            path = 'saturated/model_0.stan'
    elif model_num == 13:
        if prior:
            path = 'CFA/AZ/model_2C_prior.stan'
        else:
            path = 'CFA/AZ/model_2C.stan' 
    elif model_num == 14:
        if prior:
            path = 'CFA/EZ/model_1_prior.stan'
        else:
            path = 'CFA/EZ/model_1.stan' 
    else:
        print("model number not found")
        sys.exit()
    
    models_dir = './log/compiled_models/model%s'%model_num
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    return path_to_stan+path


def model_phonebook(model_num):
    names = dict()
    if model_num == 1:
        names['param_names'] = [
            'alpha',
            'beta'
            ]
        names['stan_names'] = [
            'alpha',
            'beta',
            'z',
            'y'
            ]
        names['latent_names'] = ['yy', 'zz']
    elif model_num == 2:
        names['param_names'] = ['beta', 'alpha']
        names['latent_names'] = ['z', 'y']
    elif model_num == 3:
        names['param_names'] = ['beta', 'alpha']
        names['stan_names'] = [
            'alpha',
            'beta',
            'zz'
            ]
        names['latent_names'] = ['y', 'z']
    elif model_num == 4:
        names['param_names'] = ['beta', 'alpha']
        names['latent_names'] = ['z', 'y']
    elif model_num in [5,6,13] :
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
        names['stan_names'] = names['param_names']
        names['latent_names'] = []
    elif model_num in [7,8,11,14] :
        names['param_names'] = [
            'sigma_square',
            'alpha',
            'beta_free',
            'Phi_cov',
            'Marg_cov',
            'beta',
            ]
        names['stan_names'] = names['param_names']
        names['latent_names'] = []
    elif model_num in [9] :
        names['param_names'] = [
            'sigma_square',
            'alpha',
            'Marg_cov',
            'beta'
            ]
        names['stan_names'] = names['param_names']
        names['latent_names'] = []
    elif model_num in [10] :
        names['param_names'] = [
            'sigma_square',
            'alpha',
            'beta',
            'Marg_cov',
            'Omega'
            ]
        names['stan_names'] = names['param_names']
        names['latent_names'] = []
    elif model_num in [12] :
        names['param_names'] = [
            'sigma_square',
            'alpha',
            'Marg_cov'
            ]
        names['stan_names'] = names['param_names']
        names['latent_names'] = []
    else:
        print("model number not found")
        sys.exit()

    return names