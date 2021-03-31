import sys, os

def model_phonebook_path(model_num, prior, data_type='cont'):
    path_to_stan = './codebase/stancode/disc/'

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
    # elif model_num == 3:
    #     if prior:
    #         path = 'CFA/AZ/model2_big5_prior.stan'
    #     else:
    #         path = 'CFA/AZ/model2_big5.stan' 
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
            'beta',
            'zz',
            'yy'
            ]
        names['latent_names'] = ['y', 'z']
    elif model_num == 2:
        names['param_names'] = ['beta', 'alpha']
        names['latent_names'] = ['z', 'y']
    elif model_num == 3:
        names['param_names'] = ['beta', 'alpha']
        names['latent_names'] = ['z', 'y']
    elif model_num == 4:
        names['param_names'] = ['beta', 'alpha']
        names['latent_names'] = ['z', 'y']
    # elif model_num == 3 :
    #     names['param_names'] = [
    #         'sigma_square',
    #         'alpha',
    #         'beta_free',
    #         'beta_zeros',
    #         'Phi_cov',
    #         'Marg_cov',
    #         'beta',
    #         'Omega'
    #         ]
    #     names['latent_names'] = []
    else:
        print("model number not found")
        sys.exit()

    return names