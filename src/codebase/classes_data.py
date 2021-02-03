from codebase.data import gen_data_master

class Data:
    def __init__(self, name, model_num, size, random_seed=None):
        self.name = name
        self.model_num = model_num
        self.size = size 
        self.random_seed = random_seed

    def generate(self):
        self.raw_data = gen_data_master(
            model_num = self.model_num,
            nsim_data = self.size,
            random_seed = self.random_seed
            )

    def get_stan_data(self):
        stan_data = dict()
        for name in self.raw_data['stan_constants']:
            stan_data[name] = self.raw_data[name]
        for name in self.raw_data['stan_data']:
            stan_data[name] = self.raw_data[name]
        return stan_data

    def get_stan_data_at_t(self, t):
        stan_data = dict()
        for name in self.raw_data['stan_constants']:
            stan_data[name] = self.raw_data[name]
        stan_data['N'] = 1
        for name in self.raw_data['stan_data']:
            stan_data[name] = self.raw_data[name][t]
        return stan_data

    def get_stan_data_at_t2(self, t):
        stan_data = dict()
        for name in self.raw_data['stan_constants']:
            stan_data[name] = self.raw_data[name]
        stan_data['N'] = 1
        for name in self.raw_data['stan_data']:
            stan_data[name] = self.raw_data[name][t].reshape(
                (1,self.raw_data['J'])
                )
        return stan_data

    def get_stan_data_upto_t(self, t):
        stan_data = dict()
        for name in self.raw_data['stan_constants']:
            stan_data[name] = self.raw_data[name]
        stan_data['N'] = t
        for name in self.raw_data['stan_data']:
            stan_data[name] = self.raw_data[name][:t]
        return stan_data