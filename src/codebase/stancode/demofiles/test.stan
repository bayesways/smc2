data {
  int<lower=0> N;
  vector[N] x;
}

transformed data{
  vector[3] mu = rep_vector(1,3);
  matrix[3, 3] I = diag_matrix(rep_vector(1, 3));

}

generated quantities {
  vector[3] alpha;
  cholesky_factor_corr[3] beta; 
  alpha = multi_normal_rng(mu, I);
  beta = lkj_corr_cholesky_rng(3, 2);
}
