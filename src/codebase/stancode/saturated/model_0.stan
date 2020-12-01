data{
  int<lower=0> N;
  int<lower=0> J;
  vector[J] y[N];

}

parameters {
  vector[J] alpha;
  cholesky_factor_corr[J] L_R; 
  vector<lower=0>[J] sigma;
}

model{
  alpha ~ normal(0,5);
  L_R ~ lkj_corr_cholesky(2);
  sigma ~ cauchy(0,2.);
  for (n in 1:N) y[n,] ~ multi_normal_cholesky(alpha, diag_pre_multiply(sigma, L_R));
}

generated quantities{
  cov_matrix[J] Marg_cov = multiply_lower_tri_self_transpose(diag_pre_multiply(sigma, L_R));  
}
