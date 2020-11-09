data{
  int<lower=0> N;
  int<lower=0> J;
  vector[J] y[N];

}

transformed data {
  matrix[J, J] I = diag_matrix(rep_vector(1, J));
}
parameters {
  vector[J] alpha;
  cholesky_factor_corr[J] L_R; 
}

model{
  alpha ~ normal(0,5);
  L_R ~ lkj_corr_cholesky(2);
  for (n in 1:N) y[n,] ~ multi_normal_cholesky(alpha, L_R);
}

generated quantities{
  cov_matrix[J] Marg_cov = multiply_lower_tri_self_transpose(L_R);  
}
