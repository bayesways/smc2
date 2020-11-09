data{
  int<lower=0> N;
  int<lower=0> J;
  int<lower=0, upper=1> y[N,J];
}

transformed data {
  matrix[J, J] I = diag_matrix(rep_vector(1, J));
}
parameters {
  vector[J] z[N];
  vector[J] alpha;
  cholesky_factor_corr[J] L_R; 
}

model{
  alpha ~ normal(0,10);
  for (n in 1:N) z[n,] ~ normal(0,1);
  L_R ~ lkj_corr_cholesky(2);
  for (n in 1:N) z[n,] ~ multi_normal_cholesky(alpha, L_R);
  for (j in 1:J) y[,j] ~ bernoulli_logit(z[,j]);
}

generated quantities{
  cov_matrix[J] Marg_cov = multiply_lower_tri_self_transpose(L_R);  
}
