data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> D[N, J];
  matrix[N,K] zz;
}

transformed data{
  vector[K] zeros_K = rep_vector(0, K);
}

parameters {
  vector[J] alpha;
  matrix[3,K] beta_free; // 3 free eleements per factor
  cholesky_factor_corr[K] L_Phi;
}

transformed parameters{
  matrix[J,K] beta;
  matrix[N,J] yy;

  for(j in 1:J) {
    for (k in 1:K) beta[j,k] = 0;
  }
  // set the free elements
  for (k in 1:K) beta[1+3*(k-1) : 3+3*(k-1), k] = beta_free[1:3,k];

  for (n in 1:N) yy[n,] = to_row_vector(alpha) + zz[n,] * beta';
}
  
model {
  to_vector(beta_free) ~ normal(0, 1);
  to_vector(alpha) ~ normal(0, 10);
  L_Phi ~ lkj_corr_cholesky(2);
  for (n in 1:N) to_vector(zz[n,])  ~ multi_normal_cholesky(zeros_K, L_Phi);
  for (j in 1:J) D[, j] ~ bernoulli_logit(yy[, j]);
  
}

generated quantities{
  corr_matrix[K] Phi_cov = multiply_lower_tri_self_transpose(L_Phi);
}
