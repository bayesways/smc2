data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  matrix[N,J] y;
  vector[J] sigma_prior;
}

transformed data{
  vector[J] zeros = rep_vector(0, J);
  cov_matrix[K] I_K = diag_matrix(rep_vector(1, K));
  real<lower=0> c0 = 2.5;
}

parameters {
  vector<lower=0>[J] sigma_square;
  vector[J] alpha;
  matrix[3,K] beta_free; // 3 free eleements per factor
  cholesky_factor_corr[K] Phi_L;
}

transformed parameters{
  cov_matrix[J] Theta;
  matrix[J,K] beta;
  cov_matrix[J] Marg_cov;
  corr_matrix[K] Phi_cov;
  
  Phi_cov = multiply_lower_tri_self_transpose(Phi_L);
  Theta = diag_matrix(sigma_square);

  for(j in 1:J) {
    for (k in 1:K) beta[j,k] = 0;
    }
  // set the free elements
  for (k in 1:K) beta[1+3*(k-1) : 3+3*(k-1), k] = beta_free[1:3,k];
  Marg_cov = beta * Phi_cov * beta'+ Theta;
}

model {
  to_vector(beta_free) ~ normal(0, 1);
  to_vector(alpha) ~ normal(0, 10);
  for(j in 1:J) sigma_square[j] ~ inv_gamma(c0, (c0-1)/sigma_prior[j]);
  Phi_L ~ lkj_corr_cholesky(2);
  for (n in 1:N){
    y[n, ] ~ multi_normal(alpha,  Marg_cov);
  }
  
}

generated quantities{
  vector<lower=0>[J] sigma = sqrt(sigma_square);
}

