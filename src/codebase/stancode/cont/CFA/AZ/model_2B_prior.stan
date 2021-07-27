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
  cov_matrix[J] I_J = diag_matrix(rep_vector(1, J));
  real<lower=0> c0 = 2.5;
  cov_matrix [K] Phi_cov;
  Phi_cov = I_K;
}

generated quantities{
  matrix[J,K] beta;
  matrix[3,K] beta_free; // 2 free eleements per factor
  matrix[J-3,K] beta_zeros; // J-3 zero elements per factor
  cov_matrix[J] Marg_cov;
  cov_matrix[J] Theta;
  vector<lower=0>[J] sigma_square;
  vector[J] alpha;
  vector<lower=0>[J] sigma;
  cov_matrix[J] Omega;
  for(j in 1:J) alpha[j] = normal_rng(0, 10);

  for (j in 1:2){
    for (k in 1:K) beta_free[j,k] = normal_rng(0, 1);
  }
  for (j in 1:(J-3)){
    for (k in 1:K) beta_zeros[j,k] = normal_rng(0, 0.1);
  }
  // set the free elements
  for (k in 1:K) beta[1+3*(k-1) : 3+3*(k-1), k] = beta_free[1:3,k];
    // set the zero elements
  beta[4:6, 1] = beta_zeros[1:3, 1];
  beta[1:3, 2] = beta_zeros[1:3, 2]; 
  for(j in 1:J) sigma_square[j] = inv_gamma_rng(c0, (c0-1)/sigma_prior[j]);
  sigma = sqrt(sigma_square);
  Theta = diag_matrix(sigma_square);
  Marg_cov = beta * Phi_cov * beta'+ Theta;
  Omega = inv_wishart_rng(J+6, I_J);
}

