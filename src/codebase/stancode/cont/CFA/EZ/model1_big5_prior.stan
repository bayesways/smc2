data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  matrix[N,J] y;
  vector[J] sigma_prior;
}

transformed data{
  vector[J] zeros = rep_vector(0, J);
  cov_matrix[J] I_J = diag_matrix(rep_vector(1, J));
  cov_matrix[K] I_K = diag_matrix(rep_vector(1, K));
  real<lower=0> c0 = 2.5;
}

generated quantities{
  matrix[J,K] beta;
  matrix[2,K] beta_free; // 2 free eleements per factor
  cov_matrix [K] Phi_cov;
  cov_matrix[J] Marg_cov;
  cov_matrix[J] Theta;
  vector<lower=0>[J] sigma_square;
  vector[J] alpha;
  vector<lower=0>[J] sigma;

  for(j in 1:J) alpha[j] = normal_rng(0, 10);

  // Create beta
  for (j in 1:2){
    for (k in 1:K) beta_free[j,k] = normal_rng(0, 1);
  }
  for(j in 1:J) {
    for (k in 1:K) beta[j,k] = 0;
    }
  // set ones
  for (k in 1:K) beta[1+3*(k-1), k] = 1;

  // set the free elements
  for (k in 1:K) beta[2+3*(k-1) : 3+3*(k-1), k] = beta_free[1:2,k];
  
  for(j in 1:J) sigma_square[j] = inv_gamma_rng(c0, (c0-1)/sigma_prior[j]);
  sigma = sqrt(sigma_square);
  Theta = diag_matrix(sigma_square);
  Phi_cov = inv_wishart_rng(J+4, I_K);  
  Marg_cov = beta * Phi_cov * beta'+ Theta;
}

