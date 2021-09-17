data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  matrix[N,J] y;
  vector[J] sigma_prior;
}

transformed data{
  real<lower=0> c0 = 2.5;
}

generated quantities{
  matrix[J,K] beta;
  cov_matrix[J] Marg_cov;
  cov_matrix[J] Theta;
  vector<lower=0>[J] sigma_square;
  vector[J] alpha;
  vector<lower=0>[J] sigma;
  for(j in 1:J) alpha[j] = normal_rng(0, 10);
  for(j in 1:J) {
    for (k in 1:K) beta[j,k] = 0;
    }
  for(j in 1:J) sigma_square[j] = inv_gamma_rng(c0, (c0-1)/sigma_prior[j]);
  sigma = sqrt(sigma_square);
  Theta = diag_matrix(sigma_square);
  Marg_cov = beta * beta'+ Theta;  
}
