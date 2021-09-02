data{
  int<lower=0> N;
  int<lower=0> J;
  vector[J] y[N];

}

parameters {
  vector[J] alpha;
  vector<lower=0>[J] sigma_square;
}

transformed parameters {
  cov_matrix[J] Marg_cov = diag_matrix(sigma_square); 
}

model{
  sigma_square ~ inv_gamma(1,1);
  alpha ~ multi_normal(alpha, Marg_cov);
  for (n in 1:N) y[n,] ~ multi_normal(alpha, Marg_cov);
}
