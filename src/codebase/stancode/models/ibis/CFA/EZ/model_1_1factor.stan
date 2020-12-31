data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> K;
  matrix[N,K] zz;
  int<lower=0, upper=1> D[N, J];
}

parameters {
  vector[J] alpha;
  matrix[J,K] beta;
}

transformed parameters{
  matrix[N,J] y_latent; 
  for (n in 1:N) y_latent[n,] = to_row_vector(alpha) + zz[n,] * beta';
}

model {
  to_vector(beta) ~ normal(0, 1);
  alpha ~ normal(0, 1);
  for (n in 1:N) D[n,] ~ bernoulli_logit(y_latent[n,]);
}
