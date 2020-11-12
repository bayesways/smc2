data {
  int<lower=1> N;
  int<lower=1> J;
  vector[N] z;
  int<lower=0, upper=1> D[N, J];
}

parameters {
  vector[J] alpha;
  vector[J] beta;
}

transformed parameters{
  matrix[N,J] y_latent; 
  for (n in 1:N) y_latent[n,] = to_row_vector(alpha) + to_row_vector(z[n]*beta);
}

model {
  beta ~ normal(0, 1);
  alpha ~ normal(0, 1);
  for (n in 1:N) D[n,] ~ bernoulli_logit(y_latent[n,]);
}
