data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> D[N, J];
}

parameters {
  vector[J] alpha;
  matrix[J,K] beta;
  matrix[N,K] z;
}

transformed parameters{
  matrix[N,J] y;
  for (n in 1:N) y[n,] = to_row_vector(alpha) + z[n,] * beta';
}
  
model {
  to_vector(beta) ~ normal(0, 1);
  to_vector(alpha) ~ normal(0, 10);
  to_vector(z) ~ normal(0, 1);
  for (j in 1:J) D[, j] ~ bernoulli_logit(y[, j]);
}
