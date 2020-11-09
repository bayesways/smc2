data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0, upper=1> DD[N, J];
}

parameters {
  vector[J] alpha;
  matrix[J,K] beta;
  matrix[N,K] zz;
}

transformed parameters{
  matrix[N,J] yy;
  for (n in 1:N) yy[n,] = to_row_vector(alpha) + zz[n,] * beta';
}
  
model {
  to_vector(beta) ~ normal(0, 1);
  to_vector(alpha) ~ normal(0, 10);
  to_vector(zz) ~ normal(0, 1);
  for (j in 1:J) DD[, j] ~ bernoulli_logit(yy[, j]);
}
