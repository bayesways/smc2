data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=0, upper=1> D[N, J];
}

parameters {
  vector[J] alpha;
  vector[J] beta;
  vector[N] zz;
}

transformed parameters{
  matrix[N,J] yy;
  for (n in 1:N) yy[n,] = to_row_vector(alpha) + zz[n] * beta';
}
  
model {
  beta ~ normal(0, 1);
  alpha ~ normal(0, 10);
  zz ~ normal(0, 1);
  for (j in 1:J) D[, j] ~ bernoulli_logit(yy[, j]);
}

generated quantities{
  matrix[J,J] betabeta;
  betabeta = beta * beta';
}