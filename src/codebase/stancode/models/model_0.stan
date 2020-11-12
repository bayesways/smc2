data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=0, upper=1> D[N, J];
}

parameters {
  vector[J] alpha;
  vector<lower=0>[J-1] beta_free;
  vector[N] zz;
}

transformed parameters{
  matrix[N,J] yy;
  vector[J] beta = append_row(1, beta_free);
  for (n in 1:N) yy[n,] = to_row_vector(alpha) + zz[n] * beta';
}
  
model {
  beta_free ~ normal(0, 1);
  alpha ~ normal(0, 10);
  zz ~ normal(0, 1);
  for (j in 1:J) D[, j] ~ bernoulli_logit(yy[, j]);
}
