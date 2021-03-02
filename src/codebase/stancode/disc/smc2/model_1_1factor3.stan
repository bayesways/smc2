data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> K;
  matrix[N,K] zz;
  int<lower=0, upper=1> D[N, J];
}

parameters {
  real beta1;
}

transformed parameters{
  matrix[N,J] y_latent;
  matrix[J,K] beta;
  vector[J] alpha;
  alpha = rep_vector(0, J);
  for (j in 1:J){
    for (k in 1:K)  beta[j,k] = beta1;
  }
  for (n in 1:N) y_latent[n,] = to_row_vector(alpha) + zz[n,] * beta';
}

model {
  beta1 ~ normal(0, 1);
  for (n in 1:N) D[n,] ~ bernoulli_logit(y_latent[n,]);
}
