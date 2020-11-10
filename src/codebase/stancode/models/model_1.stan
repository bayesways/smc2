data {
  int<lower=1> N;
  int<lower=1> J;
  vector[N] z;
  matrix[N,J] y;
}

parameters {
  vector[J] alpha;
  vector[J] beta;
  vector<lower=0>[J] sigma;
}

transformed parameters{
  matrix[N,J] mu; 
  for (n in 1:N) mu[n,] = to_row_vector(alpha) + to_row_vector(z[n]*beta);

}

model {
  to_vector(beta) ~ normal(0, 1);
  alpha ~ normal(0, 10);
  sigma ~ cauchy(0,2.);
  for (n in 1:N) y[n,] ~ logistic(mu[n,], sigma);
}
