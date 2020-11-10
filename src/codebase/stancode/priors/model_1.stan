data {
  int<lower=1> N;
  int<lower=1> J;
  matrix[N,J] y;
}

generated quantities{
  vector[J] alpha;
  vector[J] beta;
  vector<lower=0>[J] sigma;

  for(j in 1:J) beta[j] = normal_rng(0, 1);
  for(j in 1:J) alpha[j] = normal_rng(0, 10);
  for(j in 1:J) sigma[j] = fabs(cauchy_rng(0,2.));
}
