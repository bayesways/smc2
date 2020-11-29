data{
  int<lower=0> N;
  int<lower=0> J;
  int<lower=0> K;
  int<lower=0, upper=1> D[N,J];
}

transformed data{
  vector[J] zeros = rep_vector(1,J);
  matrix[J, J] I = diag_matrix(rep_vector(1, J));
}

generated quantities{
  vector[J] alpha;
  matrix[J,K] beta;
  for (j in 1:J) alpha[j] = normal_rng(0,1);
  for (j in 1:J){
    for (k in 1:K)  beta[j,k] = normal_rng(0,1);
  }
}
