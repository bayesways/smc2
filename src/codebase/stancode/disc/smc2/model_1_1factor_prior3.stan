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
  real beta1;
  beta1 = normal_rng(0,1);
  alpha = rep_vector(0, J);
  for (j in 1:J){
    for (k in 1:K)  beta[j,k] = beta1;
  }
}
