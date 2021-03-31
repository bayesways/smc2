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
  matrix[N,K] zz;
  matrix[N,J] yy;

  for (j in 1:J) alpha[j] = normal_rng(0,10);
  for (j in 1:J){
    for (k in 1:K)  beta[j,k] = normal_rng(0,1);
  }
  for (n in 1:N)
  {
    for (k in 1:K)
    {
      zz[n,k] = normal_rng(0,1);
    }
  }
  for (n in 1:N) yy[n,] = to_row_vector(alpha) + zz[n,] * beta';

}
