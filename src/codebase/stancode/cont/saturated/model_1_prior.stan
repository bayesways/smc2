data{
  int<lower=0> N;
  int<lower=0> J;
  vector[J] y[N];
}

transformed data{
  vector[J] zeros = rep_vector(0, J);
  cov_matrix[J] I_J = diag_matrix(rep_vector(1, J));
}

generated quantities{
  vector[J] alpha;
  cov_matrix[J] Marg_cov;
  for(j in 1:J) alpha[j] = normal_rng(0, 10);
  Marg_cov = inv_wishart_rng(J+6, I_J);
}
