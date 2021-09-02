data{
  int<lower=0> N;
  int<lower=0> J;
  vector[J] y[N];
}

transformed data{
  vector[J] zeros = rep_vector(0,J);
  matrix[J, J] I = diag_matrix(rep_vector(1, J));
}

generated quantities{
  vector[J] alpha;
  vector<lower=0>[J] sigma_square;
  cov_matrix[J] Marg_cov;
  for (j in 1:J) sigma_square[j] = inv_gamma_rng(1, 1);
  Marg_cov = diag_matrix(sigma_square);
  alpha = multi_normal_rng(zeros, Marg_cov);
}
