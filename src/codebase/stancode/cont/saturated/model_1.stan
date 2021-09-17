data{
  int<lower=0> N;
  int<lower=0> J;
  vector[J] y[N];
}

transformed data{
  vector[J] zeros = rep_vector(0, J);
  cov_matrix[J] I_J = diag_matrix(rep_vector(1, J));
}

parameters {
  vector[J] alpha;
  cov_matrix[J] Marg_cov;
}

model{
  Marg_cov ~ inv_wishart(J+4, I_J);
  to_vector(alpha) ~ normal(0, 10);
  for (n in 1:N) y[n,] ~ multi_normal(alpha, Marg_cov);
}
