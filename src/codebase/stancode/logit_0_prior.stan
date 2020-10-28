data{
  int<lower=0> N;
  int<lower=0> J;
  int<lower=0, upper=1> y[N,J];
}

transformed data{
  vector[J] zeros = rep_vector(1,J);
  matrix[J, J] I = diag_matrix(rep_vector(1, J));
}

generated quantities{
  vector[J] z[N];
  vector[J] alpha;
  cholesky_factor_corr[J] L_R; 
  cov_matrix[J] Marg_cov;
  for (n in 1:N) z[n,] = multi_normal_rng(zeros,I);
  alpha = multi_normal_rng(zeros,10*I);
  L_R = lkj_corr_cholesky_rng(J, 2.);
  Marg_cov = multiply_lower_tri_self_transpose(L_R);  
}
