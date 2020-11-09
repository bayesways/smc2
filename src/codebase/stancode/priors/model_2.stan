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
  vector<lower=0>[J] sigma;
  cholesky_factor_corr[J] L_R; 
  cov_matrix[J] Marg_cov;
  alpha = multi_normal_rng(zeros, 25*I);
  for (j in 1:J) sigma[j] = fabs(cauchy_rng(0, 2.));
  L_R = lkj_corr_cholesky_rng(J, 2.);
  Marg_cov = multiply_lower_tri_self_transpose(diag_pre_multiply(sigma, L_R));  
}
