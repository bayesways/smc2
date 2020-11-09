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
  cholesky_factor_corr[J] L_R; 
  cov_matrix[J] Marg_cov;
  alpha = multi_normal_rng(zeros,25*I);
  L_R = lkj_corr_cholesky_rng(J, 2.);
  Marg_cov = multiply_lower_tri_self_transpose(L_R);  
}
