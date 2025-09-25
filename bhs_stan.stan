//
// BHS stan
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;
  int<lower=0> d;
  int<lower=0> d_discrete;
  int<lower=2> K;
  matrix[N, d] X;
  matrix[N, K] lpd_point;
  real<lower = 0> tau_mu;
  real<lower = 0> tau_discrete;
  real<lower = 0> tau_con;
}

transformed data{
  matrix[N, K] exp_lpd_point = exp(lpd_point);
}

parameters {
  vector[K-1] mu;
  real mu_0;
  vector<lower = 0>[K-1] sigma;
  vector<lower = 0>[K-1] sigma_con;
  vector[d-d_discrete] beta_con[K-1];
  vector[d_discrete] tau[K-1];
}

transformed parameters{
  vector[d] beta[K-1];
  simplex[K] w[N];
  matrix[N,K] f;
  for (k in 1:(K-1))
  beta[k] = append_row(mu_0*tau_mu + mu[k]*tau_mu + sigma[k]*tau[k], sigma_con[k]* beta_con[k]);
  for (k in 1:(K-1))
  f[,k] = X*beta[k];
  f[,K] = rep_vector(0,N);
  for(n in 1:N)
  w[n] = softmax(to_vector(f[n, 1:K]));
}

model {
  for (k in 1:(K-1)){
    tau[k] ~ normal(0,1);
    beta_con[k] ~ normal(0,1);
  }
  mu ~ normal(0,1);
  mu_0 ~ normal(0,1);
  sigma ~ normal(0, tau_discrete);
  sigma_con ~ normal(0, tau_con);
  for (i in 1:N)
  target += log(exp_lpd_point[i, ]*w[i]);
}

generated quantities{
  vector[N] log_lik;
  for (i in 1:N)
  log_lik[i] = log(exp_lpd_point[i, ]*w[i]);
}



