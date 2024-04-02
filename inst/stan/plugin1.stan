data {
  int<lower=1> N;
  int<lower=1> L;
  int<lower=1> D;
  vector[N] y;
  matrix[N,L] eta;
  real<lower=0> beta_par;
  real<lower=0> alpha_par;
  matrix[N,D] C;
}

parameters {
  real delta;
  vector[D] alpha;
  simplex[L] beta;
  real<lower=0> sigma;
}

model {
  // the priors
  delta ~ normal(0, 10);
  beta ~ dirichlet( rep_vector( beta_par / L, L ) );
  sigma ~ lognormal(0,1);
  for (j in 1:D) {
    alpha[j] ~ normal(0, alpha_par);
  }
  //The likelihood
  for (i in 1:N) {
    y[i] ~ normal( delta * eta[i] * beta + C[i] * alpha, sigma);
  }
}
