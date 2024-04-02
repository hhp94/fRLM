data {
  int<lower=1> n; // Length of y vector
  int<lower=1> L; // Number of basis function for phi
  int<lower=1> K; // Number of knots
  int<lower=1> d; // Number of covariates and intercept
  int<lower=1> Nmax; // Max number of observations for the one exposure
  vector[n] y; // Vector Y
  matrix[n,d] C; // Vector of covariates
  array[n] matrix[Nmax, K] phi_mat;  // Nested array of length N where each element is a Nmax * K matrix
  matrix[K, L] J; // matrix of dimension K * L
  // DATA for gaussian processes:
  array[n] int<lower=1> Nvec; // Vector that counts how many observations of the exposures for each row. Min 1 observation
  matrix[n, Nmax] tobs; // Matrix of time variables for each exposure observation. Implied real
  matrix[n, Nmax] xobs; // Vector of length n, where each element is a vector of length Nmax
}

parameters {
  real delta; // Effect size of the exposure function
  vector[d] alpha; // Effect sizes of the covariates (control)
  simplex[L] beta; // Effect sizes of the different Z component (bsplines of time * Xhat)
  real<lower=0> sigma; // Random error
  real<lower=0> sigma_x; // Gaussian Process Random Error
  // GP parameters:
  matrix[n, K] xi; // matrix of mean value of n*K. Ragged array matrix
  array[K] real<lower=0> sigma_xi;
}

model {
  // GP Prior distribution:
  for(i in 1:n){
    xi[i] ~ normal(0, sigma_xi);
  }
  // These sigma priors are probably not the best coded. Also should probably do non-centered
  sigma_xi ~ lognormal(0, 1);
  sigma_x ~ lognormal(0, 1);
  // fRLM priors:
  delta ~ normal(0, 10);
  // Should probably relax this prior for dirichlet
  beta ~ dirichlet( rep_vector( 1.0 / L, L ) );
  sigma ~ lognormal(0, 1);
  alpha ~ std_normal(); // Prior for alpha. Probably too tight for unstandardized y
  // GP likelihood:
  for(i in 1:n){
    // For each row, the xobs runs from the first column to the last non masked value (observation)
    // This is modeled as a multivariate normal of mean phi_mat * transpose of xi which are parameters
    xobs[i,:Nvec[i]] ~ normal(phi_mat[i][:Nvec[i]] * xi[i]', sigma_x);
  }
  //The likelihood of the fRLM
  for (i in 1:n){
    y[i] ~ normal( delta * xi[i] * J * beta + C[i] * alpha, sigma );
  }
}

