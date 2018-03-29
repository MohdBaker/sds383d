 // Basic Poisson glm
 
 data {
   // Define variables in data
   // Number of observations (an integer)
   int<lower=0> N;

 
   // Covariates
   int <lower=0, upper=1> intercept[N];
   int <lower=-1, upper=12> x[N];
   int <lower=0, upper=1> x_sex[N];
   int <lower=-12, upper=12> x_int[N];
   
   // Count outcome
   int<lower=0> y[N];
 }
 
 parameters {
   // Define parameters to estimate
   real beta1;
   real beta2;
   real beta3;
   real beta4;
 }
 
 transformed parameters  {
   //
   real lp[N];
   real <lower=0> mu[N];
 
   for (i in 1:N) {
     // Linear predictor
     lp[i] <- beta1 + beta2*x[i]+ beta3*x_sex[i]+beta4*x_int[i];
 
     // Mean
     mu[i] <- exp(lp[i]);
   }
 }
 
 model {
   // Prior part of Bayesian inference
   beta1~normal(0,1);
   beta2~normal(0,1);
   beta3~normal(0,1);
   beta4~normal(0,1);
 
   // Likelihood part of Bayesian inference
   y ~ poisson(mu);
 }
