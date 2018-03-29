 // Basic Poisson glm
 
 data {
   // Define variables in data
   // Number of observations (an integer)
   int<lower=0> N;
   int<lower=0> Nc;	
 
   // Covariates
   int <lower=0, upper=1> intercept[N];
   int <lower=0, upper=1> interceptc[Nc];
   int <lower=-1, upper=12> x[N];
   int <lower=-1, upper=12> xc[Nc];
   
   // Count outcome
   int<lower=0> y[N];
   int <lower=0, upper= 4> yc[Nc];
   int <upper=4> U;
 }
 
 parameters {
   // Define parameters to estimate
   real beta1;
   real beta2;
   
 }
 
 transformed parameters  {
   //
   real lp[N];
   real <lower=0> mu[N];
   real <lower=0> muc[Nc];
   real lpc[Nc];

   for (i in 1:N) {
     // Linear predictor
     lp[i] = beta1 + beta2*x[i];
 
     // Mean
     mu[i] = exp(lp[i]);
   }
   for (i in 1:Nc) {
     // Linear predictor
     lpc[i] = beta1 + beta2*xc[i];
 
     // Mean
     muc[i] = exp(lpc[i]);
   }
 }
 
 model {
   // Prior part of Bayesian inference
   beta1~normal(0,1);
   beta2~normal(0,1);
 
 
   // Likelihood part of Bayesian inference
   y ~ poisson(mu);
   target += Nc*poisson_lcdf(U|muc);
   

 }
