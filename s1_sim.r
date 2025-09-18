# ::: STUDY 1 -- SIMULATION ::: #

{
  rm(list = ls())
  library(loo)
  library(rstan)
  library(rstanarm)
  library(LaplacesDemon)
  library(tidyverse)
  options(scipen = 999)
  options(mc.cores = parallel::detectCores())
}

# CHTC command
args <- (commandArgs(TRUE))
if(length(args)==0){
  print("No arguments supplied.")
}
arguments <- commandArgs(trailingOnly=TRUE)
rep <- as.numeric(arguments[[1]])     
c <- as.numeric(arguments[[2]])       
ni <- as.numeric(arguments[[3]])      
nj <- as.numeric(arguments[[4]])      
icc <- as.numeric(arguments[[5]])     
sigma <- as.numeric(arguments[[6]])   

set.seed(c)

# DGP
gamma00 <- 400
gamma01 <- -20
gamma02 <- 6
gamma03 <- -2
gamma04 <- 8
u_0 <- sqrt(icc)        
w_0 <- sqrt(1-icc)
sigma <- sigma

x <- matrix(99, nrow = ni*nj, ncol = 4)
y <- matrix(99, nrow = ni*nj, ncol = 1) 
r <- matrix(99, nrow = ni*nj, ncol = 1) 
U <- matrix(99, nrow = ni*nj, ncol = 3)
W <- matrix(99, nrow = ni*nj, ncol = 4)
sim <- matrix(99, nrow = ni*nj, ncol = 9) 

# DGP
df <- dgf(ni, nj, gamma00, gamma01, gamma02, gamma03, gamma04, u_0, w_0, sigma)

# Leave-one-out pointwise densities
extract_lpd <- function(x){
  g <- x$pointwise[, "elpd_loo"]
  return(g)
}

# 15 candidate models
model_strings <- c(
  "y ~ x1 + (1|i)",
  "y ~ x2 + (1|i)",
  "y ~ x3 + (1|i)",
  "y ~ x4 + (1|i)",
  "y ~ x1 + x2 + (1|i)",
  "y ~ x1 + x3 + (1|i)",
  "y ~ x1 + x4 + (1|i)",
  "y ~ x2 + x3 + (1|i)",
  "y ~ x2 + x4 + (1|i)",
  "y ~ x3 + x4 + (1|i)",
  "y ~ x1 + x2 + x3 + (1|i)",
  "y ~ x1 + x2 + x4 + (1|i)",
  "y ~ x1 + x3 + x4 + (1|i)",
  "y ~ x2 + x3 + x4 + (1|i)",
  "y ~ x1 + x2 + x3 + x4 + (1|i)"  # Model 15: True DGM
)

var_names <- c(
  "x1", "x2", "x3", "x4", "x1_x2", "x1_x3", "x1_x4",
  "x2_x3", "x2_x4", "x3_x4", "x1_x2_x3", "x1_x2_x4",
  "x1_x3_x4", "x2_x3_x4", "x1_x2_x3_x4"
)

# Analysis
for (k in 1:15){
  # Fit Bayesian multilevel models
  a <- stan_lmer(as.formula(model_strings[k]), data = df, 
                 prior_intercept = student_t(3, 400, 10), 
                 prior_covariance = decov(scale = 0.50), 
                 iter = 10000, adapt_delta=.999, thin=10)
  
  # Extract log-likelihood
  b <- log_lik(a, merge_chains = FALSE)
  assign(paste0("bms_", var_names[k]), a)
  assign(paste0("loglik_", var_names[k]), b)
  
  # Check convergence
  e <- a$stan_summary[, c("n_eff", "Rhat")] %>% colMeans()
  assign(paste0("converge_", var_names[k]), e)
}

# Collect
f <- ls(pattern = "bms_x", all.names = T)
bms_all <- do.call(list, mget(f))

d <- ls(pattern = "loglik_x", all.names = T)
loglik_all <- do.call(list, mget(d))

# Compute LOO for all models
time_loo <- system.time(loo_bms <- lapply(loglik_all, loo, cores = 4))

# Compute ensemble weights
time_bs <- time_loo + system.time(
  w_bs <- loo::loo_model_weights(loo_bms, method = "stacking"))

time_pbma <- time_loo + system.time(
  w_pbma <- loo::loo_model_weights(loo_bms, method = "pseudobma", BB = FALSE))

time_pbmabb <- time_loo + system.time(
  w_pbmabb <- loo::loo_model_weights(loo_bms, method = "pseudobma"))

# Generate predictions
n_draws <- nrow(as.matrix(bms_all[[1]]))

# BS
ypred_bs <- matrix(NA, nrow = n_draws, ncol = nobs(bms_all[[1]]))
for (d in 1:n_draws) {
  k <- sample(1:length(w_bs), size = 1, prob = w_bs)
  ypred_bs[d, ] <- posterior_predict(bms_all[[k]], draws = 1)
}

y_bs <- colMeans(ypred_bs)
d1 <- density(y_bs, kernel = c("gaussian"))$y
d0 <- density(df$y, kernel = c("gaussian"))$y
kld1 <- KLD(d1, d0)$sum.KLD.py.px

# PBMA
ypred_pbma <- matrix(NA, nrow = n_draws, ncol = nobs(bms_all[[1]]))
for (d in 1:n_draws) {
  k <- sample(1:length(w_pbma), size = 1, prob = w_pbma)
  ypred_pbma[d, ] <- posterior_predict(bms_all[[k]], draws = 1)
}

y_pbma <- colMeans(ypred_pbma)
d2 <- density(y_pbma, kernel = c("gaussian"))$y
kld2 <- KLD(d2, d0)$sum.KLD.py.px

# PBMA+
ypred_pbmabb <- matrix(NA, nrow = n_draws, ncol = nobs(bms_all[[1]]))
for (d in 1:n_draws) {
  k <- sample(1:length(w_pbmabb), size = 1, prob = w_pbmabb)
  ypred_pbmabb[d, ] <- posterior_predict(bms_all[[k]], draws = 1)
}

y_pbmabb <- colMeans(ypred_pbmabb)
d3 <- density(y_pbmabb, kernel = c("gaussian"))$y
kld3 <- KLD(d3, d0)$sum.KLD.py.px

# BHS
X <- df %>% select(x1, x2, x3, x4)
N <- nrow(X)
d <- ncol(X)
lpd_point <- do.call(cbind, lapply(loo_bms, extract_lpd))
K <- ncol(lpd_point)

dt_bhs <- list(N = N, d = d, K = K, X = X, 
               lpd_point = lpd_point, tau_mu = 1, tau_con = 1)

time_bhs <- system.time(
  fit_bhs <- stan("bhs_con.stan", data = dt_bhs, iter = 10000, save_warmup = FALSE))

wts_bhs <- rstan::extract(fit_bhs, pars = 'w')$w
w_bhs <- apply(wts_bhs, c(2,3), mean)
w_bhs_m <- as.matrix(apply(wts_bhs, 3, mean))

ypred_bhs <- matrix(NA, nrow = n_draws, ncol = nobs(bms_all[[1]]))
for (d in 1:n_draws) {
  k <- sample(1:15, size = 1, prob = w_bhs_m)
  ypred_bhs[d, ] <- posterior_predict(bms_all[[k]], draws = 1)
}

y_bhs <- colMeans(ypred_bhs)
d4 <- density(y_bhs, kernel = c("gaussian"))$y
kld4 <- KLD(d4, d0)$sum.KLD.py.px

# Combine results
ws <- data.frame(as.matrix(w_bs), as.matrix(w_pbma), as.matrix(w_pbmabb), w_bhs_m)
klds <- rbind(kld1, kld2, kld3, kld4) %>% as.data.frame()

# Timing
time <- rbind(t(data.matrix(time_bs)), t(data.matrix(time_pbma)), 
              t(data.matrix(time_pbmabb)), t(data.matrix(time_bhs))) %>% as.data.frame()

rnames <- c("bs","pbma", "pbmabb", "bhs")  
rownames(klds) <- rnames
colnames(ws) <- rnames
rownames(time) <- rnames

# Save
assign(paste0("rep_", rep, "_seed", c, "_", ni, "_", nj, "_sd", icc, "_sigma", sigma, "_ws"), ws)
assign(paste0("rep_", rep, "_seed", c, "_", ni, "_", nj, "_sd", icc, "_sigma", sigma, "_kld"), klds)
assign(paste0("rep_", rep, "_seed", c, "_", ni, "_", nj, "_sd", icc, "_sigma", sigma, "_time"), time)

sw <- paste0("rep_", rep, "_seed", c, "_", ni, "_", nj, "_sd", icc, "_sigma", sigma, "_ws")
dlk <- paste0("rep_", rep, "_seed", c, "_", ni, "_", nj, "_sd", icc, "_sigma", sigma, "_kld")
emit <- paste0("rep_", rep, "_seed", c, "_", ni, "_", nj, "_sd", icc, "_sigma", sigma, "_time")

save(list = c(sw,dlk,emit), file = paste0("rep_", rep, "_seed", c, "_", ni, "_", nj, "_sd", icc, "_sigma", sigma, "_out.RData"))

file.copy(from = paste0("rep_", rep, "_seed", c, "_", ni, "_", nj, "_sd", icc, "_sigma", sigma, "_out.RData"),
          to = paste0("/staging/mhuang233/rep_", rep, "_seed", c, "_", ni, "_", nj, "_sd", icc, "_sigma", sigma, "_out.RData"))

file.remove(paste0("rep_", rep, "_seed", c, "_", ni, "_", nj, "_sd", icc, "_sigma", sigma, "_out.RData"))
