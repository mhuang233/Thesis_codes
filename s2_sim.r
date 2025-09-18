
# ::: STUDY 2: SIMULATION ::: #
{
  rm(list = ls())
  library(MASS)
  library(loo)
  library(rstan)
  library(flexBART)
  library(BART)
  library(tidyverse)
  options(mc.cores = parallel::detectCores())
  rstan_options(auto_write = TRUE)
}

# chtc
args <- commandArgs(TRUE)
if(length(args) == 0) {
  print("No arguments supplied.")
}
arguments <- commandArgs(trailingOnly=TRUE)
icc <- as.numeric(arguments[[1]])      
sigma <- as.numeric(arguments[[2]])    
f <- as.numeric(arguments[[3]])        
seed <- as.numeric(arguments[[4]])     

set.seed(seed)

# Load
load("inputs.RData")

# Data preparation
dt <- dt %>%
  mutate(
    x1 = escs,           
    x2 = school_escs,     
    x3 = school_bullying, 
    x4 = school_immig,   
    x5 = teachint,       
    x6 = adaptivity,     
    x7 = beingbullied,   
    x8 = staffshort,     
    x9 = immig_num,      
    schoolid_map = match(schoolid, unique(schoolid)) 
  )

uni_sch <- dt$schoolid %>% unique()
nj <- length(uni_sch)
tau00 <- (icc * sigma^2) / (1 - icc)  

# Fixed effects
gamma00 <- 479.79; gamma01 <- 24.96; gamma02 <- 19.30; gamma03 <- 4.69
gamma04 <- 6.36; gamma05 <- 9.25; gamma06 <- 2.58; gamma07 <- -1.29
gamma08 <- 0.65; gamma09 <- -2.10

# Covariance T = tau00 * D * rho * D
d1 <- 0.12; d5 <- 0.45; d6 <- 0.30; d7 <- 0.20; d9 <- 0.90
D_mat <- diag(c(1, sqrt(d1), sqrt(d5), sqrt(d6), sqrt(d7), sqrt(d9)))

# Correlation matrix
rho <- matrix(c(
  1.00, -0.15, -0.10,  0.05, -0.12, -0.08,
  -0.15,  1.00,  0.18, -0.20, -0.08,  0.10,
  -0.10,  0.18,  1.00, -0.15, -0.10,  0.05,
  0.05, -0.20, -0.15,  1.00,  0.40, -0.10,
  -0.12, -0.08, -0.10,  0.40,  1.00, -0.05,
  -0.08,  0.10,  0.05, -0.10, -0.05,  1.00), 
  nrow = 6, byrow = T)

T_mat <- tau00 * D_mat %*% rho %*% D_mat

# School-level random effects
ujs <- mvrnorm(n = nj, mu = rep(0, 6), Sigma = T_mat)
colnames(ujs) <- c("u0j", "u1j", "u5j", "u6j", "u7j", "u9j")

school_effects <- data.frame(schoolid_map = 1:nj, ujs)
dt <- dt %>% left_join(school_effects, by = "schoolid_map")

# Simulated outcomes
df_sim <- dt %>%
  mutate(
    y_fix = gamma00 + gamma01 * x1 + gamma02 * x2 + gamma03 * x3 + 
      gamma04 * x4 + gamma05 * x5 + gamma06 * x6 + gamma07 * x7 + 
      gamma08 * x8 + gamma09 * x9,
    y_ran = u0j + u1j * x1 + u5j * x5 + u6j * x6 + u7j * x7 + u9j * x9,
    err = rnorm(n(), mean = 0, sd = sigma),
    y_sim = y_fix + y_ran + err
  ) %>%
  select(childid, schoolid_map, y_sim, x1:x9, u0j:u9j, err) %>%
  as.data.frame()

# 10-fold CV
train_index <- list()
test_index <- list()

for (fold in 1:10) {
  set.seed(233 + fold)
  
  # School-level split
  school_ids <- 1:nj
  schools_shuffled <- sample(school_ids)
  n_train_schools <- round(length(school_ids) * 0.8)
  
  train_schools <- schools_shuffled[1:n_train_schools]
  test_schools <- schools_shuffled[(n_train_schools + 1):length(school_ids)]
  
  # Student-level split within school groups
  train_school_data <- df_sim %>% filter(schoolid_map %in% train_schools)
  train_students <- sample(train_school_data$childid, 
                           size = round(nrow(train_school_data) * 0.8))
  
  test_school_data <- df_sim %>% filter(schoolid_map %in% test_schools)
  test_students <- sample(test_school_data$childid,
                          size = round(nrow(test_school_data) * 0.8))
  
  train_index[[fold]] <- which(df_sim$childid %in% train_students)
  test_index[[fold]] <- which(df_sim$childid %in% test_students)
}

save(dt, train_index, test_index, df_sim,
     file = paste0("sim_math_icc_", icc, "_sigma_", sigma, ".RData"))

# Extract
train_idx <- train_index[[f]]
test_idx <- test_index[[f]]
df_train <- df_sim[train_idx, ]
df_test <- df_sim[test_idx, ]

N_train <- nrow(df_train)
N_test <- nrow(df_test)

# School mappings for Stan
train_schools <- sort(unique(df_train$schoolid_map))
test_schools <- sort(unique(df_test$schoolid_map))
J_train <- length(train_schools)
J_test <- length(test_schools)

train_school_mapping <- data.frame(
  schoolid_original = train_schools,
  schoolid_stan = 1:J_train
)

test_school_mapping <- data.frame(
  schoolid_original = test_schools,
  schoolid_stan = 1:J_test
)

df_train <- df_train %>%
  left_join(train_school_mapping, by = c("schoolid_map" = "schoolid_original")) %>%
  rename(school_stan = schoolid_stan)

df_test <- df_test %>%
  left_join(test_school_mapping, by = c("schoolid_map" = "schoolid_original")) %>%
  rename(school_stan = schoolid_stan)

avg_y <- mean(df_train$y_sim)
sd_y <- sd(df_train$y_sim)
df_train$std_y <- (df_train$y_sim - avg_y) / sd_y
df_test$std_y <- (df_test$y_sim - avg_y) / sd_y

predictor_vars <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9")
X_train_raw <- as.matrix(df_train[, predictor_vars])
X_test_raw <- as.matrix(df_test[, predictor_vars])

X_means <- colMeans(X_train_raw)
X_sds <- apply(X_train_raw, 2, sd)
X_sds[X_sds < 1e-10] <- 1

X_train <- scale(X_train_raw, center = X_means, scale = X_sds)
X_test <- scale(X_test_raw, center = X_means, scale = X_sds)

# Stan data
stan_data <- list(
  N_train = N_train, J_train = J_train,
  std_y_train = df_train$std_y,
  avg_y = avg_y, sd_y = sd_y,
  X_train = X_train,
  school_train = df_train$school_stan,
  N_test = N_test, J_test = J_test,
  X_test = X_test,
  school_test = df_test$school_stan
)

# Stan models
model_specs <- list(
  # Model 1: Random intercept only
  list(code = "
    parameters {
      real gamma0; vector[9] gammas; vector[J_train] u0;
      real<lower=0> tau; real<lower=0> sigma;
    }
    model {
      gamma0 ~ normal(0, 1); gammas ~ normal(0, 0.5);
      tau ~ exponential(3); sigma ~ exponential(2);
      u0 ~ normal(0, tau);
      vector[N_train] mu_train;
      for (n in 1:N_train) {
        int j = school_train[n];
        mu_train[n] = gamma0 + X_train[n] * gammas + u0[j];
      }
      std_y_train ~ normal(mu_train, sigma);
    }
    ", random_effects = "u0[j]"),
  
  # Model 2: Random slope for x1 (ESCS)
  list(code = "
    parameters {
      real gamma0; vector[9] gammas; matrix[2, J_train] u_raw;
      vector<lower=0>[2] tau; cholesky_factor_corr[2] L_u; real<lower=0> sigma;
    }
    transformed parameters {
      matrix[J_train, 2] u = (diag_pre_multiply(tau, L_u) * u_raw)';
    }
    model {
      gamma0 ~ normal(0, 1); gammas ~ normal(0, 0.5);
      tau ~ exponential(3); L_u ~ lkj_corr_cholesky(1); sigma ~ exponential(2);
      to_vector(u_raw) ~ std_normal();
      vector[N_train] mu_train;
      for (n in 1:N_train) {
        int j = school_train[n];
        mu_train[n] = gamma0 + X_train[n] * gammas + u[j, 1] + u[j, 2] * X_train[n, 1];
      }
      std_y_train ~ normal(mu_train, sigma);
    }
    ", random_effects = "u[j, 1] + u[j, 2] * X_train[n, 1]"),
  
  # Model 3: Random slope for x5 (TEACHINT)
  list(code = "
    parameters {
      real gamma0; vector[9] gammas; matrix[2, J_train] u_raw;
      vector<lower=0>[2] tau; cholesky_factor_corr[2] L_u; real<lower=0> sigma;
    }
    transformed parameters {
      matrix[J_train, 2] u = (diag_pre_multiply(tau, L_u) * u_raw)';
    }
    model {
      gamma0 ~ normal(0, 1); gammas ~ normal(0, 0.5);
      tau ~ exponential(3); L_u ~ lkj_corr_cholesky(1); sigma ~ exponential(2);
      to_vector(u_raw) ~ std_normal();
      vector[N_train] mu_train;
      for (n in 1:N_train) {
        int j = school_train[n];
        mu_train[n] = gamma0 + X_train[n] * gammas + u[j, 1] + u[j, 2] * X_train[n, 5];
      }
      std_y_train ~ normal(mu_train, sigma);
    }
    ", random_effects = "u[j, 1] + u[j, 2] * X_train[n, 5]"),
  
  # Model 4: Random slope for x6 (ADAPTIVITY)
  list(code = "
    parameters {
      real gamma0; vector[9] gammas; matrix[2, J_train] u_raw;
      vector<lower=0>[2] tau; cholesky_factor_corr[2] L_u; real<lower=0> sigma;
    }
    transformed parameters {
      matrix[J_train, 2] u = (diag_pre_multiply(tau, L_u) * u_raw)';
    }
    model {
      gamma0 ~ normal(0, 1); gammas ~ normal(0, 0.5);
      tau ~ exponential(3); L_u ~ lkj_corr_cholesky(1); sigma ~ exponential(2);
      to_vector(u_raw) ~ std_normal();
      vector[N_train] mu_train;
      for (n in 1:N_train) {
        int j = school_train[n];
        mu_train[n] = gamma0 + X_train[n] * gammas + u[j, 1] + u[j, 2] * X_train[n, 6];
      }
      std_y_train ~ normal(mu_train, sigma);
    }
    ", random_effects = "u[j, 1] + u[j, 2] * X_train[n, 6]"),
  
  # Model 5: Random slope for x7 (BEINGBULLIED)
  list(code = "
    parameters {
      real gamma0; vector[9] gammas; matrix[2, J_train] u_raw;
      vector<lower=0>[2] tau; cholesky_factor_corr[2] L_u; real<lower=0> sigma;
    }
    transformed parameters {
      matrix[J_train, 2] u = (diag_pre_multiply(tau, L_u) * u_raw)';
    }
    model {
      gamma0 ~ normal(0, 1); gammas ~ normal(0, 0.5);
      tau ~ exponential(3); L_u ~ lkj_corr_cholesky(1); sigma ~ exponential(2);
      to_vector(u_raw) ~ std_normal();
      vector[N_train] mu_train;
      for (n in 1:N_train) {
        int j = school_train[n];
        mu_train[n] = gamma0 + X_train[n] * gammas + u[j, 1] + u[j, 2] * X_train[n, 7];
      }
      std_y_train ~ normal(mu_train, sigma);
    }
    ", random_effects = "u[j, 1] + u[j, 2] * X_train[n, 7]"),
  
  # Model 6: Random slope for x9 (IMMIG)
  list(code = "
    parameters {
      real gamma0; vector[9] gammas; matrix[2, J_train] u_raw;
      vector<lower=0>[2] tau; cholesky_factor_corr[2] L_u; real<lower=0> sigma;
    }
    transformed parameters {
      matrix[J_train, 2] u = (diag_pre_multiply(tau, L_u) * u_raw)';
    }
    model {
      gamma0 ~ normal(0, 1); gammas ~ normal(0, 0.5);
      tau ~ exponential(3); L_u ~ lkj_corr_cholesky(1); sigma ~ exponential(2);
      to_vector(u_raw) ~ std_normal();
      vector[N_train] mu_train;
      for (n in 1:N_train) {
        int j = school_train[n];
        mu_train[n] = gamma0 + X_train[n] * gammas + u[j, 1] + u[j, 2] * X_train[n, 9];
      }
      std_y_train ~ normal(mu_train, sigma);
    }
    ", random_effects = "u[j, 1] + u[j, 2] * X_train[n, 9]")
)

# Fit all 6 candidate models
models_results <- list()
total_model_time <- 0

for (m in 1:6) {
  time_start <- Sys.time()
  
  if (m == 1) {
    init_func <- function(chain_id) {
      list(gamma0 = rnorm(1, 0, 0.1), gammas = rnorm(9, 0, 0.1),
           u0 = rnorm(J_train, 0, 0.1), tau = abs(rnorm(1, 0.3, 0.05)),
           sigma = abs(rnorm(1, 1, 0.1)))
    }
  } else {
    init_func <- function(chain_id) {
      list(gamma0 = rnorm(1, 0, 0.1), gammas = rnorm(9, 0, 0.1),
           u_raw = matrix(rnorm(2 * J_train, 0, 0.1), nrow = 2),
           tau = abs(rnorm(2, 0.3, 0.05)), L_u = diag(2),
           sigma = abs(rnorm(1, 1, 0.1)))
    }
  }
  
  # Fit
  model_obj <- stan_model(model_code = model_specs[[m]]$code)
  fit <- sampling(
    object = model_obj, data = stan_data,
    iter = 3000, chains = 4, warmup = 2000,
    control = list(adapt_delta = 0.85, max_treedepth = 11),
    init = init_func, seed = seed, refresh = 0
  )
  
  # Extract
  posterior_list <- rstan::extract(fit)
  models_results[[m]] <- list(
    loglik_train = posterior_list$log_lik_train,
    loglik_test = posterior_list$log_lik_test,
    ypred_train = posterior_list$y_pred_train,
    ypred_test = posterior_list$y_pred_test
  )
  
  total_model_time <- total_model_time + as.numeric(difftime(Sys.time(), time_start, units = "secs"))
}

# Combine
Y_train <- df_train$y_sim
Y_test <- df_test$y_sim
successful_models <- 1:6
n_models <- length(successful_models)

model_preds_train <- sapply(successful_models, function(m) colMeans(models_results[[m]]$ypred_train))
model_preds_test <- sapply(successful_models, function(m) colMeans(models_results[[m]]$ypred_test))
model_full_train <- lapply(successful_models, function(m) models_results[[m]]$ypred_train)
model_full_test <- lapply(successful_models, function(m) models_results[[m]]$ypred_test)

X_train_std <- as.data.frame(X_train)
X_test_std <- as.data.frame(X_test)

# BMA
time_start_bma <- Sys.time()

loo_objects <- list()
lpd_train <- matrix(NA, nrow = N_train, ncol = n_models)

for(i in seq_along(successful_models)) {
  m <- successful_models[i]
  loo_obj <- loo(models_results[[m]]$loglik_train, cores = 4)
  loo_objects[[i]] <- loo_obj
  lpd_train[, i] <- loo_obj$pointwise[, "elpd_loo"]
}

elpd_loo_scores <- colSums(lpd_train, na.rm = TRUE)
max_elpd <- max(elpd_loo_scores)
log_weights <- elpd_loo_scores - max_elpd
raw_weights <- exp(log_weights)
bma_weights <- raw_weights / sum(raw_weights)

bma_pred_train <- as.vector(model_preds_train %*% bma_weights)
bma_pred_test <- as.vector(model_preds_test %*% bma_weights)

n_iter <- nrow(model_full_train[[1]])
bma_samples_train <- array(0, dim = c(n_iter, length(Y_train)))
bma_samples_test <- array(0, dim = c(n_iter, length(Y_test)))

for(i in seq_along(successful_models)) {
  bma_samples_train <- bma_samples_train + bma_weights[i] * model_full_train[[i]]
  bma_samples_test <- bma_samples_test + bma_weights[i] * model_full_test[[i]]
}

bma_train_q025 <- apply(bma_samples_train, 2, quantile, probs = 0.025)
bma_train_q975 <- apply(bma_samples_train, 2, quantile, probs = 0.975)
bma_test_q025 <- apply(bma_samples_test, 2, quantile, probs = 0.025)
bma_test_q975 <- apply(bma_samples_test, 2, quantile, probs = 0.975)

time_bma <- as.numeric(difftime(Sys.time(), time_start_bma, units = "secs"))

# BS
time_start_bs <- Sys.time()

stacking_weights <- loo_model_weights(loo_objects, method = "stacking")

bs_pred_train <- as.vector(model_preds_train %*% as.vector(stacking_weights))
bs_pred_test <- as.vector(model_preds_test %*% as.vector(stacking_weights))

bs_samples_train <- array(0, dim = c(n_iter, N_train))
bs_samples_test <- array(0, dim = c(n_iter, N_test))

for(i in seq_along(successful_models)) {
  weight_i <- as.vector(stacking_weights)[i]
  bs_samples_train <- bs_samples_train + weight_i * model_full_train[[i]]
  bs_samples_test <- bs_samples_test + weight_i * model_full_test[[i]]
}

bs_train_q025 <- apply(bs_samples_train, 2, quantile, probs = 0.025)
bs_train_q975 <- apply(bs_samples_train, 2, quantile, probs = 0.975)
bs_test_q025 <- apply(bs_samples_test, 2, quantile, probs = 0.025)
bs_test_q975 <- apply(bs_samples_test, 2, quantile, probs = 0.975)

time_bs <- as.numeric(difftime(Sys.time(), time_start_bs, units = "secs"))

# BHS
time_start_bhs <- Sys.time()

X_train_scaled <- scale(X_train_std)
X_test_scaled <- scale(X_test_std, 
                       center = attr(X_train_scaled, "scaled:center"),
                       scale = attr(X_train_scaled, "scaled:scale"))

bhs_stan_code <- "
data {
  int<lower=1> N_train; int<lower=1> N_test; int<lower=2> K; int<lower=1> P;
  matrix[N_train, P] X_train; matrix[N_test, P] X_test;
  matrix[N_train, K] lpd_point;
}
parameters {
  matrix[K-1, P] beta_raw;
  real<lower=0> tau_global; vector<lower=0>[K-1] tau_model; vector<lower=0>[P] tau_pred;
}
transformed parameters {
  matrix[K-1, P] beta;
  for(k in 1:(K-1)) {
    for(p in 1:P) {
      beta[k, p] = beta_raw[k, p] * tau_global * tau_model[k] * tau_pred[p];
    }
  }
}
model {
  tau_global ~ normal(0, 0.05); tau_model ~ normal(0, 0.1); tau_pred ~ normal(0, 0.1);
  to_vector(beta_raw) ~ std_normal();
  matrix[N_train, K] f; matrix[N_train, K] log_weights;
  f[, 1:(K-1)] = X_train * beta'; f[, K] = rep_vector(0, N_train);
  log_weights = f + lpd_point;
  for(n in 1:N_train) {
    target += log_sum_exp(log_weights[n]);
  }
}
generated quantities {
  matrix[N_test, K] weights_test; matrix[N_train, K] weights_train;
  matrix[N_test, K] f_test; f_test[, 1:(K-1)] = X_test * beta'; f_test[, K] = rep_vector(0, N_test);
  for(n in 1:N_test) { weights_test[n] = to_row_vector(softmax(to_vector(f_test[n]))); }
  matrix[N_train, K] f_train; f_train[, 1:(K-1)] = X_train * beta'; f_train[, K] = rep_vector(0, N_train);
  for(n in 1:N_train) { weights_train[n] = to_row_vector(softmax(to_vector(f_train[n]))); }
}
"

bhs_stan_data <- list(
  N_train = N_train, N_test = N_test, K = n_models, P = ncol(X_train_scaled),
  X_train = X_train_scaled, X_test = X_test_scaled, lpd_point = lpd_train
)

bhs_model <- stan_model(model_code = bhs_stan_code)
bhs_fit <- sampling(
  object = bhs_model, data = bhs_stan_data,
  iter = 2000, chains = 4, warmup = 1000,
  control = list(adapt_delta = 0.99, max_treedepth = 15),
  refresh = 0
)

weights_test_samples <- rstan::extract(bhs_fit, pars = 'weights_test')$weights_test
weights_train_samples <- rstan::extract(bhs_fit, pars = 'weights_train')$weights_train

avg_weights_test <- apply(weights_test_samples, c(2,3), mean)
avg_weights_train <- apply(weights_train_samples, c(2,3), mean)

bhs_pred_train <- rowSums(model_preds_train * avg_weights_train)
bhs_pred_test <- rowSums(model_preds_test * avg_weights_test)

time_bhs <- as.numeric(difftime(Sys.time(), time_start_bhs, units = "secs"))

# VCBART
time_start_vcbart <- Sys.time()

# Prepare VCBART data
model_names <- paste0("M", successful_models)
train_df <- data.frame(y = Y_train, model_preds_train, X_train_std, 
                       school_id = as.factor(df_train$schoolid_map))
test_df <- data.frame(y = Y_test, model_preds_test, X_test_std, 
                      school_id = as.factor(df_test$schoolid_map))

colnames(train_df)[2:(1+n_models)] <- model_names
colnames(test_df)[2:(1+n_models)] <- model_names

# VCBART formula construction
original_vars <- paste0("x", 1:9)
contextual_vars <- c(original_vars, "school_id")
bart_context <- paste("bart(", paste(contextual_vars, collapse = " + "), ")")

interaction_terms <- character(n_models)
for(i in 1:n_models) {
  interaction_terms[i] <- paste(model_names[i], "*", bart_context)
}

formula_string <- paste("y ~", bart_context, "+", paste(interaction_terms, collapse = " + "))
formula_vcbart <- as.formula(formula_string)

y_range <- max(Y_train) - min(Y_train)
custom_sigest <- y_range^2 / 16

fit_vcbart <- flexBART(
  formula = formula_vcbart, 
  train_data = train_df, 
  test_data = test_df,
  nd = 1000, burn = 1000, M_vec = 75,
  nest_v = TRUE, nest_v_option = 3, nest_c = TRUE,
  initialize_sigma = TRUE, sigest = custom_sigest, 
  sigquant = 0.95, nu = 5, verbose = FALSE
)

vcbart_pred_train <- fit_vcbart$yhat.train.mean
vcbart_pred_test <- fit_vcbart$yhat.test.mean

n_samples <- nrow(fit_vcbart$yhat.train)
vcbart_train_q025 <- vcbart_train_q975 <- numeric(N_train)
vcbart_test_q025 <- vcbart_test_q975 <- numeric(N_test)

for(i in 1:N_train) {
  mu_samples <- fit_vcbart$yhat.train[, i]
  sigma_samples <- fit_vcbart$sigma
  ystar_samples <- mu_samples + sigma_samples * rnorm(n_samples, 0, 1)
  vcbart_train_q025[i] <- quantile(ystar_samples, 0.025)
  vcbart_train_q975[i] <- quantile(ystar_samples, 0.975)
}

for(i in 1:N_test) {
  mu_samples <- fit_vcbart$yhat.test[, i]
  sigma_samples <- fit_vcbart$sigma
  ystar_samples <- mu_samples + sigma_samples * rnorm(n_samples, 0, 1)
  vcbart_test_q025[i] <- quantile(ystar_samples, 0.025)
  vcbart_test_q975[i] <- quantile(ystar_samples, 0.975)
}

time_vcbart <- as.numeric(difftime(Sys.time(), time_start_vcbart, units = "secs"))

# soloBART
time_start_solobart <- Sys.time()

school_train_factor <- factor(df_train$schoolid_map)
school_test_factor <- factor(df_test$schoolid_map)
all_schools <- sort(unique(c(df_train$schoolid_map, df_test$schoolid_map)))
school_train_factor <- factor(school_train_factor, levels = all_schools)
school_test_factor <- factor(school_test_factor, levels = all_schools)

X_solobart_train <- data.frame(X_train_std, school = school_train_factor)
X_solobart_test <- data.frame(X_test_std, school = school_test_factor)

fit_solobart <- wbart(
  x.train = X_solobart_train, y.train = Y_train, x.test = X_solobart_test,
  sparse = TRUE, ntree = 50, ndpost = 1000, nskip = 1000,
  keepevery = 1, printevery = 500
)

solobart_pred_train <- fit_solobart$yhat.train.mean
solobart_pred_test <- fit_solobart$yhat.test.mean
solobart_train_q025 <- apply(fit_solobart$yhat.train, 2, quantile, probs = 0.025)
solobart_train_q975 <- apply(fit_solobart$yhat.train, 2, quantile, probs = 0.975)
solobart_test_q025 <- apply(fit_solobart$yhat.test, 2, quantile, probs = 0.025)
solobart_test_q975 <- apply(fit_solobart$yhat.test, 2, quantile, probs = 0.975)

time_solobart <- as.numeric(difftime(Sys.time(), time_start_solobart, units = "secs"))

# Model 7
time_start_m7 <- Sys.time()

# Model 7 Stan code (complete multilevel specification)
m7_stan_code <- "
data {
  int<lower=1> N_train; int<lower=1> J_train; vector[N_train] std_y_train;
  real avg_y; real sd_y; matrix[N_train, 9] X_train; int<lower=1, upper=J_train> school_train[N_train];
  int<lower=1> N_test; int<lower=1> J_test; matrix[N_test, 9] X_test; int<lower=1, upper=J_test> school_test[N_test];
}
parameters {
  real gamma0; vector[9] gammas; matrix[6, J_train] u_raw;
  vector<lower=0>[6] tau; cholesky_factor_corr[6] L_u; real<lower=0> sigma;
}
transformed parameters {
  matrix[J_train, 6] u = (diag_pre_multiply(tau, L_u) * u_raw)';
}
model {
  gamma0 ~ normal(0, 1); gammas ~ normal(0, 0.5);
  tau ~ exponential(3); L_u ~ lkj_corr_cholesky(1); sigma ~ exponential(2);
  to_vector(u_raw) ~ std_normal();
  vector[N_train] mu_train;
  for (n in 1:N_train) {
    int j = school_train[n];
    mu_train[n] = gamma0 + X_train[n] * gammas + u[j, 1] + 
                  u[j, 2] * X_train[n, 1] + u[j, 3] * X_train[n, 5] + 
                  u[j, 4] * X_train[n, 6] + u[j, 5] * X_train[n, 7] + 
                  u[j, 6] * X_train[n, 9];
  }
  std_y_train ~ normal(mu_train, sigma);
}
generated quantities {
  vector[N_train] y_pred_train; vector[N_test] y_pred_test;
  for (n in 1:N_train) {
    int j = school_train[n];
    real mu_train = gamma0 + X_train[n] * gammas + u[j, 1] + 
                    u[j, 2] * X_train[n, 1] + u[j, 3] * X_train[n, 5] + 
                    u[j, 4] * X_train[n, 6] + u[j, 5] * X_train[n, 7] + 
                    u[j, 6] * X_train[n, 9];
    y_pred_train[n] = avg_y + sd_y * normal_rng(mu_train, sigma);
  }
  matrix[J_test, 6] u_test;
  for (j in 1:J_test) {
    vector[6] u_test_raw = to_vector(normal_rng(rep_vector(0, 6), 1));
    u_test[j] = (diag_pre_multiply(tau, L_u) * u_test_raw)';
  }
  for (n in 1:N_test) {
    int j = school_test[n];
    real mu_test = gamma0 + X_test[n] * gammas + u_test[j, 1] + 
                   u_test[j, 2] * X_test[n, 1] + u_test[j, 3] * X_test[n, 5] + 
                   u_test[j, 4] * X_test[n, 6] + u_test[j, 5] * X_test[n, 7] + 
                   u_test[j, 6] * X_test[n, 9];
    y_pred_test[n] = avg_y + sd_y * normal_rng(mu_test, sigma);
  }
}
"

model7 <- stan_model(model_code = m7_stan_code)
fit_m7 <- sampling(
  object = model7, data = stan_data,
  iter = 3000, chains = 4, warmup = 2000,
  control = list(adapt_delta = 0.90, max_treedepth = 12),
  seed = seed, refresh = 0
)

posterior_m7 <- rstan::extract(fit_m7)
m7_pred_train <- colMeans(posterior_m7$y_pred_train)
m7_pred_test <- colMeans(posterior_m7$y_pred_test)
m7_train_q025 <- apply(posterior_m7$y_pred_train, 2, quantile, probs = 0.025)
m7_train_q975 <- apply(posterior_m7$y_pred_train, 2, quantile, probs = 0.975)
m7_test_q025 <- apply(posterior_m7$y_pred_test, 2, quantile, probs = 0.025)
m7_test_q975 <- apply(posterior_m7$y_pred_test, 2, quantile, probs = 0.975)

time_m7 <- as.numeric(difftime(Sys.time(), time_start_m7, units = "secs"))

# Evaluation function
calc_metrics <- function(pred_train, pred_test, q025_train, q975_train, q025_test, q975_test, time_run) {
  train_var <- var(Y_train)
  
  in_rmse <- sqrt(mean((Y_train - pred_train)^2))
  out_rmse <- sqrt(mean((Y_test - pred_test)^2))
  in_smse <- mean((Y_train - pred_train)^2) / train_var
  out_smse <- mean((Y_test - pred_test)^2) / train_var
  in_coverage <- mean((Y_train >= q025_train) & (Y_train <= q975_train))
  out_coverage <- mean((Y_test >= q025_test) & (Y_test <= q975_test))
  
  data.frame(
    in_rmse = in_rmse, out_rmse = out_rmse,
    in_smse = in_smse, out_smse = out_smse,
    in_coverage = in_coverage, out_coverage = out_coverage,
    time_run = time_run,
    total_time = time_run + total_model_time
  )
}

# Compile all
results_summary <- rbind(
  BMA = calc_metrics(bma_pred_train, bma_pred_test, bma_train_q025, bma_train_q975, bma_test_q025, bma_test_q975, time_bma),
  BS = calc_metrics(bs_pred_train, bs_pred_test, bs_train_q025, bs_train_q975, bs_test_q025, bs_test_q975, time_bs),
  BHS = calc_metrics(bhs_pred_train, bhs_pred_test, rep(NA, N_train), rep(NA, N_train), rep(NA, N_test), rep(NA, N_test), time_bhs),
  VCBART = calc_metrics(vcbart_pred_train, vcbart_pred_test, vcbart_train_q025, vcbart_train_q975, vcbart_test_q025, vcbart_test_q975, time_vcbart),
  soloBART = calc_metrics(solobart_pred_train, solobart_pred_test, solobart_train_q025, solobart_train_q975, solobart_test_q025, solobart_test_q975, time_solobart),
  M7 = calc_metrics(m7_pred_train, m7_pred_test, m7_train_q025, m7_train_q975, m7_test_q025, m7_test_q975, time_m7)
)

# Save
output_file <- paste0("study2_simulation_results_icc", icc, "_sigma", sigma, "_f", f, ".RData")
save(results_summary, bma_weights, stacking_weights, 
     file = output_file)

file.copy(from = output_file, to = paste0("/staging/mhuang233/", output_file))
file.remove(output_file)

