
# ::: STUDY 2 -- EMPIRICAL ::: #

{
  rm(list = ls())
  library(loo)
  library(rstan)
  library(flexBART)
  library(BART)
  library(tidyverse)
  set.seed(seed)
  options(mc.cores = parallel::detectCores())
  rstan_options(auto_write = TRUE)
}

# Load data
load("inputs.RData")
nrow(dt)
uni_sch <- dt$schoolid %>% unique() 
nj <- length(uni_sch)

# Rename variables for simplicity
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
    schoolid_map = match(schoolid, uni_sch) 
  )

# CV 80/20 split
train_index <- list()
test_index <- list()

for (f in 1:10) {
  set.seed(seed + f)
  
  train_students <- c()
  test_students <- c()
  
  for (school in 1:nj) {
    school_students <- dt %>% 
      filter(schoolid_map == school) %>% 
      pull(childid)
    
    if (length(school_students) == 0) next
    
    school_students_shuffled <- sample(school_students)
    n_students <- length(school_students_shuffled)
    
    if (n_students == 1) {
      train_students <- c(train_students, school_students_shuffled[1])
    } else if (n_students == 2) {
      train_students <- c(train_students, school_students_shuffled[1])
      test_students <- c(test_students, school_students_shuffled[2])
    } else {
      n_train <- max(1, round(n_students * 0.8))
      n_test <- n_students - n_train
      
      if (n_test == 0) {
        n_train <- n_students - 1
        n_test <- 1
      }
      
      train_students <- c(train_students, school_students_shuffled[1:n_train])
      test_students <- c(test_students, school_students_shuffled[(n_train + 1):n_students])
    }
  }
  
  train_index[[f]] <- which(dt$childid %in% train_students)
  test_index[[f]] <- which(dt$childid %in% test_students)
}

# Model fit
stan_models <- list(
  # Model 1: Random intercept only
  m1 = "
  data {
    int<lower=1> N_train; 
    int<lower=1> J_train; 
    vector[N_train] std_y_train;
    real avg_y; 
    real sd_y; 
    matrix[N_train, 9] X_train; 
    int<lower=1, upper=J_train> school_train[N_train];
    int<lower=1> N_test; int<lower=1> J_test; 
    matrix[N_test, 9] X_test; 
    int<lower=1, upper=J_test> school_test[N_test];
  }
  parameters {
    real gamma0; 
    vector[9] gammas; 
    vector[J_train] u0_raw; 
    real<lower=0> tau0; 
    real<lower=0> sigma;
  }
  transformed parameters {
    vector[J_train] u0 = tau0 * u0_raw;
  }
  model {
    gamma0 ~ normal(0, 1); 
    gammas ~ normal(0, 0.5); 
    tau0 ~ exponential(3); 
    sigma ~ exponential(2);
    u0_raw ~ std_normal();
    vector[N_train] mu_train;
    for (n in 1:N_train) {
      int j = school_train[n];
      mu_train[n] = gamma0 + X_train[n] * gammas + u0[j];
    }
    std_y_train ~ normal(mu_train, sigma);
  }
  generated quantities {
    vector[N_train] y_pred_train; 
    vector[N_train] log_lik_train;
    vector[N_test] y_pred_test; 
    vector[N_test] log_lik_test;
    for (n in 1:N_train) {
      int j = school_train[n];
      real mu_train = gamma0 + X_train[n] * gammas + u0[j];
      y_pred_train[n] = avg_y + sd_y * normal_rng(mu_train, sigma);
      log_lik_train[n] = normal_lpdf(std_y_train[n] | mu_train, sigma);
    }
    vector[J_test] u0_test = tau0 * to_vector(normal_rng(rep_vector(0, J_test), 1));
    for (n in 1:N_test) {
      int j = school_test[n];
      real mu_test = gamma0 + X_test[n] * gammas + u0_test[j];
      y_pred_test[n] = avg_y + sd_y * normal_rng(mu_test, sigma);
      log_lik_test[n] = normal_lpdf((y_pred_test[n] - avg_y) / sd_y | mu_test, sigma);
    }
  }
  ",
  
  # Model 2: Random intercept + ESCS slope (x1)
  m2 = "
  data {
    int<lower=1> N_train; 
    int<lower=1> J_train; 
    vector[N_train] std_y_train;
    real avg_y; real sd_y; 
    matrix[N_train, 9] X_train; 
    int<lower=1, upper=J_train> school_train[N_train];
    int<lower=1> N_test; int<lower=1> J_test; 
    matrix[N_test, 9] X_test; 
    int<lower=1, upper=J_test> school_test[N_test];
  }
  parameters {
    real gamma0; 
    vector[9] gammas; 
    matrix[2, J_train] u_raw;
    vector<lower=0>[2] tau; 
    cholesky_factor_corr[2] L_u; 
    real<lower=0> sigma;
  }
  transformed parameters {
    matrix[J_train, 2] u = (diag_pre_multiply(tau, L_u) * u_raw)';
  }
  model {
    gamma0 ~ normal(0, 1); 
    gammas ~ normal(0, 0.5); 
    tau ~ exponential(3);
    L_u ~ lkj_corr_cholesky(1); 
    sigma ~ exponential(2);
    to_vector(u_raw) ~ std_normal();
    vector[N_train] mu_train;
    for (n in 1:N_train) {
      int j = school_train[n];
      mu_train[n] = gamma0 + X_train[n] * gammas + u[j, 1] + u[j, 2] * X_train[n, 1];
    }
    std_y_train ~ normal(mu_train, sigma);
  }
  generated quantities {
    vector[N_train] y_pred_train; vector[N_train] log_lik_train;
    vector[N_test] y_pred_test; vector[N_test] log_lik_test;
    for (n in 1:N_train) {
      int j = school_train[n];
      real mu_train = gamma0 + X_train[n] * gammas + u[j, 1] + u[j, 2] * X_train[n, 1];
      y_pred_train[n] = avg_y + sd_y * normal_rng(mu_train, sigma);
      log_lik_train[n] = normal_lpdf(std_y_train[n] | mu_train, sigma);
    }
    matrix[J_test, 2] u_test;
    for (j in 1:J_test) {
      vector[2] u_test_raw = to_vector(normal_rng(rep_vector(0, 2), 1));
      u_test[j] = (diag_pre_multiply(tau, L_u) * u_test_raw)';
    }
    for (n in 1:N_test) {
      int j = school_test[n];
      real mu_test = gamma0 + X_test[n] * gammas + u_test[j, 1] + u_test[j, 2] * X_test[n, 1];
      y_pred_test[n] = avg_y + sd_y * normal_rng(mu_test, sigma);
      log_lik_test[n] = normal_lpdf((y_pred_test[n] - avg_y) / sd_y | mu_test, sigma);
    }
  }
  ",
  
  # Model 3: Random intercept + TEACHINT slope (x5)
  m3 = "
  data {
    int<lower=1> N_train; 
    int<lower=1> J_train; 
    vector[N_train] std_y_train;
    real avg_y; 
    real sd_y; 
    matrix[N_train, 9] X_train; 
    int<lower=1, upper=J_train> school_train[N_train];
    int<lower=1> N_test; 
    int<lower=1> J_test; 
    matrix[N_test, 9] X_test; 
    int<lower=1, upper=J_test> school_test[N_test];
  }
  parameters {
    real gamma0; 
    vector[9] gammas; 
    matrix[2, J_train] u_raw;
    vector<lower=0>[2] tau; 
    cholesky_factor_corr[2] L_u; 
    real<lower=0> sigma;
  }
  transformed parameters {
    matrix[J_train, 2] u = (diag_pre_multiply(tau, L_u) * u_raw)';
  }
  model {
    gamma0 ~ normal(0, 1); 
    gammas ~ normal(0, 0.5); 
    tau ~ exponential(3);
    L_u ~ lkj_corr_cholesky(1); 
    sigma ~ exponential(2);
    to_vector(u_raw) ~ std_normal();
    vector[N_train] mu_train;
    for (n in 1:N_train) {
      int j = school_train[n];
      mu_train[n] = gamma0 + X_train[n] * gammas + u[j, 1] + u[j, 2] * X_train[n, 5];
    }
    std_y_train ~ normal(mu_train, sigma);
  }
  generated quantities {
    vector[N_train] y_pred_train; vector[N_train] log_lik_train;
    vector[N_test] y_pred_test; vector[N_test] log_lik_test;
    for (n in 1:N_train) {
      int j = school_train[n];
      real mu_train = gamma0 + X_train[n] * gammas + u[j, 1] + u[j, 2] * X_train[n, 5];
      y_pred_train[n] = avg_y + sd_y * normal_rng(mu_train, sigma);
      log_lik_train[n] = normal_lpdf(std_y_train[n] | mu_train, sigma);
    }
    matrix[J_test, 2] u_test;
    for (j in 1:J_test) {
      vector[2] u_test_raw = to_vector(normal_rng(rep_vector(0, 2), 1));
      u_test[j] = (diag_pre_multiply(tau, L_u) * u_test_raw)';
    }
    for (n in 1:N_test) {
      int j = school_test[n];
      real mu_test = gamma0 + X_test[n] * gammas + u_test[j, 1] + u_test[j, 2] * X_test[n, 5];
      y_pred_test[n] = avg_y + sd_y * normal_rng(mu_test, sigma);
      log_lik_test[n] = normal_lpdf((y_pred_test[n] - avg_y) / sd_y | mu_test, sigma);
    }
  }
  ",
  
  # Model 4: Random intercept + ADAPTIVITY slope (x6)
  m4 = "
  data {
    int<lower=1> N_train; 
    int<lower=1> J_train; 
    vector[N_train] std_y_train;
    real avg_y; 
    real sd_y; 
    matrix[N_train, 9] X_train; 
    int<lower=1, upper=J_train> school_train[N_train];
    int<lower=1> N_test; 
    int<lower=1> J_test; 
    matrix[N_test, 9] X_test; 
    int<lower=1, upper=J_test> school_test[N_test];
  }
  parameters {
    real gamma0; vector[9] gammas; matrix[2, J_train] u_raw;
    vector<lower=0>[2] tau; cholesky_factor_corr[2] L_u; real<lower=0> sigma;
  }
  transformed parameters {
    matrix[J_train, 2] u = (diag_pre_multiply(tau, L_u) * u_raw)';
  }
  model {
    gamma0 ~ normal(0, 1); gammas ~ normal(0, 0.5); tau ~ exponential(3);
    L_u ~ lkj_corr_cholesky(1); sigma ~ exponential(2);
    to_vector(u_raw) ~ std_normal();
    vector[N_train] mu_train;
    for (n in 1:N_train) {
      int j = school_train[n];
      mu_train[n] = gamma0 + X_train[n] * gammas + u[j, 1] + u[j, 2] * X_train[n, 6];
    }
    std_y_train ~ normal(mu_train, sigma);
  }
  generated quantities {
    vector[N_train] y_pred_train; vector[N_train] log_lik_train;
    vector[N_test] y_pred_test; vector[N_test] log_lik_test;
    for (n in 1:N_train) {
      int j = school_train[n];
      real mu_train = gamma0 + X_train[n] * gammas + u[j, 1] + u[j, 2] * X_train[n, 6];
      y_pred_train[n] = avg_y + sd_y * normal_rng(mu_train, sigma);
      log_lik_train[n] = normal_lpdf(std_y_train[n] | mu_train, sigma);
    }
    matrix[J_test, 2] u_test;
    for (j in 1:J_test) {
      vector[2] u_test_raw = to_vector(normal_rng(rep_vector(0, 2), 1));
      u_test[j] = (diag_pre_multiply(tau, L_u) * u_test_raw)';
    }
    for (n in 1:N_test) {
      int j = school_test[n];
      real mu_test = gamma0 + X_test[n] * gammas + u_test[j, 1] + u_test[j, 2] * X_test[n, 6];
      y_pred_test[n] = avg_y + sd_y * normal_rng(mu_test, sigma);
      log_lik_test[n] = normal_lpdf((y_pred_test[n] - avg_y) / sd_y | mu_test, sigma);
    }
  }
  ",
  
  # Model 5: Random intercept + BEINGBULLIED slope (x7)
  m5 = "
  data {
    int<lower=1> N_train; 
    int<lower=1> J_train; 
    vector[N_train] std_y_train;
    real avg_y; 
    real sd_y; 
    matrix[N_train, 9] X_train; 
    int<lower=1, upper=J_train> school_train[N_train];
    int<lower=1> N_test; 
    int<lower=1> J_test; 
    matrix[N_test, 9] X_test; int<lower=1, upper=J_test> school_test[N_test];
  }
  parameters {
    real gamma0; 
    vector[9] gammas; 
    matrix[2, J_train] u_raw;
    vector<lower=0>[2] tau; 
    cholesky_factor_corr[2] L_u; 
    real<lower=0> sigma;
  }
  transformed parameters {
    matrix[J_train, 2] u = (diag_pre_multiply(tau, L_u) * u_raw)';
  }
  model {
    gamma0 ~ normal(0, 1); 
    gammas ~ normal(0, 0.5); 
    tau ~ exponential(3);
    L_u ~ lkj_corr_cholesky(1); 
    sigma ~ exponential(2);
    to_vector(u_raw) ~ std_normal();
    vector[N_train] mu_train;
    for (n in 1:N_train) {
      int j = school_train[n];
      mu_train[n] = gamma0 + X_train[n] * gammas + u[j, 1] + u[j, 2] * X_train[n, 7];
    }
    std_y_train ~ normal(mu_train, sigma);
  }
  generated quantities {
    vector[N_train] y_pred_train; 
    vector[N_train] log_lik_train;
    vector[N_test] y_pred_test; 
    vector[N_test] log_lik_test;
    for (n in 1:N_train) {
      int j = school_train[n];
      real mu_train = gamma0 + X_train[n] * gammas + u[j, 1] + u[j, 2] * X_train[n, 7];
      y_pred_train[n] = avg_y + sd_y * normal_rng(mu_train, sigma);
      log_lik_train[n] = normal_lpdf(std_y_train[n] | mu_train, sigma);
    }
    matrix[J_test, 2] u_test;
    for (j in 1:J_test) {
      vector[2] u_test_raw = to_vector(normal_rng(rep_vector(0, 2), 1));
      u_test[j] = (diag_pre_multiply(tau, L_u) * u_test_raw)';
    }
    for (n in 1:N_test) {
      int j = school_test[n];
      real mu_test = gamma0 + X_test[n] * gammas + u_test[j, 1] + u_test[j, 2] * X_test[n, 7];
      y_pred_test[n] = avg_y + sd_y * normal_rng(mu_test, sigma);
      log_lik_test[n] = normal_lpdf((y_pred_test[n] - avg_y) / sd_y | mu_test, sigma);
    }
  }
  ",
  
  # Model 6: Random intercept + IMMIG slope (x9)
  m6 = "
  data {
    int<lower=1> N_train; 
    int<lower=1> J_train; 
    vector[N_train] std_y_train;
    real avg_y; 
    real sd_y; 
    matrix[N_train, 9] X_train; 
    int<lower=1, upper=J_train> school_train[N_train];
    int<lower=1> N_test; 
    int<lower=1> J_test; 
    matrix[N_test, 9] X_test; 
    int<lower=1, upper=J_test> school_test[N_test];
  }
  parameters {
    real gamma0; 
    vector[9] gammas; 
    matrix[2, J_train] u_raw;
    vector<lower=0>[2] tau; 
    cholesky_factor_corr[2] L_u; 
    real<lower=0> sigma;
  }
  transformed parameters {
    matrix[J_train, 2] u = (diag_pre_multiply(tau, L_u) * u_raw)';
  }
  model {
    gamma0 ~ normal(0, 1); 
    gammas ~ normal(0, 0.5); 
    tau ~ exponential(3);
    L_u ~ lkj_corr_cholesky(1); 
    sigma ~ exponential(2);
    to_vector(u_raw) ~ std_normal();
    vector[N_train] mu_train;
    for (n in 1:N_train) {
      int j = school_train[n];
      mu_train[n] = gamma0 + X_train[n] * gammas + u[j, 1] + u[j, 2] * X_train[n, 9];
    }
    std_y_train ~ normal(mu_train, sigma);
  }
  generated quantities {
    vector[N_train] y_pred_train; 
    vector[N_train] log_lik_train;
    vector[N_test] y_pred_test;
    vector[N_test] log_lik_test;
    for (n in 1:N_train) {
      int j = school_train[n];
      real mu_train = gamma0 + X_train[n] * gammas + u[j, 1] + u[j, 2] * X_train[n, 9];
      y_pred_train[n] = avg_y + sd_y * normal_rng(mu_train, sigma);
      log_lik_train[n] = normal_lpdf(std_y_train[n] | mu_train, sigma);
    }
    matrix[J_test, 2] u_test;
    for (j in 1:J_test) {
      vector[2] u_test_raw = to_vector(normal_rng(rep_vector(0, 2), 1));
      u_test[j] = (diag_pre_multiply(tau, L_u) * u_test_raw)';
    }
    for (n in 1:N_test) {
      int j = school_test[n];
      real mu_test = gamma0 + X_test[n] * gammas + u_test[j, 1] + u_test[j, 2] * X_test[n, 9];
      y_pred_test[n] = avg_y + sd_y * normal_rng(mu_test, sigma);
      log_lik_test[n] = normal_lpdf((y_pred_test[n] - avg_y) / sd_y | mu_test, sigma);
    }
  }
  "
)

# Function for fitting m1 - m6
fit_model <- function(fold, model_num, model_code) {
  train_idx <- train_index[[fold]]
  test_idx <- test_index[[fold]]
  df_train <- dt[train_idx, ]
  df_test <- dt[test_idx, ]
  
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
  
  avg_y <- mean(df_train$pv1math)
  sd_y <- sd(df_train$pv1math)
  df_train$std_y <- (df_train$pv1math - avg_y) / sd_y
  df_test$std_y <- (df_test$pv1math - avg_y) / sd_y
  
  predictor_vars <- paste0("x", 1:9)
  X_train_raw <- as.matrix(df_train[, predictor_vars])
  X_test_raw <- as.matrix(df_test[, predictor_vars])
  
  X_means <- colMeans(X_train_raw)
  X_sds <- apply(X_train_raw, 2, sd)
  X_sds[X_sds < 1e-10] <- 1
  
  X_train <- scale(X_train_raw, center = X_means, scale = X_sds)
  X_test <- scale(X_test_raw, center = X_means, scale = X_sds)
  
  stan_data <- list(
    N_train = nrow(df_train), J_train = J_train,
    std_y_train = df_train$std_y, avg_y = avg_y, sd_y = sd_y,
    X_train = X_train, school_train = df_train$school_stan,
    N_test = nrow(df_test), J_test = J_test,
    X_test = X_test, school_test = df_test$school_stan
  )
  
  if (model_num == 1) {
    init_fun <- function(chain_id) {
      list(gamma0 = rnorm(1, 0, 0.1), gammas = rnorm(9, 0, 0.1),
           u0_raw = rnorm(J_train, 0, 0.1), tau0 = abs(rnorm(1, 0.3, 0.05)),
           sigma = abs(rnorm(1, 1, 0.1)))
    }
  } else {
    init_fun <- function(chain_id) {
      list(gamma0 = rnorm(1, 0, 0.1), gammas = rnorm(9, 0, 0.1),
           u_raw = matrix(rnorm(2 * J_train, 0, 0.1), nrow = 2),
           tau = abs(rnorm(2, 0.3, 0.05)), L_u = diag(2),
           sigma = abs(rnorm(1, 1, 0.1)))
    }
  }
  
  # Fit model
  time_start <- Sys.time()
  model_stan <- stan_model(model_code = model_code)
  fit <- sampling(
    object = model_stan, data = stan_data,
    iter = 3000, chains = 4, warmup = 2000,
    control = list(adapt_delta = 0.90, max_treedepth = 12),
    init = init_fun, seed = seed, refresh = 0
  )
  
  time_elapsed <- as.numeric(difftime(Sys.time(), time_start, units = "secs"))
  
  # Extract
  posterior_list <- rstan::extract(fit)
  return(list(
    loglik_train = posterior_list$log_lik_train,
    loglik_test = posterior_list$log_lik_test,
    ypred_train = posterior_list$y_pred_train,
    ypred_test = posterior_list$y_pred_test,
    Y_train = df_train$pv1math,
    Y_test = df_test$pv1math,
    X_train = as.data.frame(X_train),
    X_test = as.data.frame(X_test),
    school_train_numeric = df_train$schoolid_map,
    school_test_numeric = df_test$schoolid_map,
    time_elapsed = time_elapsed
  ))
}

all_models <- list()
total_model_time <- 0

# Combine modle fits for later analysis
for (f in 1:10) {
  fold_results <- list()
  
  for (m in 1:6) {
    model_code <- stan_models[[paste0("m", m)]]
    result <- fit_model(f, m, model_code)
    fold_results[[m]] <- result
    total_model_time <- total_model_time + result$time_elapsed
  }
  
  all_models[[f]] <- fold_results
}

all_fold_results <- list()

for (f in 1:10) {
  
  fold_data <- all_models[[f]]
  
  # Extract combined data
  Y_train <- fold_data[[1]]$Y_train
  Y_test <- fold_data[[1]]$Y_test
  X_train <- fold_data[[1]]$X_train
  X_test <- fold_data[[1]]$X_test
  
  # prediction
  model_preds_train <- sapply(1:6, function(m) colMeans(fold_data[[m]]$ypred_train))
  model_preds_test <- sapply(1:6, function(m) colMeans(fold_data[[m]]$ypred_test))
  model_full_train <- lapply(1:6, function(m) fold_data[[m]]$ypred_train)
  model_full_test <- lapply(1:6, function(m) fold_data[[m]]$ypred_test)
  loglik_train_list <- lapply(1:6, function(m) fold_data[[m]]$loglik_train)
  
  # BMA
  time_start_bma <- Sys.time()
  
  loo_objects <- lapply(loglik_train_list, function(ll) loo(ll, cores = 4))
  lpd_train <- sapply(loo_objects, function(loo_obj) loo_obj$pointwise[, "elpd_loo"])
  
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
  
  for(i in 1:6) {
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
  
  bs_samples_train <- array(0, dim = c(n_iter, length(Y_train)))
  bs_samples_test <- array(0, dim = c(n_iter, length(Y_test)))
  
  for(i in 1:6) {
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
  
  X_train_scaled <- scale(X_train)
  X_test_scaled <- scale(X_test, 
                         center = attr(X_train_scaled, "scaled:center"),
                         scale = attr(X_train_scaled, "scaled:scale"))
  
  bhs_stan_code <- "
  data {
    int<lower=1> N_train; 
    int<lower=1> N_test; 
    int<lower=2> K; 
    int<lower=1> P;
    matrix[N_train, P] X_train; 
    matrix[N_test, P] X_test; 
    matrix[N_train, K] lpd_point;
  }
  parameters {
    matrix[K-1, P] beta_raw; 
    real<lower=0> tau_global;
    vector<lower=0>[K-1] tau_model; 
    vector<lower=0>[P] tau_pred;
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
    tau_global ~ normal(0, 0.05); 
    tau_model ~ normal(0, 0.1); 
    tau_pred ~ normal(0, 0.1);
    to_vector(beta_raw) ~ std_normal();
    matrix[N_train, K] f; 
    matrix[N_train, K] log_weights;
    f[, 1:(K-1)] = X_train * beta'; 
    f[, K] = rep_vector(0, N_train);
    log_weights = f + lpd_point;
    for(n in 1:N_train) { target += log_sum_exp(log_weights[n]); }
  }
  generated quantities {
    matrix[N_test, K] weights_test; 
    matrix[N_train, K] weights_train;
    matrix[N_test, K] f_test; 
    f_test[, 1:(K-1)] = X_test * beta'; 
    f_test[, K] = rep_vector(0, N_test);
    for(n in 1:N_test) { weights_test[n] = to_row_vector(softmax(to_vector(f_test[n]))); }
    matrix[N_train, K] f_train; 
    f_train[, 1:(K-1)] = X_train * beta'; 
    f_train[, K] = rep_vector(0, N_train);
    for(n in 1:N_train) { weights_train[n] = to_row_vector(softmax(to_vector(f_train[n]))); }
  }
  "
  # bhs data list
  bhs_stan_data <- list(
    N_train = length(Y_train), N_test = length(Y_test), K = 6, P = ncol(X_train_scaled),
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
  
  model_names <- paste0("M", 1:6)
  train_df <- data.frame(y = Y_train, model_preds_train, X_train, 
                         school_id = as.factor(fold_data[[1]]$school_train_numeric))
  test_df <- data.frame(y = Y_test, model_preds_test, X_test, 
                        school_id = as.factor(fold_data[[1]]$school_test_numeric))
  
  colnames(train_df)[2:7] <- model_names
  colnames(test_df)[2:7] <- model_names
  
  all_school_ids <- unique(c(fold_data[[1]]$school_train_numeric, fold_data[[1]]$school_test_numeric))
  train_df$school_id <- factor(train_df$school_id, levels = all_school_ids)
  test_df$school_id <- factor(test_df$school_id, levels = all_school_ids)
  
  predictor_names <- paste0("x", 1:9)
  contextual_vars <- c(predictor_names, "school_id")
  bart_context <- paste("bart(", paste(contextual_vars, collapse = " + "), ")")
  
  interaction_terms <- character(6)
  
  for(i in 1:6) {
    interaction_terms[i] <- paste(model_names[i], "*", bart_context)
  }
  
  formula_string <- paste("y ~", bart_context, "+", paste(interaction_terms, collapse = " + "))
  vcbart_formula <- as.formula(formula_string)
  
  y_range <- max(Y_train) - min(Y_train)
  
  fit <- tryCatch({
    flexBART(formula = vcbart_formula, train_data = train_df, test_data = test_df,
             nd = 1000, burn = 1000, M_vec = 75,
             nest_v = TRUE, nest_v_option = 3, nest_c = TRUE,
             initialize_sigma = TRUE, sigest = custom_sigest, 
             sigquant = 0.95, nu = 5, verbose = FALSE)
  }, error = function(e) NULL)
  
  time_vcbart <- as.numeric(difftime(Sys.time(), time_start_vcbart, units = "secs"))
  
  # soloBART
  time_start_solobart <- Sys.time()
  
  school_train_factor <- factor(fold_data[[1]]$school_train_numeric, levels = all_school_ids)
  school_test_factor <- factor(fold_data[[1]]$school_test_numeric, levels = all_school_ids)
  
  X_solobart_train <- data.frame(X_train, school = school_train_factor)
  X_solobart_test <- data.frame(X_test, school = school_test_factor)
  
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
  
  m7_stan_code <- "
  data {
    int<lower=1> N_train; 
    int<lower=1> J_train; 
    vector[N_train] std_y_train;
    real avg_y; 
    real sd_y; 
    matrix[N_train, 9] X_train; 
    int<lower=1, upper=J_train> school_train[N_train];
    int<lower=1> N_test; 
    int<lower=1> J_test; 
    matrix[N_test, 9] X_test; 
    int<lower=1, upper=J_test> school_test[N_test];
  }
  parameters {
    real gamma0; 
    vector[9] gammas; 
    matrix[6, J_train] u_raw;
    vector<lower=0>[6] tau; 
    cholesky_factor_corr[6] L_u; 
    real<lower=0> sigma;
  }
  transformed parameters {
    matrix[J_train, 6] u = (diag_pre_multiply(tau, L_u) * u_raw)';
  }
  model {
    gamma0 ~ normal(0, 1); 
    gammas ~ normal(0, 0.5); 
    tau ~ exponential(3);
    L_u ~ lkj_corr_cholesky(1); 
    sigma ~ exponential(2);
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
  
  # School mappings
  first_model_data <- fit_model(f, 1, stan_models$m1)  # Get proper data structure
  
  model7 <- stan_model(model_code = m7_stan_code)
  fit_m7 <- sampling(
    object = model7, 
    data = list(
      N_train = length(first_model_data$Y_train),
      J_train = length(unique(first_model_data$school_train_numeric)),
      std_y_train = scale(first_model_data$Y_train)[,1],
      avg_y = mean(first_model_data$Y_train),
      sd_y = sd(first_model_data$Y_train),
      X_train = as.matrix(first_model_data$X_train),
      school_train = as.numeric(as.factor(first_model_data$school_train_numeric)),
      N_test = length(first_model_data$Y_test),
      J_test = length(unique(first_model_data$school_test_numeric)), 
      X_test = as.matrix(first_model_data$X_test),
      school_test = as.numeric(as.factor(first_model_data$school_test_numeric))
    ),
    iter = 3000, chains = 4, warmup = 2000,
    control = list(adapt_delta = 0.90, max_treedepth = 12),
    seed = 233, refresh = 0
  )
  
  posterior_m7 <- rstan::extract(fit_m7)
  m7_pred_train <- colMeans(posterior_m7$y_pred_train)
  m7_pred_test <- colMeans(posterior_m7$y_pred_test)
  m7_train_q025 <- apply(posterior_m7$y_pred_train, 2, quantile, probs = 0.025)
  m7_train_q975 <- apply(posterior_m7$y_pred_train, 2, quantile, probs = 0.975)
  m7_test_q025 <- apply(posterior_m7$y_pred_test, 2, quantile, probs = 0.025)
  m7_test_q975 <- apply(posterior_m7$y_pred_test, 2, quantile, probs = 0.975)
  
  time_m7 <- as.numeric(difftime(Sys.time(), time_start_m7, units = "secs"))
  
  
  # calculate evaluation metrics
  calc_metrics <- function(pred_train, pred_test, q025_train, q975_train, q025_test, q975_test, time_run) {
    train_var <- var(Y_train)
    
    in_rmse <- sqrt(mean((Y_train - pred_train)^2))
    out_rmse <- sqrt(mean((Y_test - pred_test)^2))
    in_smse <- mean((Y_train - pred_train)^2) / train_var
    out_smse <- mean((Y_test - pred_test)^2) / train_var
    in_coverage <- mean((Y_train >= q025_train) & (Y_train <= q975_train))
    out_coverage <- mean((Y_test >= q025_test) & (Y_test <= q975_test))
    
    data.frame(
      fold = f, in_rmse = in_rmse, out_rmse = out_rmse,
      in_smse = in_smse, out_smse = out_smse,
      in_coverage = in_coverage, out_coverage = out_coverage,
      time = time_run
    )
  }
  
  # Compile results for current fold
  fold_results <- rbind(
    BMA = calc_metrics(bma_pred_train, bma_pred_test, bma_train_q025, bma_train_q975, bma_test_q025, bma_test_q975, time_bma),
    BS = calc_metrics(bs_pred_train, bs_pred_test, bs_train_q025, bs_train_q975, bs_test_q025, bs_test_q975, time_bs),
    BHS = calc_metrics(bhs_pred_train, bhs_pred_test, rep(NA, length(Y_train)), rep(NA, length(Y_train)), rep(NA, length(Y_test)), rep(NA, length(Y_test)), time_bhs),
    VCBART = calc_metrics(vcbart_pred_train, vcbart_pred_test, vcbart_train_q025, vcbart_train_q975, vcbart_test_q025, vcbart_test_q975, time_vcbart),
    soloBART = calc_metrics(solobart_pred_train, solobart_pred_test, solobart_train_q025, solobart_train_q975, solobart_test_q025, solobart_test_q975, time_solobart),
    M7 = calc_metrics(m7_pred_train, m7_pred_test, m7_train_q025, m7_train_q975, m7_test_q025, m7_test_q975, time_m7)
  )
  
  all_fold_results[[f]] <- fold_results
}

# Combine result
combined_results <- do.call(rbind, all_fold_results)
combined_results$method <- rep(c("BMA", "BS", "BHS", "VCBART", "soloBART", "M7"), 10)

summary_stats <- combined_results %>%
  group_by(method) %>%
  summarise(
    n_folds = n(),
    mean_in_rmse = round(mean(in_rmse, na.rm = TRUE), 4),
    sd_in_rmse = round(sd(in_rmse, na.rm = TRUE), 4),
    mean_out_rmse = round(mean(out_rmse, na.rm = TRUE), 4),
    sd_out_rmse = round(sd(out_rmse, na.rm = TRUE), 4),
    mean_in_coverage = round(mean(in_coverage, na.rm = TRUE), 4),
    mean_out_coverage = round(mean(out_coverage, na.rm = TRUE), 4),
    mean_time_minutes = round(mean(time, na.rm = TRUE) / 60, 2),
    .groups = 'drop'
  ) %>%
  arrange(mean_out_rmse)

# Save
save(combined_results, summary_stats, bma_weights, stacking_weights,
     file = "study2_empirical_results_complete.RData")
