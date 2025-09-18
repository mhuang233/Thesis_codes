# ::: STUDY 1 -- EMPIRICAL ::: #

{
  library(loo)
  library(rstanarm)
  library(rstan)
  library(LaplacesDemon)
  library(kableExtra)
  library(bayesplot)
  library(tidyverse)
  options(mc.cores = parallel::detectCores())
  set.seed(seed)
}

# Load PISA 2018
df0 <- read.csv("pisa2018.BayesBook.csv")

df <- df0 %>%
  dplyr::select(SchoolID, CNTSTUID, Female, ESCS, METASUM, PERFEED, HOMEPOS, 
                ADAPTIVITY, TEACHINT, ICTRES, ATTLNACT, COMPETE, JOYREAD,
                WORKMAST, GFOFAIL, SWBP, MASTGOAL, BELONG, SCREADCOMP, 
                PISADIFF, Public, PV1READ, SCREADDIFF)

# Filter schools >10 students
sch <- table(df$SchoolID)
dt0 <- subset(df, SchoolID %in% names(sch[sch > 10]))

# Check school distribution
dt0 %>%
  group_by(SchoolID) %>%
  summarise(n=n())

# ::: REDUCED SAMPLE ANALYSIS (N = 500) ::: #
# Sample 10 students per school, then select 50 schools
dt <- dt0 %>% group_by(SchoolID) %>% slice_sample(n = 10)
SchID <- dt$SchoolID 
unique_sch <- unique(dt$SchoolID)

sch_sample <- sample(unique_sch, 50)
sch_index <- which(SchID %in% sch_sample)

df <- dt[sch_index, ]

# Fit candidate models
bsm <- list()
loo_bs <- list()

# Model 1: Demographic
bsm[[1]] <- stan_lmer(
  PV1READ ~ Female + ESCS + HOMEPOS + ICTRES + (1 + ICTRES|SchoolID), data = df, 
  prior_intercept = student_t(3, 470, 100),
  iter = 10000, chains = 4, adapt_delta=.999, thin=10)

# Model 2: Reading behavior
bsm[[2]] <- stan_lmer(
  PV1READ ~ JOYREAD + PISADIFF + SCREADCOMP + SCREADDIFF + (1|SchoolID),
  data = df, prior_intercept = student_t(3, 470, 100), iter = 10000, chains = 4,
  adapt_delta=.999, thin=10)

# Model 3: Academic mindset
bsm[[3]] <- stan_lmer(
  PV1READ ~ METASUM + GFOFAIL + MASTGOAL + SWBP + WORKMAST + ADAPTIVITY + COMPETE + (1|SchoolID),
  data = df, prior_intercept = student_t(3, 470, 100), iter = 10000, chains = 4,
  adapt_delta=.999, thin=10)

# Model 4: School climate
bsm[[4]] <- stan_lmer(
  PV1READ ~ PERFEED + TEACHINT + BELONG + (1 + TEACHINT|SchoolID),
  data = df, prior_intercept = student_t(3, 470, 100), iter = 10000, chains = 4,
  adapt_delta=.999, thin=10)

# Compute LOO
loo_bs[[1]] <- loo(log_lik(bsm[[1]]))
loo_bs[[2]] <- loo(log_lik(bsm[[2]]))
loo_bs[[3]] <- loo(log_lik(bsm[[3]]))
loo_bs[[4]] <- loo(log_lik(bsm[[4]]))

# Ensemble weights for stakcing-based methods
system.time(w_bs <- loo_model_weights(loo_bs, method = "stacking"))
system.time(w_pbma <- loo_model_weights(loo_bs, method = "pseudobma", BB=FALSE))
system.time(w_pbmabb <- loo_model_weights(loo_bs, method = "pseudobma"))

# Extract pointwise LOO densities
lpd_point <- as.matrix(cbind(loo_bs[[1]]$pointwise[, "elpd_loo"],
                             loo_bs[[2]]$pointwise[, "elpd_loo"],
                             loo_bs[[3]]$pointwise[, "elpd_loo"],
                             loo_bs[[4]]$pointwise[, "elpd_loo"]))

n_draws <- nrow(as.matrix(bsm[[1]]))

# BS
ypred_bs <- matrix(NA, nrow = n_draws, ncol = nobs(bsm[[1]]))
for (d in 1:n_draws) {
  k <- sample(1:length(w_bs), size = 1, prob = w_bs)
  ypred_bs[d, ] <- posterior_predict(bsm[[k]], draws = 1)
}

y_bs <- colMeans(ypred_bs)
d1 <- density(y_bs, kernel = c("gaussian"))$y
d0 <- density(df$PV1READ, kernel = c("gaussian"))$y
kld1 <- KLD(d1, d0)$sum.KLD.py.px

# PBMA
ypred_bma <- matrix(NA, nrow = n_draws, ncol = nobs(bsm[[1]]))
for (d in 1:n_draws) {
  k <- sample(1:length(w_pbma), size = 1, prob = w_pbma)
  ypred_bma[d, ] <- posterior_predict(bsm[[k]], draws = 1)
}

y_bma <- colMeans(ypred_bma)
d2 <- density(y_bma, kernel = c("gaussian"))$y
kld2 <- KLD(d2, d0)$sum.KLD.py.px

# PBMA+
ypred_bmabb <- matrix(NA, nrow = n_draws, ncol = nobs(bsm[[1]]))
for (d in 1:n_draws) {
  k <- sample(1:length(w_pbmabb), size = 1, prob = w_pbmabb)
  ypred_bmabb[d, ] <- posterior_predict(bsm[[k]], draws = 1)
}

y_bmabb <- colMeans(ypred_bmabb)
d3 <- density(y_bmabb, kernel = c("gaussian"))$y
kld3 <- KLD(d3, d0)$sum.KLD.py.px

# BHS
d_discrete <- 1
X <- df[, c("ESCS","HOMEPOS","ICTRES",
            "JOYREAD","PISADIFF","SCREADCOMP","SCREADDIFF",
            "METASUM","GFOFAIL","MASTGOAL","SWBP","WORKMAST","ADAPTIVITY","COMPETE",
            "PERFEED","TEACHINT","BELONG")] 

stan_bhs <- list(X = X, N = nrow(X), d = ncol(X), d_discrete = d_discrete,
                 lpd_point = lpd_point, K = ncol(lpd_point), tau_mu = 1,
                 tau_sigma = 1, tau_discrete = .5, tau_con = 1)

# Fit stan
fit_bhs <- stan("bhs_stan.stan", data = stan_bhs, chains = 4, iter = 10000)

# Extract weights
wts_bhs <- rstan::extract(fit_bhs, pars = 'w')$w
w_bhs_m <- as.matrix(apply(wts_bhs, 3, mean))

ypred_bhs_r <- matrix(NA, nrow = n_draws, ncol = nobs(bsm[[1]]))
for (d in 1:n_draws) {
  k <- sample(1:4, size = 1, prob = w_bhs_m)
  ypred_bhs_r[d, ] <- posterior_predict(bsm[[k]], draws = 1)
}

y_bhs_r <- colMeans(ypred_bhs_r)
d4 <- density(y_bhs_r, kernel = c("gaussian"))$y
kld4 <- KLD(d4, d0)$sum.KLD.py.px

# Combine results
wr <- data.frame(as.matrix(w_bs), as.matrix(w_pbma), as.matrix(w_pbmabb), w_bhs_m)
colnames(wr) <- c("bs","pbma", "pbmabb", "bhs")

klds <- rbind(kld1, kld2, kld3, kld4)

save(lpd_point, fit_bhs, wr, klds, bsm, loo_bs, 
     file = "real_input.RData")


# ::: FULL SAMPLE ::: #

df <- df0 %>%
  dplyr::select(SchoolID, CNTSTUID, Female, ESCS, METASUM, PERFEED, HOMEPOS, 
                ADAPTIVITY, TEACHINT, ICTRES, ATTLNACT, COMPETE, JOYREAD,
                WORKMAST, GFOFAIL, SWBP, MASTGOAL, BELONG, SCREADCOMP, 
                PISADIFF, Public, PV1READ, SCREADDIFF)

# Fit
bsm <- list()
loo_bs <- list()

bsm[[1]] <- stan_lmer(
  PV1READ ~ Female + ESCS + HOMEPOS + ICTRES + (1 + ICTRES|SchoolID), data = dt0, 
  prior_intercept = student_t(3, 470, 100),
  iter = 10000, chains = 4, adapt_delta=.999, thin=10)

bsm[[2]] <- stan_lmer(
  PV1READ ~ JOYREAD + PISADIFF + SCREADCOMP + SCREADDIFF + (1|SchoolID),
  data = dt0, prior_intercept = student_t(3, 470, 100), iter = 10000, chains = 4,
  adapt_delta=.999, thin=10)

bsm[[3]] <- stan_lmer(
  PV1READ ~ METASUM + GFOFAIL + MASTGOAL + SWBP + WORKMAST + ADAPTIVITY + COMPETE + (1|SchoolID),
  data = dt0, prior_intercept = student_t(3, 470, 100), iter = 10000, chains = 4,
  adapt_delta=.999, thin=10)

bsm[[4]] <- stan_lmer(
  PV1READ ~ PERFEED + TEACHINT + BELONG + (1 + TEACHINT|SchoolID),
  data = dt0, prior_intercept = student_t(3, 470, 100), iter = 10000, chains = 4,
  adapt_delta=.999, thin=10)

# Compute LOO
loo_bs[[1]] <- loo(log_lik(bsm[[1]]))
loo_bs[[2]] <- loo(log_lik(bsm[[2]]))
loo_bs[[3]] <- loo(log_lik(bsm[[3]]))
loo_bs[[4]] <- loo(log_lik(bsm[[4]]))

w_bs <- loo_model_weights(loo_bs, method = "stacking")
w_pbma <- loo_model_weights(loo_bs, method = "pseudobma", BB=FALSE)
w_pbmabb <- loo_model_weights(loo_bs, method = "pseudobma")

# Extract pointwise densities
lpd_point <- as.matrix(cbind(loo_bs[[1]]$pointwise[, "elpd_loo"],
                             loo_bs[[2]]$pointwise[, "elpd_loo"],
                             loo_bs[[3]]$pointwise[, "elpd_loo"],
                             loo_bs[[4]]$pointwise[, "elpd_loo"]))

n_draws <- nrow(as.matrix(bsm[[1]]))

# BS
ypred_bs <- matrix(NA, nrow = n_draws, ncol = nobs(bsm[[1]]))
for (d in 1:n_draws) {
  k <- sample(1:length(w_bs), size = 1, prob = w_bs)
  ypred_bs[d, ] <- posterior_predict(bsm[[k]], draws = 1)
}

y_bs <- colMeans(ypred_bs)
d1 <- density(y_bs, kernel = c("gaussian"))$y
d0 <- density(dt0$PV1READ, kernel = c("gaussian"))$y
kld1 <- KLD(d1, d0)$sum.KLD.py.px

# PBMA
ypred_bma <- matrix(NA, nrow = n_draws, ncol = nobs(bsm[[1]]))
for (d in 1:n_draws) {
  k <- sample(1:length(w_pbma), size = 1, prob = w_pbma)
  ypred_bma[d, ] <- posterior_predict(bsm[[k]], draws = 1)
}

y_bma <- colMeans(ypred_bma)
d2 <- density(y_bma, kernel = c("gaussian"))$y
kld2 <- KLD(d2, d0)$sum.KLD.py.px

# PBMA+
ypred_bmabb <- matrix(NA, nrow = n_draws, ncol = nobs(bsm[[1]]))
for (d in 1:n_draws) {
  k <- sample(1:length(w_pbmabb), size = 1, prob = w_pbmabb)
  ypred_bmabb[d, ] <- posterior_predict(bsm[[k]], draws = 1)
}

y_bmabb <- colMeans(ypred_bmabb)
d3 <- density(y_bmabb, kernel = c("gaussian"))$y
kld3 <- KLD(d3, d0)$sum.KLD.py.px

# BHS
d_discrete <- 1
X <- dt0[, c("ESCS","HOMEPOS","ICTRES",
             "JOYREAD","PISADIFF","SCREADCOMP","SCREADDIFF",
             "METASUM","GFOFAIL","MASTGOAL","SWBP","WORKMAST","ADAPTIVITY","COMPETE",
             "PERFEED","TEACHINT","BELONG")] 

stan_bhs <- list(X = X, N = nrow(X), d = ncol(X), d_discrete = d_discrete,
                 lpd_point = lpd_point, K = ncol(lpd_point), tau_mu = 1,
                 tau_sigma = 1, tau_discrete = .5, tau_con = 1)

fit_bhs <- stan("bhs_stan.stan", data = stan_bhs, chains = 4, iter = 10000)

wts_bhs <- rstan::extract(fit_bhs, pars = 'w')$w
w_bhs_m <- as.matrix(apply(wts_bhs, 3, mean))

ypred_bhs_r <- matrix(NA, nrow = n_draws, ncol = nobs(bsm[[1]]))
for (d in 1:n_draws) {
  k <- sample(1:4, size = 1, prob = w_bhs_m)
  ypred_bhs_r[d, ] <- posterior_predict(bsm[[k]], draws = 1)
}

y_bhs_r <- colMeans(ypred_bhs_r)
d4 <- density(y_bhs_r, kernel = c("gaussian"))$y
kld4 <- KLD(d4, d0)$sum.KLD.py.px

# Combine results
wr_full <- data.frame(as.matrix(w_bs), as.matrix(w_pbma), as.matrix(w_pbmabb), w_bhs_m)
colnames(wr_full) <- c("bs","pbma", "pbmabb", "bhs")

klds_full <- rbind(kld1, kld2, kld3, kld4)

print(wr)
print(wr_full)

print(klds)
print(klds_full)
