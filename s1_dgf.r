dgf <- function(ni, nj, gamma00, gamma01, gamma02, gamma03, gamma04, u_0, w_0, sigma){
  
  # for random effects and sd
  corU <- matrix(c(1, 0, 0, 0, 1, 0, 0, 0, 1), nrow = 3, ncol = 3, byrow = T)
  sdU <- matrix(c(u_0, 0, 0, 0, u_0, 0, 0, 0, u_0), nrow = 3, ncol = 3, byrow = T)
  covU <- sdU%*%corU%*%sdU
  I <- covU
  
  # for fixed effects and sd
  corW <- diag(4)
  sdW <- diag(4)*w_0
  covW <- sdW%*%corW%*%sdW
  J <- covW
  
  # for outer loop or the macro level --> school levels
  for (i in 1:ni){
    
    U[i,1] <- I[1,1]*rnorm(1, 0, .5)
    U[i,2] <- I[1,2]*U[i,1] + I[2,2]*rnorm(1)
    U[i,3] <- I[1,3]*U[i,1] + I[1,2]*U[i,2] + I[3,3]*rnorm(1)
    
    # x[i, 4] <- rbinom(1, 1, .5) # public
    # x[i, 5] <- rnorm(1, .15, 1) # sd
    
    # For inner loop or the micro level --> individual levels
    for (j in 1:nj){
      
      x[j,1] <- J[1,1]*rnorm(1, means[5], sds[5])
      x[j,2] <- J[1,2]*x[j,1] + J[2,2]*rnorm(1, means[6], sds[6]) 
      x[j,3] <- J[1,3]*x[j,1] + J[2,3]*x[j,2] + J[3,3]*rnorm(1, means[7], means[7])
      x[j,4] <- J[1,4]*x[j,1] + J[2,4]*x[j,2] + J[3,4]*x[j,3] + J[4,4]*rnorm(1, means[8], sds[8])
      
    }
    
    ind <- 1
    
    for (i in 1:ni){
      for (j in 1:nj) {
        
        # for response and error
        
        r[ind, 1] <- sigma*rnorm(1, 0, 1)
        y[ind, 1] <- gamma00 + gamma01*x[j,1] + gamma02*x[j,2] + gamma03*x[j,3] + gamma04*x[j,4] + U[i,1] + r[ind, 1]
        
        # pull out the data
        
        tmp <- c(i, j, y[ind, 1], x[j, 1], x[j, 2], x[j, 3], x[j, 4], U[i,1], r[ind, 1])
        
        sim[ind, ] <- tmp
        
        ind <- ind + 1
      }
    }
  }
  
  colnames(sim) <- c("i", "j", "y", "x1", "x2", "x3", "x4", "u_0", "r")
  
  sim <- as.data.frame(sim)
  
  return(sim)
  
}
