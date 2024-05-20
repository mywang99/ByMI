
# ByMI procedure ----
ByMI <- function(id, mu_hat, Xbar2, Omega, alpha = 0.05, method = "Filtering", Sig_hat = NULL, proj_dim = 0, is_scale = T, is_plot = F, is_adjusted = F){
  m <- nrow(Xbar2) / 2 - 1
  p <- ncol(Xbar2)
  mu_hat = c(mu_hat)
  T1 <- t(t(Xbar2[seq(3, nrow(Xbar2), 2), ]) - mu_hat) 
  T2 <- t(t(Xbar2[seq(4, nrow(Xbar2), 2), ]) - mu_hat)
  if(proj_dim > 0) {
    if(is.null(Sig_hat)) {
      out_svd <- svd(T1)
      v <- out_svd$v
    }else{
      out_svd <- eigen(t(T1) %*% T1 / nrow(T1) - Sig_hat)
      v <- out_svd$vectors
    }
    T1 <- T1 %*% v[, 1:proj_dim]
    T2 <- T2 %*% v[, 1:proj_dim]
    if(is_scale == T)
      T1 <- T1 %*% solve(t(v[, 1:proj_dim]) %*% solve(Omega) %*% v[, 1:proj_dim])
  }else{
    if(is_scale == T)
      T1 <- T1 %*% Omega
  }
  
  W <- apply(T1 * T2, 1, sum)
  rm(list=c("T1", "T2"))
  
  #print(mean(W[id_clean -1]))
  W_sort <- sort(abs(W))
  Ta <- sapply(W_sort, function(x)
  {(1+(length(W[W <= (-x)]))) / max(1, length(W[W >= x]))
  })
  if(length(which(Ta <= alpha)) == 0) {
    if(is_adjusted == F){
      plot(W, xlab = method,col = "darkgrey", pch = 16, cex = 0.7)
      points(x = id-1, y = W[id - 1], col="red", pch = 17,cex = 0.7)
      return(c())
    }else{
      L <- min(W_sort[which.min(Ta)])
    }
  }else{
    L <- min(W_sort[which(Ta <= alpha)])
  }
  # U <- rep(0,m)
  # for(i in m:1){
  #   U[i] <- sum(W <= -W_sort[i]) / max(1, sum(W >= W_sort[i]))
  #   if(U[i] > alpha) break;
  # }
  # L <- W_sort[i+1]
  index_B <- which(W >= L)
  id_clean <- setdiff(c(2:(m+1)), id) 
  id_fd <- intersect(index_B, (id_clean - 1))
  if(is_plot == T){
    plot(W, xlab = method,col = "darkgrey", pch = 16, cex = 0.7)
    points(x = id-1, y = W[id - 1], col="red", pch = 17,cex = 0.7)
    points(x = index_B, y = W[index_B], col="green", pch = 17, cex = 0.7)
    points(x = id_fd, y = W[id_fd], col="blue", pch = 17, cex = 0.7)
    abline(h = L, lty = 2)
  }
  return(index_B+1)
}
est_dim_one <- function(w0, Z, ep, loop_max){
  w <- w0
  for(t in 1:loop_max){
    mu_w <- as.numeric(w %*% Z)
    Y <- (Z - mu_w)^2
    m_b <- floor(ep*length(Y))
    id_det <- tail(order(Y), m_b)
    id_choose <- setdiff(1:length(Y), id_det)
    w0 <- w
    w[id_det] <- 0
    w[id_choose] <- 1 / (length(Y)-m_b)
  }
  return(list(w %*% Z, w))
}

# Multinomial logistic regression ----
# Define the logistic regression model
logistic_regression <- function(X, theta) {
  X %*% theta
}

# Define the softmax function
softmax <- function(Z) {
  exp(Z) / sum(exp(Z))
}

# Define the cross-entropy loss function
cross_entropy_loss <- function(Y, Y_hat) {
  mean(Y * log(Y_hat))
}

# Define the gradient of the cross-entropy loss function
cross_entropy_loss_gradient <- function(X, Y, Y_hat) {
  t(X) %*% t(Y_hat - Y) / nrow(X)
}

# Define the gradient descent function
gradient_descent <- function(X, theta, Y) {
  Z <- logistic_regression(X, theta)
  # print(Z[1,])
  Y_hat <- apply(Z, 1, softmax)
  #print(Y_hat[1,])
  loss <- cross_entropy_loss(Y, Y_hat)
  gradient <- cross_entropy_loss_gradient(X, Y, Y_hat)
  return(list(gradient, loss))
}
`%!in%` <- Negate(`%in%`)
class.ind <- function(cl)
{
  n <- length(cl)
  cl <- as.factor(cl)
  x <- matrix( 0,  n ,  length(levels(cl)) )
  x[(1:n) + n*(unclass(cl)-1)] <- 1
  dimnames(x) <- list(names(cl), levels(cl))
  x
}
# Other detection methods ----
Krum <- function(X, f){
  m <- nrow(X)
  p <- ncol(X)
  score <- c()
  for(i in 1:m){
    dis <- apply(X, 1, function(x) sum((x-X[i,])^2))
    score[i] <- sum(sort(dis)[1:(m-f-2)])
  }
  i_star <- which.min(score)
  dis <- apply(X, 1, function(x) sum((x-X[i_star,])^2))
  id <- tail(order(dis), f)
  return(list(id, X[i_star, ]))
}

FABA <- function(X, alpha){
  k <- 1
  m <- nrow(X)
  g0 <- apply(X, 2, mean)
  id <- c()
  iter <- 0
  id_choose <- c(1:m)
  while(k <= ceiling(alpha*(m+1))){
    iter <- iter + 1
    dis <- apply(X, 1, function(x) sum((x-g0)^2))
    dis[id] <- 0
    id[iter] <- which.max(dis)
    id_choose <- setdiff(id_choose, id[iter])
    g0 <- apply(X[id_choose, ], 2, mean)
    k <- k + 1
  }
  G_hat <- apply(X[id_choose, ], 2, mean)
  return(list(id, G_hat))
}


Zeno <- function(X, theta0, learning_rate, rho, X_train0, y_train0, b){
  grad_zeno <- matrix(0, nrow = nrow(X) - 1, ncol = nrow(theta0) * ncol(theta0))
  update0 <- gradient_descent(X_train0,theta0, t(y_train0))
  loss0 <- update0[[2]]
  loss_zeno <- c()
  for(i in 2:nrow(X)){
    theta_zeno <- theta0 - learning_rate * matrix(X[i, ], ncol = ncol(theta0))
    update_zeno <-  gradient_descent(X_train0, theta_zeno, t(y_train0))
    loss_zeno[(i-1)] <- update_zeno[[2]]
  }
  scores <- -(loss_zeno - loss0 + rho*apply(X[-1,], 1, function(x) sum(x^2))) 
  index <- order(scores)[1:b]
  return(index)
}
RMDP<-function(y,alpha=0.05,itertime=100)   ############RMDP procedure
{
  n<-nrow(y)
  p<-ncol(y)
  h<-round(n/2)+1
  init_h=2
  delta=alpha/2
  bestdet=0
  jvec<-array(0,dim=c(n,1))
  for(A in 1:itertime)
  {
    id=sample(n,init_h)
    ny=y[id,]
    mu_t=apply(ny,2,mean)
    var_t=apply(ny,2,var) 
    dist<-apply((t((t(y)-mu_t)/var_t^0.5))^2,1,sum)
    crit=10
    l=0
    while(crit!=0&l<=15)
    {	l=l+1
    ivec<-array(0,dim=c(n,1))
    dist_perm<-order(dist)
    ivec[dist_perm[1:h]]=1
    crit=sum(abs(ivec-jvec))
    jvec=ivec
    newy=y[dist_perm[1:h],]
    mu_t=apply(newy,2,mean)
    var_t=apply(newy,2,var) 
    dist<-apply((t((t(y)-mu_t)/var_t^0.5))^2,1,sum)
    }
    tempdet=prod(var_t)
    if(bestdet==0|tempdet<bestdet) 
    {bestdet=tempdet
    final_vec=jvec
    }
  }
  submcd<-seq(1,n)[final_vec!=0]
  mu_t=apply(y[submcd,],2,mean)
  var_t=apply(y[submcd,],2,var) 
  dist<-apply((t((t(y)-mu_t)/var_t^0.5))^2,1,sum)
  dist=dist*p/median(dist)
  ER=cor(y[submcd,])
  ER=ER%*%ER
  tr2_h=sum(diag(ER))
  tr2=tr2_h-p^2/h
  cpn_0=1+(tr2_h)/p^1.5
  w0=(dist-p)/sqrt(2*tr2*cpn_0)<qnorm(1-delta)
  nw=sum(w0)
  sub=seq(1,n)[w0]	
  mu_t=apply(y[sub,],2,mean)
  var_t=apply(y[sub,],2,var) 
  ER=cor(y[sub,])
  ER=ER%*%ER
  tr2_h=sum(diag(ER))
  tr2=tr2_h-p^2/nw
  dist<-apply((t((t(y)-mu_t)/var_t^0.5))^2,1,sum)
  scale=1+1/sqrt(2*pi)*exp(-qnorm(1-delta)^2/2)/(1-delta)*sqrt(2*tr2)/p
  dist=dist/scale
  cpn_1=1+(tr2_h)/p^1.5
  w1=(dist-p)/sqrt(2*tr2*cpn_1)<qnorm(1-alpha)
  list(w1=w1)
}
bh.func<-function(pv, q)
{
  # the input
  # pv: the p-values
  # q: the FDR level
  # the output
  # nr: the number of hypothesis to be rejected
  # th: the p-value threshold
  # de: the decision rule
  
  m=length(pv)
  st.pv<-sort(pv)
  pvi<-st.pv/1:m
  de<-rep(0, m)
  if (sum(pvi<=q/m)==0)
  {
    k<-0
    pk<-1
  }
  else
  {
    k<-max(which(pvi<=(q/m)))
    pk<-st.pv[k]
    de[which(pv<=pk)]<-1
  }
  y<-list(nr=k, th=pk, de=de)
  return (y)
}

RMDP_BH <- function(X, mu_hat,n, Sig_hat, alpha, is_muhat = F){
  m <- nrow(X)
  p <- ncol(X)
  D <- diag(diag(Sig_hat)/ n) 
  D_inv <- diag(1 /( diag(Sig_hat)/ n))
  if(is_muhat == T){
    d2 <- lapply(c(1:m), function(i)
      t(as.matrix(X[i,] - as.numeric(mu_hat)))%*% D_inv %*% as.matrix(X[i,] - as.numeric(mu_hat)))
  }else{
    X_mean <- apply(X, 2 ,mean)
    X_scale <- t(t(X) - X_mean) * sqrt(n)
    res_RMDP <- RMDP(X_scale, alpha = 0.01, itertime = 100)
    X_hat_RMDP <- res_RMDP$w1 %*% X_scale / (sum(res_RMDP$w1 == 1))
    X_hat_RMDP <- X_hat_RMDP / sqrt(n) + X_mean
    d2 <- lapply(c(1:m), function(i)
      t(as.matrix(X[i,] - as.numeric(X_hat_RMDP)))%*% D_inv %*% as.matrix(X[i,] - as.numeric(X_hat_RMDP)))
  }
  R <- sqrt(D_inv)%*% (Sig_hat / n) %*% sqrt(D_inv)
  S_rmdp <- (as.numeric(d2) - p) / sqrt(2 * sum(diag(R%*%R)))
  pvalue <- pnorm(S_rmdp, lower.tail = FALSE)
  res_BH <- bh.func(pvalue, alpha)
  id_BH <- which(res_BH[[3]] == 1) 
  return(id_BH)
}

# Filtering algorithm (robust mean estimation) ----
filtering_refine <- function(Xbar, w0, ep, Sig_0, iter_pow = 5000, tol_pow = 1e-8){
  w <- w0
  m <- nrow(Xbar)
  d <- ncol(Xbar)
  mu_hat <- w %*% Xbar
  for (i in 1 : (ep * m)){
    cov <- t(Xbar * w) %*% Xbar - t(mu_hat) %*% mu_hat - Sig_0
    # eig <- eigen(cov)
    # eig_val <- eig$val[1]
    # eig_vec <- eig$vectors[,1]
    u <- rep(1 / sqrt(d), d)
    lambda <- 0
    for (j in 1 : iter_pow) {
      u_pow_old = u
      lambda_old = lambda
      u = u %*% cov 
      lambda = sqrt(sum(u ^ 2)) / sqrt(sum(u_pow_old ^ 2))
      u = u / sqrt(sum(u^2))
      if (abs((lambda - lambda_old)/lambda) < tol_pow) break
    }
    eig_vec <- u / sqrt(sum(u ^ 2))
    eig_val = eig_vec %*% cov %*% t(eig_vec)
    # expansion <- (2 * (1 - ep) / ((1 - 2 * ep) ^ 2)) * (1 + (d * log(d / 0.01) / (m * ep)))
    # if (eig_val * eig_val <= expansion * sigma * sigma / (m^2)) break
    
    # Choice 1
    tau <- apply(Xbar, 1, function(y) (((y - mu_hat) %*% t(eig_vec)) ^ 2))#- eig_vec %*% Sig_0 %*% t(eig_vec))
    tau[which(w < 1e-18)] = 0
    tau_idx <- which.max(abs(tau))
    tau_max <- tau[tau_idx]
    print(tau_max)
    if (eig_val >= 0){
      w <- w * (1 - tau / tau_max)
    }else{
      w <- w * (1 + tau / tau_max)
    }
    #  Xbar <- Xbar[-tau_idx, ]
    #  w <- w[-tau_idx]
    w[which(w < 1e-18)] = 0
    w <- w / sum(w)
    mu_hat <- w %*% Xbar
    w_std <- w / (max(w) * m)
    if(sum(w_std) <= 1 - ep) break
    # 
    # # Choice 2
    # Xbar <- Xbar[-tau_idx, ]
    # w <- w[-tau_idx]
    # w <- w / sum(w)
    # mu_hat <- w %*% Xbar
  }
  return(list(t(mu_hat), w))
}
