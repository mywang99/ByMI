rm(list = ls())
shhh <- suppressPackageStartupMessages # It's a library, so shhh!
shhh(library(plyr))
shhh(library(foreach))
shhh(library(doParallel))

if(!require(geom, quietly = TRUE)) install.packages("./geom_1.0.tar.gz", repos = NULL, type = "source")
source("ByMI_function.R")

# Detect function ----
byz_detect <- function(attack= "OOD", nworker, s = 0.7, b = 1, s2 = 0.2, sig = 0.7, dataset, x_train, y_train, pca_num = 20, const_f = 1.2, b1 = 0.8, c_gamma = 0.2, alpha = 0.1, eps, num_classes, learning_rate, nepoch, is_adjusted = F){
  
  method_name = c("ByMI-Filter", "ByMI-GEOM",  "RMDP-BH", "Krum", "FABA", "Zeno","ByMI-Filter+")
  n_m  <- length(method_name)
  N <- nrow(x_train)
  b_index <- sort(sample(2:nworker, floor((nworker - 1) * eps), replace = F))
  id_clean <- setdiff(c(2:nworker), b_index)
  is_scale <- TRUE;
  
  ## split data uniformly ----
  id_num <- vector("list", 10)
  y_train_split <- list()
  x_train_split <- list()
  if(dataset == "mnist"){
    for(num in 1:10){
      id_num[[num]] <- which(y_train == num)
      local_size_num <-  length(id_num[[num]]) %/% nworker
      sizes_num <-  rep(local_size_num, nworker)
      sizes_num[1] <-  sizes_num[1] + length(id_num[[num]]) - sum(sizes_num)
      group_num <- sample(rep(1:nworker, sizes_num), sum(sizes_num), replace = F)
      if(num == 1){
        y_train_split <- split(y_train[id_num[[num]]], group_num)
        x_train_split0 <- split(x_train[id_num[[num]], ], group_num)
        x_train_split <- lapply(x_train_split0, function(x) as.numeric(t(matrix(x, ncol = pca_num))))
      }else{
        y_train_split <- mapply(append, y_train_split, split(y_train[id_num[[num]]], group_num), SIMPLIFY = F)
        x_train_split0 <- split(x_train[id_num[[num]], ], group_num)
        x_train_split0 <- lapply(x_train_split0, function(x) as.numeric(t(matrix(x, ncol = pca_num))))
        x_train_split <- mapply(append, x_train_split, x_train_split0, SIMPLIFY = F)
      }
    }
    sizes <- lengths(y_train_split)
    id0_num <- vector("list", 10)
    id01_n <- c()
    for(num in 1:10){
      id0_num[[num]] <- which(y_train_split[[1]] == num)
      id01_n <- c(id01_n, sample(id0_num[[num]], (N %/% nworker)/10, replace = F))
    }
    id01 <- setdiff(c(1:sizes[1]), id01_n)
    group01 <- rep(c(2:(nworker)), N / nworker - sizes[2] )
    y_train01 <- split(y_train_split[[1]][id01], group01)
    y_train01 <- c(list(c(rep(0,N / nworker - sizes[2]))), y_train01)
    y_train_split <- mapply(append, y_train_split , y_train01)
    y_train_split[[1]] <- y_train_split[[1]][id01_n]
    
    x_train_group <- lapply(x_train_split, function(x) matrix(x, byrow = T,ncol = pca_num))
    x_train01 <- split(x_train_group[[1]][id01,], group01)
    x_train01 <- lapply(x_train01, function(x) matrix(x, ncol = pca_num))
    
    x_train01 <- c(list(matrix(0,nrow = N / nworker - sizes[2], ncol = pca_num)), x_train01)
    x_train01 <- lapply(x_train01, function(x) as.numeric(t(x)))
    x_train_group <- lapply(x_train_group, function(x) as.numeric(t(x)))
    x_train_group <- mapply(append, x_train_group , x_train01)
    x_train_group <- lapply(x_train_group, function(x) matrix(x, byrow = T,ncol = pca_num))
    x_train_group[[1]] <- x_train_group[[1]][id01_n,]
    sizes <- lengths(y_train_split)
    y_train_group <- lapply(y_train_split, class.ind)
    rm(x_train_split, x_train_split0, x_train01, id01, id01_n, id0_num, id_num, group01, y_train01)
  }else if(dataset == "fashion"||dataset == "cifar10"){
    for(num in 1:10){
      id_num[[num]] <- which(y_train == num)
      local_size_num <-  length(id_num[[num]]) %/% nworker
      sizes_num <-  rep(local_size_num, nworker)
      sizes_num[1] <-  sizes_num[1] + length(id_num[[num]]) - sum(sizes_num)
      group_num <- sample(rep(1:nworker, sizes_num), sum(sizes_num), replace = F)
      if(num == 1){
        y_train_split <- split(y_train[id_num[[num]]], group_num)
        x_train_split0 <- split(x_train[id_num[[num]], ], group_num)
        x_train_split <- lapply(x_train_split0, function(x) as.numeric(t(matrix(x, ncol = pca_num))))
      }else{
        y_train_split <- lapply(c(1:nworker), function(i) append(y_train_split[[i]],  split(y_train[id_num[[num]]], group_num)[[i]])) #mapply(append, y_train_split, split(y_train[id_num[[num]]], group_num))
        x_train_split0 <- split(x_train[id_num[[num]], ], group_num)
        x_train_split0 <- lapply(x_train_split0, function(x) as.numeric(t(matrix(x, ncol = pca_num))))
        x_train_split <- lapply(c(1:nworker), function(i) append(x_train_split[[i]], x_train_split0[[i]])) #mapply(append, x_train_split, x_train_split0)
      }
    }
    sizes <- lengths(y_train_split)
    x_train_group <- lapply(x_train_split, function(x) matrix(x, byrow = T,ncol = pca_num))
    y_train_group <- lapply(y_train_split, class.ind)
    rm(x_train_split, x_train_split0, id_num)
  }
  ### size #####
  local_size <- sizes[2]
  batchsize <- local_size / 2
  
  ## contamination on X ####
  x_train_group_c <- x_train_group
  y_train_group_c <- y_train_group
  if(attack == "OOD"){
    x_mean <- apply(x_train, 2, mean)
    p <- length(x_mean)
    v <- rnorm(p)
    for(worker in b_index){
      local_size_w <- nrow(x_train_group_c[[worker]])
      x_train_group_c[[worker]] <- t(s * t(x_train_group_c[[worker]])  + b * v + s2 * matrix(rnorm(p * local_size_w), nrow = p)) 
    }
  }
  
  ## obtain theta0 ####
  acc_best <- 0
  counter <- 0
  for(epoch in 1:nepoch){
    if(epoch == 1){
      theta <- matrix(rep(1,(pca_num+1) * num_classes), ncol = num_classes)
      X_train <- cbind(x_train_group[[1]], rep(1,nrow(x_train_group[[1]])))
      y_train_onehot <- y_train_group[[1]]
    }
    learning_rate <- learning_rate
    length_grad <- (pca_num + 1) * 10
    update <- gradient_descent(X_train, theta, t(y_train_onehot))
    grad01 <- update[[1]]
    loss0 <- update[[2]]
    theta <- theta - learning_rate * grad01
    scores <- logistic_regression(X_test, theta)
    y_pred_lg <- apply(scores, 1, which.max)
    lg <- table(y_pred_lg,y_test)
    acc <- sum(diag(lg)) / sum(lg)
    if(acc > acc_best) {
      acc_best <- acc
      theta_best = theta
      counter = 0
    }
    if(acc < acc_best) counter = counter + 1
    if(counter > 10) break
    cat(sprintf("Epoch %d: train loss: %3f accuracy: %f\n", epoch, loss0 , acc))
  }
  theta0 <- theta_best
  
  ## calculate gradients ####
  grad <- matrix(0, nrow = nworker, ncol = length_grad)
  grad1 <- matrix(0, nrow = nworker, ncol = length_grad)
  grad1_clean <- matrix(0, nrow = nworker, ncol = length_grad)
  grad2 <- matrix(0, nrow = nworker, ncol = length_grad)
  grad2_clean <- matrix(0, nrow = nworker, ncol = length_grad)
  j <- 0
  id1_worker <- vector("list", nworker)
  id2_worker <- vector("list", nworker)
  for(i in 1:nworker){
    X_train <- cbind(x_train_group_c[[i]], rep(1,nrow(x_train_group_c[[i]])))
    X_train_clean <- cbind(x_train_group[[i]], rep(1,nrow(x_train_group[[i]])))
    y_train_onehot <- y_train_group_c[[i]]
    y_train_onehot_clean <- y_train_group[[i]]
    id1_num <- vector("list", 10)
    id1_n <- c()
    id_sign = 0
    for(num in 1:10){
      id1_num[[num]] <- which(y_train_split[[i]] == num)
      if(length(id1_num[[num]]) %% 2 == 0)
        id1_n <- c(id1_n, sample(id1_num[[num]], floor(length(id1_num[[num]])/2), replace = F))
      else{
        id1_n <- c(id1_n, sample(id1_num[[num]], floor(length(id1_num[[num]])/2) + id_sign, replace = F))
        id_sign <- !id_sign
      }
    }
    id1 <- setdiff(c(1:local_size), id1_n)
    id2 <- id1_n
    id1_worker[[i]] <- id1
    id2_worker[[i]] <- id2
    update1 <- gradient_descent(X_train[id1, ],
                                theta0, t(y_train_onehot[id1, ]))
    
    grad1_i <- update1[[1]]
    update1_clean <- gradient_descent(X_train_clean[id1, ],
                                      theta0, t(y_train_onehot_clean[id1, ]))
    
    grad1_i_clean <- update1_clean[[1]]
    update2 <- gradient_descent(X_train[id2, ],
                                theta0, t(y_train_onehot[id2, ]))
    grad2_i <- update2[[1]]
    update2_clean <- gradient_descent(X_train_clean[id2, ],
                                      theta0, t(y_train_onehot_clean[id2, ]))
    grad2_i_clean <- update2_clean[[1]]
    grad[i, ] <- as.numeric((grad1_i*(length(id1)) + grad2_i*length(id2)) / local_size)
    grad1[i, ] <- as.numeric(grad1_i)
    grad1_clean[i, ] <- as.numeric(grad1_i_clean)
    grad2[i, ] <- as.numeric(grad2_i)
    grad2_clean[i, ] <- as.numeric(grad2_i_clean)
    ## estimate Sig_hat ####
    if(i == 1){
      Sig_hat = 0
    }
    if(i == 1){
      n_grad <- local_size 
      grad0 <- matrix(0, nrow = n_grad, ncol = length_grad)
      iter <- 0
      X_train0 <- cbind(x_train_group_c[[1]], rep(1,nrow(x_train_group_c[[1]])))
      y_train_onehot0 <- y_train_group_c[[1]]
      for(k in 1:local_size){
        iter <- iter + 1
        update0 <- gradient_descent(t(as.matrix(X_train0[k, ])), theta0, as.matrix(y_train_onehot0[k, ]))
        grad0[iter, ] <- as.numeric(update0[[1]])
      }
      grad0_mean <- apply(grad0, 2, mean)
      Sig_hat <-  (t(grad0) - grad0_mean) %*% t(t(grad0) - grad0_mean) / n_grad
    }
  }
    
  ## attack gradients ----
  if(attack == "grad"){
    grad_all_mean <- apply(grad, 2, mean)
    grad_all_sd <- apply(grad, 2, sd)
    v <- rnorm(length_grad)
    v <- (v / sqrt(sum(v^2))) * grad_all_sd * sqrt(local_size)
    ss <- sig * grad_all_sd
    random_values1 <- foreach(i = 1:length(b_index), .combine = "rbind") %do% rnorm(length_grad, sd = ss) / sqrt(local_size / 2)
    random_values2 <- foreach(i = 1:length(b_index), .combine = "rbind") %do% rnorm(length_grad, sd = ss) / sqrt(local_size / 2)
    grad1[b_index, ] <- t(t(random_values1) + grad_all_mean + b1 * v)
    grad2[b_index, ] <- t(t(random_values2) + grad_all_mean + b1 * v)
    grad[b_index, ] <- (grad1[b_index, ] + grad2[b_index,]) / 2
  }else if(attack == "IPM"){
    grad_o <- apply(grad[id_clean, ], 2, mean)
    grad_c <- - c_gamma * grad_o
    for(i in b_index){
      grad1[i, ] <- grad_c
      grad2[i, ] <- grad_c
      grad[i, ] <- grad_c
    }
  }
  ## detection procedure ----
  ### initialization ----
  FDP <- Power <- size_det <- error <-  rep(0, n_m)
  w0 <- rep(1/nworker , nworker)
  ### estimate grad_hat ----
  grad1_mean <- apply(grad1, 2, mean)
  grad1_scale <- t(t(grad1) - grad1_mean) * sqrt(batchsize)
  
  ### obtain the projection direction (the largest eigenvector of the sample covariance matrix) ####
  Sig1 <- t(grad1_scale) %*% grad1_scale / nworker
  v1 <-  eigen(Sig1)$vectors[ , 1]
  
  ### scale estimator ----
  MAD <- apply(grad, 2, mad)
  scale_MAD <- solve(diag(MAD))
  scale_proj <- v1 %*% t(v1)
  
  mean_clean <- apply(grad1_clean, 2 ,mean)
  grad_mean <- apply(grad, 2 ,mean)
  grad_scale <- t(t(grad) - grad_mean) * sqrt(local_size)
  
  ### Sample split ----
  grad_split <- matrix(0, nrow = 2*nworker, ncol = length_grad)
  for(jj in 1:nworker){
    grad_split[(2*jj-1),] <- grad1[jj, ]
    grad_split[2*jj,] <- grad2[jj, ]
  }
  id <- b_index
  id_clean <- setdiff(c(2:nworker), b_index)
  grad_clean <- apply(grad[id_clean, ], 2, mean)
  
  ### Various methods ----
  #### ByMI-Filtering ----
  out_rme_refine <- filtering_refine(grad1_scale, w0, ep = const_f * eps, Sig_0 = Sig_hat, iter_pow = 5000, tol_pow = 1e-8)
  grad_hat_scrme <- out_rme_refine[[1]]  / sqrt(batchsize) + grad1_mean
  id_sda_scrme <- ByMI(id = id, mu_hat = grad_hat_scrme, Xbar2 = grad_split, is_plot = F,
                               is_scale = is_scale, Omega = scale_MAD, alpha = alpha, method = "Filtering", is_adjusted = is_adjusted)
  id_sda_1 <- setdiff(c(2:nworker), id_sda_scrme)
  grad_scrme <- apply(matrix(grad[id_sda_1, ], nrow = length(id_sda_1)), 2, mean)
  error[1] <- sum((grad_scrme - grad_clean)^2)
  FDP[1] <- length(intersect(id_sda_scrme, id_clean)) / max(1, length(id_sda_scrme))
  Power[1] <- length(intersect(id_sda_scrme, b_index)) / max(1, length(b_index))
  size_det[1] <- length(id_sda_scrme)
  
  #### ByMI-GEOM ----
  grad_hat_gmom <- geom_solver_C(X = grad1_scale, tol = 1e-10, is_echo = F)[[1]]/ sqrt(batchsize)   + grad1_mean
  id_sda_gmom <- ByMI(id = id, mu_hat = grad_hat_gmom, Xbar2 = grad_split, is_plot = F,
                              is_scale = is_scale, Omega = scale_MAD, alpha = alpha, method = "GMOM", is_adjusted = is_adjusted)
  id_sda_gmom1 <- setdiff(c(2:nworker), id_sda_gmom)
  grad_gmom <- apply(matrix(grad[id_sda_gmom1, ], nrow = length(id_sda_gmom1)), 2, mean)
  error[2] <- sum((grad_gmom - grad_clean)^2)
  FDP[2] <- length(intersect(id_sda_gmom, id_clean)) / max(1, length(id_sda_gmom))
  Power[2] <- length(intersect(id_sda_gmom, id)) / max(1, length(id))
  size_det[2] <- length(id_sda_gmom)
  
  #### RMDP-BH ####
  g0 <- grad_hat_scrme
  id_BH <- RMDP_BH(X = grad[-1, ], mu_hat = g0, n = local_size, is_muhat = F, Sig_hat = Sig_hat, alpha = alpha) + 1
  if(length(id_BH) == (nworker - 1)){
    grad_bh <- rep(0, length_grad)
  }else{
    id_BH_1 <- setdiff(c(2:nworker), id_BH)
    if(length(id_BH_1) == 1){
      grad_bh <- grad[id_BH_1]
    }else{
      grad_bh <- apply(grad[id_BH_1, ], 2, mean)
    }
  }
  error[3] <- sum((grad_bh- grad_clean)^2)
  FDP[3] <- length(intersect(id_BH, id_clean)) / max(1, length(id_BH))
  Power[3] <- length(intersect(id_BH, id)) / max(1, length(id))
  size_det[3] <- length(id_BH)
  
  
  #### KRUM ####
  id_Krum <- Krum(grad_scale[-1,], f = ceiling(nworker*const_f*eps))[[1]] + 1
  id_Krum_1 <- setdiff(c(2:nworker), id_Krum)
  grad_Krum <- apply(matrix(grad[id_Krum_1, ], nrow = length(id_Krum_1)), 2, mean)
  error[4] <- sum((grad_Krum- grad_clean)^2)
  FDP[4] <- length(intersect(id_Krum, id_clean)) / max(1, length(id_Krum))
  Power[4] <- length(intersect(id_Krum, id)) / max(1, length(id))
  size_det[4] <- length(id_Krum)
  
  #### FABA ####
  id_FABA <- FABA(grad_scale[-1,], alpha = const_f*eps)[[1]] + 1
  id_FABA_1 <- setdiff(c(2:nworker), id_FABA)
  grad_FABA <- apply(matrix(grad[id_FABA_1, ], nrow = length(id_FABA_1)), 2, mean)
  error[5] <- sum((grad_FABA- grad_clean)^2)
  FDP[5] <- length(intersect(id_FABA, id_clean)) / max(1, length(id_FABA))
  Power[5] <- length(intersect(id_FABA, id)) / max(1, length(id))
  size_det[5] <- length(id_FABA)
  
  
  #### Zeno ####
  X_train0 <- cbind(x_train_group_c[[1]], rep(1,nrow(x_train_group_c[[1]])))
  y_train_onehot0 <- y_train_group_c[[1]]
  res_Zeno <- Zeno(grad_scale, X_train0 = X_train0, y_train0 = y_train_onehot0, 
                   b = ceiling(nworker*const_f*eps), theta0 = theta0, rho = 0.01, learning_rate = learning_rate)
  id_Zeno <- res_Zeno + 1
  id_Zeno_1 <- setdiff(c(2:nworker), id_Zeno)
  grad_Zeno <- apply(matrix(grad[id_Zeno_1, ], nrow = length(id_Zeno_1)), 2, mean)
  error[6] <- sum((grad_Zeno- grad_clean)^2)
  FDP[6] <- length(intersect(id_Zeno, id_clean)) / max(1, length(id_Zeno))
  Power[6] <- length(intersect(id_Zeno, id)) / max(1, length(id))
  size_det[6] <- length(id_Zeno)
  
  #### ByMI-Filter+ (Projection one dimension) -----
  Z <- grad1 %*% v1
  grad_hat_p <- est_dim_one(w0 = w0, Z = Z, ep = const_f*eps, loop_max = 2000)[[1]]
  id_sda_p <- ByMI(id = id, mu_hat = grad_hat_p, Xbar2 = grad_split %*% v1, is_plot = F,
                           is_scale = is_scale, Omega = 1, alpha = alpha, method = "Projection", is_adjusted = is_adjusted)
  id_sda_p_1 <- setdiff(c(2:nworker), id_sda_p)
  grad_p <- apply(matrix(grad[id_sda_p_1, ], nrow = length(id_sda_p_1)), 2, mean)
  error[7] <- sum((grad_p- grad_clean)^2)
  FDP[7] <- length(intersect(id_sda_p, id_clean)) / max(1, length(id_sda_p))
  Power[7] <- length(intersect(id_sda_p, id)) / max(1, length(id))
  size_det[7] <- length(id_sda_p)
  
  id_list <- list(id_sda_scrme, id_sda_gmom, id_BH, id_Krum, id_FABA, id_Zeno, id_sda_p)
  
  return(list(as.numeric(FDP), as.numeric(Power), as.numeric(size_det), as.numeric(error), id_list, id))
}

# parameter set ----
dataset <- "mnist";  nworker <- 150; learning_rate = 0.1
# dataset <- "fashion"; nworker <- 150; learning_rate = 0.05
# dataset <- "cifar10"; nworker <- 125; learning_rate = 0.05

attack <- "OOD"
# attack <- "IPM"
# attack <- "grad"

eps <- 0.1; pca_num <- 20; const_f <- 1.2; num_classes = 10
s <-  0.7; s2 <- 0.2; b <- 1;# OOD attack
sig <- 0.7; b1 <- 0.8; # Random gradient attack
c_gamma <- 0.2; # IPM attack


load(paste0("./data/", dataset,"_resnet18_pca_",pca_num, ".Rdata"))
X_test <- cbind(x_test,rep(1,nrow(x_test)))
pca_num <- ncol(x_train)

if(attack == "OOD"){
  tem <- c(dataset, attack, nworker, const_f, eps, learning_rate, b, s, s2)
  names(tem) <- c("dataset", "attack", "nworker", "const_f", "eps", "lr", "b", "s", "s2")
}else if(attack == "grad"){
  tem <- c(dataset, attack, nworker, const_f, eps, learning_rate, b1, sig)
  names(tem) <- c("dataset", "attack", "nworker", "const_f", "eps", "lr", "b1", "sig")
}else if(attack == "IPM"){
  tem <- c(dataset, attack, nworker, const_f, eps, learning_rate, c_gamma)
  names(tem) <- c("dataset", "attack", "nworker", "const_f", "eps", "lr", "c_gamma")
}
print(tem)

# Experiment ----
ncore <- 2
rep_times <- 10
cl <- makeCluster(ncore)
registerDoParallel(cl)
res1 <- foreach(j = 1:rep_times, .combine = 'rbind', .packages = c("geom", "plyr", "foreach")) %dopar%
  byz_detect(attack= attack, nworker = nworker,  s = s, b = b, s2 = s2, sig = sig, b1 = b1, dataset = dataset,
             x_train = x_train, pca_num = pca_num, y_train = y_train, const_f = const_f, c_gamma = c_gamma,
             eps = eps, num_classes = num_classes, learning_rate = learning_rate, alpha = 0.1, nepoch = 200, is_adjusted = F)
stopCluster(cl)

# Summary result ----
FDR <- foreach(i = 1:rep_times, .combine = "rbind") %do% {return(res1[i, 1][[1]])}
Power <- foreach(i = 1:rep_times, .combine = "rbind") %do% {return(res1[i, 2][[1]])}
size_det <- foreach(i = 1:rep_times, .combine = "rbind") %do% {return(res1[i, 3][[1]])}
error <- foreach(i = 1:rep_times, .combine = "rbind") %do% {return(res1[i, 4][[1]])}

alpha_n <- 0.1
method_name = c("ByMI-Filter", "ByMI-GEOM",  "RMDP-BH", "Krum", "FABA", "Zeno","ByMI-Filter+")
method_name_order = c("ByMI-Filter", "ByMI-Filter+", "ByMI-GEOM",  "RMDP-BH", "Krum", "FABA", "Zeno")
n_m = length(method_name)

data <- data.frame(alpha = rep(alpha_n, each = n_m), FDR = apply(FDR, 2 ,mean), Power = apply(Power, 2, mean), 
                   Pa = apply(Power, 2, function(x) sum(x==1)) / rep_times, detsize = apply(size_det, 2, mean), 
                   RMSE = sqrt(apply(error, 2, mean)),
                   method = rep(method_name, length(alpha_n)))
data$method <- factor(data$method, levels = method_name_order)
data <- data[order(data$method), ]
print(data)


print("####### sd ########")
data_sd <- data.frame(alpha = rep(alpha_n, each = n_m), FDR = apply(FDR, 2 ,sd) / sqrt(rep_times), Power = apply(Power, 2, sd) / sqrt(rep_times), 
                      detsize = apply(size_det, 2, sd)/ sqrt(rep_times), error = apply(error, 2, sd)/ sqrt(rep_times),
                      method = rep(method_name, length(alpha_n)))
data_sd$method <- factor(data_sd$method, levels = method_name_order)
data_sd <- data_sd[order(data_sd$method), ]
print(data_sd)

# save data ----
# if(attack == "OOD"){
#   save(res1, tem, rep_times, file = paste0(dataset, "_FDR_attack_",attack, "_b", b, "_eps", opt$eps, "_rep", rep_times, "_", time_token, ".Rdata"))
# }else if(attack == "grad"){
#   save(res1, tem, rep_times, file = paste0(dataset, "_FDR_attack_",attack, "_b", b1, "_eps", opt$eps, "_rep", rep_times, "_", time_token, ".Rdata"))
# }else if(attack == "IPM"){
#   save(res1, tem, rep_times, file = paste0(dataset, "_FDR_attack_",attack, "_c", c_gamma, "_eps", opt$eps, "_rep", rep_times, "_", time_token, ".Rdata"))
# }

