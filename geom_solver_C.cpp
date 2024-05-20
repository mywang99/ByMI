/**
 * @file geom_solver_C.cpp
 * @author Qian Chengde
 * @email qianchd at gmail dot com
 * @version 1.0
 * @date 2022-10-28
 * @brief geometric median
 * @details 
 * @see some url
 * @note some remark
 */


// we only include RcppArmadillo.h which pulls Rcpp.h in for us

#include <RcppArmadillo.h>
#include <iostream>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::field<arma::vec> geom_solver_C(
        const arma::mat &   X, 
        const bool &        is_echo,
        const double &      tol)
{
    arma::uword n = X.n_rows, d = X.n_cols;
    arma::vec med(d), med_old(d), eta(n);
    arma::vec square_x = arma::sum(arma::square(X), 1);
    
    int num_tie;
    double ave_scale, r_value, obj = DBL_MAX;
    arma::vec x_tie, x_tilde;
    arma::uvec ind_tie, ind_no_tie;
    
    med = arma::mean(X, 0).t();
    eta = arma::sqrt(square_x - 2 * X * med + arma::dot(med, med));
    
    arma::field<arma::vec> res(3);
    
    med_old = med;
    for(int i=0; i<=10000; ++i) {
        num_tie = arma::accu(eta < 1e-6);
        if(num_tie == 0) {
            med = X.t() * (1/eta) / arma::accu(1/eta);
        }
        else {
            ind_tie = arma::find(eta < 1e-6);
            ind_no_tie = arma::find(eta >= 1e-6);
            x_tie = arma::mean(X.rows(ind_tie), 0).t();
            x_tilde = X.rows(ind_no_tie).t() * (1 / eta(ind_no_tie));
            ave_scale = arma::accu(1 / eta(ind_no_tie));
            r_value = arma::norm(x_tilde - med * ave_scale);
            if(r_value <= num_tie) break;
            else {
                med = (1 - num_tie / r_value) * x_tilde + x_tie * num_tie / r_value;
            }
        }
        eta = arma::sqrt(square_x - 2 * X * med + arma::dot(med, med));
        obj = arma::accu(eta) / n;
        if(arma::norm(med_old - med) < tol) break;
        else med_old = med;
    }
    res(0) = med;
    res(1) = eta;
    res(2) = obj;
    return res;
}