### implementing a model to test whether or not the body size of bears and pandas follow significantly different distributions ###


####### Likelihood/Prior functions ######

log_pdf_exp <- function(x, l){
  # convert input into vector
  x = c(x)
  loglik = log(l)-l*x
  # sum up loglik values to get lik of whole data
  loglik_data = sum(loglik)
  return(loglik_data)
}


log_pdf_uniform <- function(x,lower,upper){ # assumes that x is a single number 
  # convert input into vector
  x = c(x)
  # if the minimum and maximum values of the array lay within the boundaries
  if (min(x)>lower & max(x)<upper){
    loglik = -log(upper-lower)
    # add together one lik value per data point, i.e. multiply lik with length(x)
    loglik_data = loglik*length(x)
  }else{ # if at least one of the values is outside boundaries, the lik is -inf
    loglik_data = -Inf
  }
  return(loglik_data)
}


log_pdf_normal <- function(x,mu,sd){ 
  # convert input into vector
  x = c(x)
  log_lik = -0.5*log(2*pi*sd^2) - ((x - mu)^2)/(2*sd^2)
  # sum up loglik values to get lik of whole data
  loglik_data = sum(log_lik)
  return(loglik_data)
}


log_pdf_gamma <- function(x,a,b){
  # convert input into vector
  x = c(x)
  log_lik = (a-1)*log(x)+(-b*x)-(log(b)*(-a)+ lgamma(a))
  # sum up loglik values to get lik of whole data
  loglik_data = sum(log_lik)
  return(loglik_data)
}


####### Proposal functions ############
sliding_window <- function(x,windowsize){
  lower = x-0.5*windowsize
  upper = x+0.5*windowsize
  new_x = runif(1, lower, upper)
  new_x = abs(new_x)
  hastings_ratio = 0
  return(c(new_x,hastings_ratio))
}

multiplier_proposal <- function(i,d=1.2){ # d must be > 1!!
  u <- runif(1)
  l <- 2*log(d)
  m <- exp(l*(u-.5))
  new_x <- i * m
  hastings_ratio <- log(m)
  return( c(new_x, hastings_ratio) )
}


####### Data #######
pandas_bm =c(84.74, 84.88, 94.60, 96.37, 102.93, 109.11, 125.76) # body sizes pandas
bears_bm = c(67.65, 92.13, 58.92, 87.64, 76.31, 88.86) # body sizes brown bears



####### Logfile #######
setwd("/Users/gfennmolt/Documents/Courses/Bayesian_analysis/Bayesian_R")
logfile <- "r_mcmc_samples_2bears.txt"

cat(c("it","post","likelihood","prior","mu_panda","mu_bear","dif_mu", "sigma_panda","sigma_bear", "shape", 'rate', "\n"),file=logfile,sep="\t")



####### Priors #######
# define priors: uniform on mu
minMu = 10
maxMu = 1000

hp_rate = 0.1 




###### Define the starting point ######
# initial parameters (could also be drawn randomely instead)
current_mu_panda   = 80 #runif(1,minMu,maxMu)
current_sig_panda  = 5 #rgamma(1,shape,rate)
current_mu_bear   = 80 #runif(1,minMu,maxMu)
current_sig_bear  = 5 #rgamma(1,shape,rate)

# and for the hyper-priors 
current_shape_hp = 1 #rexp(1, 0.1)
current_rate_hp = 1 #rexp(1, 0.1)





###### Initial likelihood and priors #######
current_lik = sum(log_pdf_normal(pandas_bm,current_mu_panda,current_sig_panda), log_pdf_normal(bears_bm,current_mu_bear,current_sig_bear)) # add likelihoods together to get the likelihood of all data

current_prior = sum(log_pdf_uniform(current_mu_panda, minMu, maxMu), log_pdf_gamma(current_sig_panda, current_shape_hp, current_rate_hp), log_pdf_uniform(current_mu_bear, minMu, maxMu), log_pdf_gamma(current_sig_bear, current_shape_hp, current_rate_hp), log_pdf_exp(current_shape_hp, hp_rate), log_pdf_exp(current_shape_hp, hp_rate)) # remember to add together the log-prior probs of all parameters

current_posterior = sum(current_lik, current_prior) # apply Bayes theorem to calculate the posterior



###### MCMC settings #######
n_iterations  = 100000
sampling_freq = 100 # write to file every n iterations
print_freq    = 5000 # print to screen every 5000 iterations

window_size_update_mu = 4 
window_size_update_sig = 1.2 
window_size_update_hp_a = 1.4 
window_size_update_hp_b = 1.4 



################ MCMC loop #######################
for (iteration in 0:n_iterations){
  
  # reset new parameters
  new_mu_panda = current_mu_panda
  new_mu_bear = current_mu_bear

  new_sig_panda = current_sig_panda
  new_sig_bear = current_sig_bear
  
  new_shape_hp = current_shape_hp
  new_rate_hp = current_rate_hp
  
  
  ####### Propose new parameters #######
  # propose new mu
  proposal_mu_panda <- sliding_window(current_mu_panda, window_size_update_mu) # use the proposal function and pre-defined window sizes in our settings to propose a new value for mu
  proposal_mu_bear <- sliding_window(current_mu_bear, window_size_update_mu) 
  new_mu_panda <- proposal_mu_panda[1]
  new_mu_bear <- proposal_mu_bear[1]
  hastings_ratio_mu <- proposal_mu[2]
  
  # propose new sigma
  proposal_sig_panda <- sliding_window(current_sig_panda, window_size_update_sig) # same for sigma
  proposal_sig_bear <- sliding_window(current_sig_bear, window_size_update_sig) 
  new_sig_panda <- abs(proposal_sig_panda[1]) # sigma however cannot be negative, so we use reflection at the boundary (0)
  new_sig_bear <- abs(proposal_sig_bear[1])
  hastings_ratio_sig_panda <- proposal_sig_panda[2]
  hastings_ratio_sig_bear <- proposal_sig_bear[2]
  
  # propose new shape_hp 
  proposal_shape_hp <- sliding_window(current_shape_hp, window_size_update_hp_a) 
  new_shape_hp <- abs(proposal_shape_hp[1]) 
  hastings_ratio_shape_hp<- proposal_shape_hp[2]

  # propose new rate_hp 
  proposal_rate_hp <- sliding_window(current_rate_hp, window_size_update_hp_b) 
  new_rate_hp <- abs(proposal_rate_hp[1]) 
  hastings_ratio_rate_hp<- proposal_rate_hp[2]
  
  # get the overall hastings ratio of new parameter proposals
  hastings_ratio = sum(hastings_ratio_mu_panda, hastings_ratio_mu_bear, hastings_ratio_sig_panda, hastings_ratio_sig_bear, hastings_ratio_shape_hp, hastings_ratio_rate_hp)

  
  ###### Calculate the Posterior Ratio #######
  # new likelihood and priors
  new_lik = sum(log_pdf_normal(pandas_bm,new_mu_panda,new_sig_panda), log_pdf_normal(bears_bm,new_mu_bear, new_sig_bear)) # calculate the likelihood under the new proposed parameter values
  new_prior = sum(log_pdf_uniform(new_mu_panda, minMu, maxMu), log_pdf_gamma(new_sig_panda, new_shape_hp, new_rate_hp), log_pdf_uniform(new_mu_bear, minMu, maxMu), log_pdf_exp(new_shape_hp, hp_rate), log_pdf_exp(new_rate_hp, hp_rate),log_pdf_gamma(new_sig_bear, new_shape_hp, new_rate_hp))
  
  # calculate the prior probability of the new parameter values (remember to add together the log-probs of the individual parameters)
  new_posterior = new_lik + new_prior # apply Bayes theorem to calculate the new posterior
  
  # calculate the posterior ratio 
  posterior_ratio = new_posterior - current_posterior
  
  # get the acceptance ratio from posterior ratio and Hastings ratio
  r = posterior_ratio + hastings_ratio
 
  # accept or reject proposal
  random_number = runif(1)
  if (log(random_number) <= r){
    # if new state accepted, set current parameters to the new ones
    current_mu_panda = new_mu_panda
    current_mu_bear = new_mu_bear
    current_sig_panda = new_sig_panda
    current_sig_bear = new_sig_bear
    current_shape_hp = new_shape_hp
    current_rate_hp = new_rate_hp 
    current_lik = new_lik
    current_prior = new_prior
    current_posterior = new_posterior
  }
  # print to screen
  if (iteration %% print_freq == 0){
    print(c(iteration,current_lik, current_prior, current_mu_panda, current_mu_bear, current_sig_panda, current_sig_bear, current_shape_hp, current_rate_hp))
  }
  # save to file
  if (iteration %% sampling_freq == 0){
    cat(c(iteration, current_posterior, current_lik, current_prior, current_mu_panda, current_mu_bear, current_mu_panda-current_mu_bear, current_sig_panda, current_sig_bear, current_shape_hp, current_rate_hp,"\n"), sep="\t", file=logfile, append=T)
  }
}


