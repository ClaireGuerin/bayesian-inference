from mcmc import MCMC

#### PANDAS ####
# Initialize mu and sigma values
initial_mu_value = 80
initial_sigma_value = 5

# Parameters
nIterations = 10000
windowsize = 4
a_mu = 10
b_mu = 1000
alpha_sigma = 2
beta_sigma = 0.01

simPandas = MCMC("output_pandas.txt")
simPandas.normal_pandas(initial_mu_value, initial_sigma_value, nIterations, windowsize, a_mu, b_mu, alpha_sigma, beta_sigma)

# #### CLIMATE CHANGE ####
# # Initialize mu0, sigma0 and slope (a) values
# initial_mu0_value = 10
# initial_sigma0_value = 2
# initial_slope = 0.5

# # Parameters
# nIterations = 1000000
# assert False, "set a different window size for each parameter!"
# windowsize = 0.035
# a_mu0 = -10
# b_mu0 = 40
# alpha_sigma0 = 2
# beta_sigma0 = 1
# mu_slope = 0
# sigma_slope = 0.15


# simClimate = MCMC("output_climate.txt")
# simClimate.white_noise_linear_trend_climate(initial_mu0_value, initial_sigma0_value, initial_slope, nIterations, windowsize, a_mu0, b_mu0, alpha_sigma0, beta_sigma0, mu_slope, sigma_slope)

#### COMPARE BEAR AND PANDA BODY SIZE DISTRIBUTIONS - HYPER-PRIORS ####
# Initialize mu panda, sigma panda, mu bear, sigma bear, alpha and beta
initial_mu_panda_value = 80
initial_sigma_panda_value = 5
initial_mu_bear_value = 70
initial_sigma_bear_value = 5
initial_alpha_value = 2
initial_beta_value = 0.01

# Parameters
nIterations = 10000
winsize_mu = 4
winsize_sigma = 2
winsize_alpha = 1
a_mu = 10
b_mu = 1000


simPanBear = MCMC("output_panbear.txt")
simPanBear.normal_pandas_bears(initial_mu_panda_value, initial_sigma_panda_value, initial_mu_bear_value, initial_sigma_bear_value, initial_alpha_value, initial_beta_value, nIterations, winsize_mu, winsize_sigma, winsize_alpha, a_mu, b_mu)