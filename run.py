from mcmc import MCMC

#### PANDAS ####
# Initialize mu and sigma values
initial_mu_value = 80
initial_sigma_value = 5

# Parameters
nIterations = 100000
windowsize = 4
a_mu = 10
b_mu = 1000
alpha_sigma = 2
beta_sigma = 0.01

simPandas = MCMC("output_pandas.txt")
simPandas.normal_pandas(initial_mu_value, initial_sigma_value, nIterations, windowsize, a_mu, b_mu, alpha_sigma, beta_sigma)

#### CLIMATE CHANGE ####
# Initialize mu0, sigma0 and slope (a) values
initial_mu0_value = 10
initial_sigma0_value = 2
initial_slope = 0.5

# Parameters
nIterations = 100000
windowsize = 4
a_mu0 = -10
b_mu0 = 40
alpha_sigma0 = 2
beta_sigma0 = 1
mu_slope = 0
sigma_slope = 0.15


simClimate = MCMC("output_climate.txt")
simClimate.white_noise_linear_trend_climate(initial_mu0_value, initial_sigma0_value, initial_slope, nIterations, windowsize, a_mu0, b_mu0, alpha_sigma0, beta_sigma0, mu_slope, sigma_slope)