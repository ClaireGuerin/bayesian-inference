from pdf import PDF as pdf
from data import pandas_bm
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def sliding_window(parameter, windowsize):
    lower = parameter - windowsize / 2
    upper = parameter + windowsize / 2
    new_parameter = np.random.uniform(lower,upper)
    log_hastings_ratio = 0
    return new_parameter, log_hastings_ratio

def plot_gamma(x_min,x_max,shape,rate):    
    x =  np.linspace(x_min, x_max, 1000)
    plt.plot(x, stats.gamma.pdf(x, a = shape, scale=1/rate))

# shape = 2
# rate = 2
# plot_gamma(0,100,shape,rate)

#### MCMC algorithm ####

def mcmc_pandas(initial_mu_value, initial_sigma_value, nIterations, windowsize, a_mu, b_mu, alpha_sigma, beta_sigma):
	with open("output.txt", "w") as f:

		f.write("iter\t posterior\t likelihood\t prior\t mu\t sigma\n")
		current_mu = initial_mu_value
		current_sigma = initial_sigma_value
		current_prior_mu = pdf.log_pdf_uniform(current_mu,a_mu,b_mu)
		current_prior_sigma = pdf.log_pdf_gamma(current_sigma,alpha_sigma,beta_sigma)
		current_likelihood = np.sum(pdf.log_pdf_normal(pandas_bm, current_mu, current_sigma))

		for iteration in range(nIterations):
			print('iteration number {0}'.format(iteration))

			# 1: propose new values for our parameters
			## in this example we are pretending there is no hastings ratio put out by our function 
			new_mu, log_h_ratio_mu = sliding_window(current_mu, windowsize)
			new_sigma, log_h_ratio_sigma = sliding_window(current_sigma, windowsize)
			## Make sure the new value is positive
			new_mu = abs(new_mu)
			new_sigma = abs(new_sigma)

			print("mu = {0}, sigma = {1}".format(new_mu, new_sigma))

			# 2: calculate the prior probability of these parameter values
			new_prior_mu = pdf.log_pdf_uniform(new_mu,a_mu,b_mu)
			new_prior_sigma = pdf.log_pdf_gamma(new_sigma,alpha_sigma,beta_sigma)

			#print("prior = {0}".format(prior_mu * prior_sigma))

			# 3: calculate the likelihood of the data with these parameter value
			new_likelihood = np.sum(pdf.log_pdf_normal(pandas_bm, new_mu, new_sigma))

			#print("likelihood = {0}".format(likelihood))

			# 4: calculate the posterior
			posterior_ratio = np.exp(new_likelihood + new_prior_mu + new_prior_sigma -(current_likelihood + current_prior_mu + current_prior_sigma))

			#print("posterior = {0}".format(posterior_ratio))

			# 5: compare new posterior with previous one and decide if we accept the new parameter values or not
			## we use our posterior ratio as our acceptance probability
			acceptance_probability = posterior_ratio
			## draw a random number from a uniform distribution between 0 and 1
			random_number = np.random.uniform(0,1)
			print(random_number <= acceptance_probability)
			## if smaller or equal to our acceptance probability -> accept. Otherwise -> reject
			## If accepted -> current_mu = new_mu
			## If rejected -> current_mu = current_mu
			if random_number <= acceptance_probability:
				print("accepted")
				current_mu = new_mu
				current_sigma = new_sigma
				current_likelihood = new_likelihood
				current_prior_mu = new_prior_mu
				current_prior_sigma = new_prior_sigma
			else:
				print("rejected")
			f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(iteration, posterior_ratio, current_likelihood, current_prior_mu + current_prior_sigma, current_mu, current_sigma))
			print("writing output...")


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

mcmc_pandas(initial_mu_value, initial_sigma_value, nIterations, windowsize, a_mu, b_mu, alpha_sigma, beta_sigma)