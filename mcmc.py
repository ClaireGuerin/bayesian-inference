from pdf import PDF as pdf
from data import pandas_bm
from data import climate
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def plot_gamma(x_min,x_max,shape,rate):    
    x =  np.linspace(x_min, x_max, 1000)
    plt.plot(x, stats.gamma.pdf(x, a = shape, scale=1/rate))

# shape = 2
# rate = 2
# plot_gamma(0,100,shape,rate)

#### MCMC algorithm ####

class MCMC(object):

	def __init__(self, outputfile):
		self.outputfile = outputfile

	def sliding_window(self, parameter, windowsize):
		lower = parameter - windowsize / 2
		upper = parameter + windowsize / 2
		new_parameter = np.random.uniform(lower,upper)
		log_hastings_ratio = 0
		return new_parameter, log_hastings_ratio

	def multiplier_proposal(self, x,d=1.2):
		u = np.random.uniform(0,1,np.shape(x))
		# d must be > 1!!
		l = 2*np.log(d)
		m = np.exp(l*(u-.5))
		new_x = x * m
		hastings_ratio=sum([np.log(m)])
		return new_x, hastings_ratio

	def normal_pandas(self, initial_mu_value, initial_sigma_value, nIterations, windowsize, a_mu, b_mu, alpha_sigma, beta_sigma):
		with open(self.outputfile, "w") as f:

			f.write("iter\t posterior\t likelihood\t prior\t mu\t sigma\n")
			current_mu = initial_mu_value
			current_sigma = initial_sigma_value
			current_prior_mu = pdf.log_pdf_uniform(current_mu,a_mu,b_mu)
			current_prior_sigma = pdf.log_pdf_gamma(current_sigma,alpha_sigma,beta_sigma)
			current_likelihood = np.sum(pdf.log_pdf_normal(pandas_bm, current_mu, current_sigma))
			current_posterior_ratio = 1

			for iteration in range(nIterations):
				print('iteration number {0}'.format(iteration))

				# 1: propose new values for our parameters
				## in this example we are pretending there is no hastings ratio put out by our function 
				new_mu, log_h_ratio_mu = self.sliding_window(current_mu, windowsize)
				new_sigma, log_h_ratio_sigma = self.sliding_window(current_sigma, windowsize)
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
				new_posterior_ratio = new_likelihood + new_prior_mu + new_prior_sigma -(current_likelihood + current_prior_mu + current_prior_sigma)

				#print("posterior = {0}".format(posterior_ratio))

				# 5: compare new posterior with previous one and decide if we accept the new parameter values or not
				## we use our posterior ratio as our acceptance probability
				acceptance_probability = np.exp(new_posterior_ratio)
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
					current_posterior_ratio = new_posterior_ratio
				else:
					print("rejected")
				f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(iteration, current_posterior_ratio, current_likelihood, current_prior_mu + current_prior_sigma, current_mu, current_sigma))
				print("writing output...")

	def white_noise_linear_trend_climate(self, initial_mu0, initial_sigma0, initial_slope, nIterations, windowsize, a_mu, b_mu, alpha_sigma, beta_sigma, mu_slope, sigma_slope):
		with open(self.outputfile, "w") as f:
			time = np.array(climate.year.tolist())
			temperature = np.array(climate.temperature.tolist())

			f.write("iter\t posterior\t likelihood\t prior\t mu0\t sigma0\t slope\n")
			current_mu0 = initial_mu0
			current_sigma0 = initial_sigma0
			current_slope = initial_slope
			current_mu_list = current_mu0 + current_slope * time
			current_prior_mu0 = pdf.log_pdf_uniform(current_mu0, a_mu, b_mu)
			current_prior_sigma0 = pdf.log_pdf_gamma(current_sigma0, alpha_sigma, beta_sigma)
			current_prior_slope = pdf.log_pdf_normal(current_slope, mu_slope, sigma_slope)
			current_likelihood = np.sum(pdf.log_pdf_normal(temperature, current_mu_list, current_sigma0))
			current_posterior_ratio = 1

			for iteration in range(nIterations):
				print('iteration number {0}'.format(iteration))

				# 1: propose new values for our parameters
				## in this example we are pretending there is no hastings ratio put out by our function 
				new_mu0, log_h_ratio_mu0 = self.sliding_window(current_mu0, windowsize)
				new_sigma0, log_h_ratio_sigma0 = self.sliding_window(current_sigma0, windowsize)
				new_slope, log_h_ratio_slope = self.sliding_window(current_slope, windowsize)
				## Make sure the new value is positive
				#new_mu0 = abs(new_mu0)
				new_sigma0 = abs(new_sigma0) # here, only the sigma parameter needs to be positive

				print("mu0 = {0}, sigma0 = {1}, slope = {2}".format(new_mu0, new_sigma0, new_slope))

				# 2: calculate the prior probability of these parameter values
				new_prior_mu0 = pdf.log_pdf_uniform(new_mu0, a_mu, b_mu)
				new_prior_sigma0 = pdf.log_pdf_gamma(new_sigma0, alpha_sigma, beta_sigma)
				new_prior_slope = pdf.log_pdf_normal(new_slope, mu_slope, sigma_slope)

				#print("prior = {0}".format(prior_mu * prior_sigma))

				# 3: calculate the likelihood of the data with these parameter value
				new_mu_list = new_mu0 + new_slope * time
				new_likelihood = np.sum(pdf.log_pdf_normal(temperature, new_mu_list, new_sigma0))

				#print("likelihood = {0}".format(likelihood))

				# 4: calculate the posterior
				new_posterior_ratio = new_likelihood + new_prior_mu0 + new_prior_sigma0 + new_prior_slope -(current_likelihood + current_prior_mu0 + current_prior_sigma0 + current_prior_slope)

				#print("posterior = {0}".format(posterior_ratio))

				# 5: compare new posterior with previous one and decide if we accept the new parameter values or not
				## we use our posterior ratio as our acceptance probability
				acceptance_probability = np.exp(new_posterior_ratio)
				## draw a random number from a uniform distribution between 0 and 1
				random_number = np.random.uniform(0,1)
				print(random_number <= acceptance_probability)
				## if smaller or equal to our acceptance probability -> accept. Otherwise -> reject
				## If accepted -> current_mu = new_mu
				## If rejected -> current_mu = current_mu
				if random_number <= acceptance_probability:
					print("accepted")
					current_mu = new_mu0
					current_sigma = new_sigma0
					current_slope = new_slope
					current_likelihood = new_likelihood
					current_prior_mu0 = new_prior_mu0
					current_prior_sigma0 = new_prior_sigma0
					current_prior_slope = new_prior_slope
					current_posterior_ratio = new_posterior_ratio
				else:
					print("rejected")
				f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(iteration, current_posterior_ratio, current_likelihood, current_prior_mu0 + current_prior_sigma0, current_mu0, current_sigma0, current_slope))
				print("writing output...")