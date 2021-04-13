import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

class SurfacePlots(object):

	pandas_bm = np.array([84.74, 84.88, 94.60, 96.37, 102.93, 109.11, 125.76])
	lambdas = np.arange(0.001,0.1,0.0001)
	log_likelihood = []

	for par in lambdas:
		point_likelihood = PDF.log_pdf_exp(pandas_bm, par)
		data_likelihood = np.sum(point_likelihood) # sum all probabilities because of the log transformation ( log(product) )
		log_likelihood.append(data_likelihood)

	highest_likelihood = max(log_likelihood)
	most_likely_lambda = lambdas[log_likelihood.index(max(log_likelihood))]

	plt.plot(lambdas,log_likelihood)
	plt.plot(most_likely_lambda, highest_likelihood, 'ro')
	#plt.show()

	n_steps = 100
	mean_array = np.linspace(84,125,n_steps)
	std_array = np.linspace(6,50,n_steps)

	# create an empty 2D array
	z_2d  = np.zeros((n_steps,n_steps))

	# loop through the indices
	for i in range(n_steps):
	    for j in range(n_steps):
	        x = mean_array[i]
	        y = std_array[j]
	        # fill loglik of param-combo at position row j, column i  
	        z_2d[i,j] = np.sum(PDF.log_pdf_normal(pandas_bm, x, y))

	fig = go.Figure(data=[go.Surface(x=mean_array, y=std_array, z=z_2d.T)])
	fig.show()

	# Get maximum likelihood parameters
	# get indices where 2D likelihood array is at its maximum
	i,j = np.where(z_2d==np.max(z_2d))
	# apply the selected indices to the arrays
	max_lik_value = z_2d[i,j]
	max_mu = mean_array[i]
	max_std = std_array[j]
	# print the values to the screen
	#print("max likelihood:",max_lik_value,"mu:", max_mu, "sig:", max_std)

	# Calculate prior likelihood
	# create an empty 2D array
	z_2dPrior  = np.zeros((n_steps,n_steps))

	# loop through the indices
	for i in range(n_steps):
	    for j in range(n_steps):
	        mu = mean_array[i]
	        sigma = std_array[j]
	        likelihood = np.sum(PDF.log_pdf_normal(pandas_bm, mu, sigma))
	        prior_mu = PDF.log_pdf_uniform(mu,10,1000)
	        prior_sigma = PDF.log_pdf_gamma(sigma,2,2)
	        posterior = likelihood + prior_mu + prior_sigma
	        z_2dPrior[i,j] = posterior

	fig = go.Figure(data=[go.Surface(x=mean_array, y=std_array, z=z_2dPrior.T)])
	fig.show()

	n_steps = 100
	alpha_array = np.linspace(20,80,n_steps)
	betas_array = np.linspace(0.2,1,n_steps)

	# Calculate prior likelihood
	# create an empty 2D array
	z_2dLogLik = np.zeros((n_steps,n_steps))
	z_2dGamma  = np.zeros((n_steps,n_steps))

	# loop through the indices
	for i in range(n_steps):
	    for j in range(n_steps):
	        alphas = alpha_array[i]
	        betas = betas_array[j]
	        likelihood = np.sum(PDF.log_pdf_gamma(pandas_bm, alphas, betas))
	        prior_alpha = PDF.log_pdf_uniform(alphas,0,100)
	        prior_beta = PDF.log_pdf_uniform(betas,0,100)
	        posterior = likelihood + prior_alpha + prior_beta
	        z_2dGamma[i,j] = posterior
	        z_2dLogLik[i,j] = likelihood

	fig = go.Figure(data=[go.Surface(x=alpha_array, y=betas_array, z=np.exp(z_2dGamma.T))])
	fig.show()

	fig = go.Figure(data=[go.Surface(x=alpha_array, y=betas_array, z=z_2dLogLik.T)])
	fig.show()