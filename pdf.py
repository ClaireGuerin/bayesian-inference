import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.special import gamma
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

euler = m.e
euler_log = np.log(m.e)

def pdf_exp(x, lbd):
	p = lbd * np.power(euler, (- lbd * x))
	return p

pandas_bm = np.array([84.74, 84.88, 94.60, 96.37, 102.93, 109.11, 125.76])

#print(pdf_exp(84.74,0.1))
#print(pdf_exp(pandas_bm,0.1))

def log_pdf_exp(x, lbd):
	p_log = np.log(lbd) - lbd * x * euler_log
	return p_log

#print(log_pdf_exp(84.74,0.1))
#print(log_pdf_exp(pandas_bm,0.1))

lambdas = np.arange(0.001,0.1,0.0001)
log_likelihood = []

for par in lambdas:
	point_likelihood = log_pdf_exp(pandas_bm, par)
	data_likelihood = np.sum(point_likelihood) # sum all probabilities because of the log transformation ( log(product) )
	log_likelihood.append(data_likelihood)

highest_likelihood = max(log_likelihood)
most_likely_lambda = lambdas[log_likelihood.index(max(log_likelihood))]

plt.plot(lambdas,log_likelihood)
plt.plot(most_likely_lambda, highest_likelihood, 'ro')
#plt.show()

def log_pdf_uniform(x, a, b):
	data = np.array(x)
	in_range = a < data < b
	data[not in_range] = - np.inf
	data[in_range] = -np.log(b - a)

	return data

#print(log_pdf_uniform(84.74,10,1000))

def log_pdf_normal(x, mu, sigma):
	x = np.array(x)
	to_square = x - mu
	sigma_square = 2 * sigma * sigma
	p = - 0.5 * np.log(np.pi * sigma_square) - (to_square * to_square) / sigma_square

	return p

#print(log_pdf_normal(pandas_bm,100,5))

def log_pdf_gamma(x, alpha, beta):
	x = np.array(x)
	p = alpha * np.log(beta) - np.log(gamma(alpha)) + (alpha - 1) * np.log(x) - beta * x * euler_log

	return p

#print(log_pdf_gamma(pandas_bm,2,0.1))

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
        z_2d[i,j] = np.sum(log_pdf_normal(pandas_bm, x, y))

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
        likelihood = np.sum(log_pdf_normal(pandas_bm, mu, sigma))
        prior_mu = log_pdf_uniform(mu,10,1000)
        prior_sigma = log_pdf_gamma(sigma,2,2)
        posterior = likelihood + prior_mu + prior_sigma
        z_2dPrior[i,j] = posterior

fig = go.Figure(data=[go.Surface(x=mean_array, y=std_array, z=z_2dPrior.T)])
fig.show()