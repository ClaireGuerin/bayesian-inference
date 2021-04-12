import numpy as np
import math as m
import matplotlib.pyplot as plt

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
plt.show()

def log_pdf_uniform(x, a, b):
	if x < a or x > b:
		p = 0
	else:
		p = 1 / (b - a)

def log_pdf_normal():
	pass

def log_pdf_gamma():
	pass