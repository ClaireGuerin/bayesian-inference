import numpy as np
import math as m
from scipy.special import gamma

euler = m.e
euler_log = np.log(m.e)

class PDF(object):

	def pdf_exp(x, lbd):
		p = lbd * np.power(euler, (- lbd * x))
		return p

	#print(pdf_exp(84.74,0.1))
	#print(pdf_exp(pandas_bm,0.1))

	def log_pdf_exp(x, lbd):
		p_log = np.log(lbd) - lbd * x * euler_log
		return p_log

	#print(log_pdf_exp(84.74,0.1))
	#print(log_pdf_exp(pandas_bm,0.1))

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