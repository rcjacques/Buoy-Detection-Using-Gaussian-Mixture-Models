'''
@Authors: Rene Jacques, Zachary Zimits
'''

import numpy as np

def EM(x,num_iter,num_gauss):
	'''EM algorithm main function'''

	# initialize parameters
	mu,sigma,mix = init_params(num_gauss)

	# iterate until the iteration limit is reached
	for i in range(num_iter):
		r = E(x,mu,sigma,mix) # expectation step
		mu,sigma,mix = M(x,r,mu,sigma,mix) # maximization step

	return mu,sigma,mix
	
def gauss(x,mu,sigma):
	'''Gaussian equation'''
	return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))

def E(x,mu,sigma,mix):
	'''Expectation'''
	r = np.zeros([len(x),len(mu)]) # initialize responsabilities 

	# for every item in the data list
	for i in range(len(x)):
		# for each of the desired number of gaussians
		for k in range(len(mu)):
			den = 0
			# for each of the desired number of gaussians
			for j in range(len(mu)):
				den += mix[j]*gauss(x[i],mu[j],sigma[j]) # sum the denominator of the equation
			r[i,k] = (mix[k]*gauss(x[i],mu[k],sigma[k]))/den # calculate responsabilities

	return r

def M(x,r,mu,sigma,mix):
	'''Maximization'''
	# for each of the responsability groups
	for k in range(len(r[1])):
		Nc = np.sum(r[:,k]) # sum the responsabilities corresponding to k
		mu[k] = mu_calc(x,r[:,k],Nc) # update the mean corresponding to k
		sigma[k] = sigma_calc(x,mu[k],r[:,k],Nc) # update the variance corresponding to k
		mix[k] = mix_calc(Nc,len(x)) # update the mixture coefficient corresponding to k

	return mu,sigma,mix

def init_params(num_gauss):
	'''Initialize EM parameters'''
	mus = np.zeros(num_gauss)
	sigmas = np.zeros(num_gauss)
	mix = np.zeros(num_gauss)

	for i in range(num_gauss):
		mus[i] = (i+1)*75
		sigmas[i] = (i+1)*50
		mix[i] = 0.5

	return mus,sigmas,mix

def mu_calc(x,r,Nc):
	'''Mean calculation'''
	return (1/Nc)*np.sum(r*x)

def sigma_calc(x,mu,r,Nc):
	'''Variance calculation'''
	return np.sqrt((1/Nc)*np.sum(r*(x-mu)**2))

def mix_calc(Nc,n):
	'''Mixture coefficient calculation'''
	return Nc/n