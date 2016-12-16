import nengo
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import scipy.stats as st
from nengo.spa import Vocabulary
from nengo.dists import UniformHypersphere
import cPickle as pickle
import pylab


import nengo.utils.function_space
nengo.dists.Function = nengo.utils.function_space.Function
nengo.FunctionSpace = nengo.utils.function_space.FunctionSpace
import sys


if len(sys.argv) < 1:
    print "Error: Please specify a pickle file name to store output data"


max_age = dim = 120

# our domain is thetas (i.e., age from 1 to 120)
thetas = np.linspace(start=1, stop=max_age, num=max_age)


# likelihood parameters
#x = 35  # observed age

# prior parameters
skew = -4
loc = 97
scale = 28

def likelihood(x):
    like = np.asarray([1/p for p in thetas])
    like[0:x-1] = [0]*np.asarray(x-1)
    return like


def skew_gauss(skew, loc, scale):
    return [(st.skewnorm.pdf(p, a=skew, loc=loc, scale=scale)) for p in thetas] 
    
    
def posterior(x, skew, loc, scale):
    post = likelihood(x=x)*skew_gauss(skew=skew, loc=loc, scale=scale)
    post = post/sum(post)
    return post


try: 
	ages = np.linspace(start=1, stop=100, num=100)
	data = {}
	for x in ages:

		# create spaces
		space = nengo.dists.Function(skew_gauss,
				             skew=nengo.dists.Uniform(skew-1, skew+2), 
				          loc=nengo.dists.Uniform(loc-1,loc+2), 
				          scale=nengo.dists.Uniform(scale-1, scale+2))


		lik_space = nengo.dists.Function(likelihood,
				             x=nengo.dists.Uniform(x-1,x+2))


		post_space = nengo.dists.Function(posterior,
				             x=nengo.dists.Uniform(x-1,x+2),
				            skew=nengo.dists.Uniform(skew-1, skew+2), 
				          loc=nengo.dists.Uniform(loc-1,loc+2), 
				          scale=nengo.dists.Uniform(scale-1, scale+2))



		# Nengo model
		model = nengo.Network(seed=12)
		with model:
		    stim = nengo.Node(skew_gauss(skew=skew, loc=loc, scale=scale))
		    ens = nengo.Ensemble(n_neurons=50, dimensions=dim,
				         encoders=space,
				         eval_points=space,
				        )
		    
		    stim2 = nengo.Node(likelihood(x=x))
		    ens2 = nengo.Ensemble(n_neurons=50, dimensions=dim,
				         encoders=lik_space,
				         eval_points=lik_space,
				        )
		    
		    
		    nengo.Connection(stim, ens)
		    probe_func = nengo.Probe(ens, synapse=0.03)
		    
		    nengo.Connection(stim2, ens2)
		    probe_func2 = nengo.Probe(ens2, synapse=0.03)
		    
		    # elementwise multiplication
		    ens_posterior = nengo.Ensemble(n_neurons=50, dimensions=dim,
				             encoders=post_space,
				             eval_points=post_space,
				            )
		    product = nengo.networks.Product(n_neurons=50*2, dimensions=dim, input_magnitude=1)
		    
		    nengo.Connection(ens, product.A)
		    nengo.Connection(ens2, product.B)
		    nengo.Connection(product.output, ens_posterior)
		    probe_func3 = nengo.Probe(ens_posterior, synapse=0.03)
		    
		    # divisive normalization
		    def normalize(a):
			return a
			total = np.sum(a)
			if total == 0:
			    return 0
			return [x / total for x in a]
		    
		    
		    norm_post = nengo.Ensemble(n_neurons=50, dimensions=dim, 
				               encoders=post_space,
				             eval_points=post_space)
		    
		    nengo.Connection(ens_posterior, norm_post, function=normalize)
		    probe_func4 = nengo.Probe(norm_post, synapse=0.03)
		    
		    # prediction
		    def median(b):
			med = 0
			for n in np.arange(len(b)):
			    cum = sum(b[:n+1])
			    if cum == 0.5 or cum > 0.5:
				med = n + 1
				break
			return int(med)
		    
		    prediction1 = nengo.Ensemble(n_neurons=50, dimensions=1, 
				          encoders=nengo.dists.Uniform(1,121, integer=False), 
				          eval_points=nengo.dists.Uniform(1,121, integer=False))
		    
		    prediction2 = nengo.Node(output=None, size_in=1)
		       
		    nengo.Connection(norm_post, prediction1, function=median)
		    nengo.Connection(norm_post, prediction2, function=median, synapse=0.03)
		    probe_func5 = nengo.Probe(prediction1, synapse=0.03)
		    probe_func6 = nengo.Probe(prediction2, synapse=0.03)
		    
		    
		sim = nengo.Simulator(model)
		sim.run(0.2)

		ens_prediction = sim.data[probe_func5][-1]
		node_prediction = sim.data[probe_func6][-1]
		data[x] = [ens_prediction, node_prediction]
except:
	print "SS - Exception occured"	

finally:
	fname = 'nengo_predictions.p'
	pickle.dump(data, open(fname, 'wb'))
	print("pickle complete")
	print(fname)
