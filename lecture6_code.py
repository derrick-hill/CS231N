import numpy as np

#initialise random inputs
#500 input values
#1000 whats?
D = np.random.randn(1000, 500)

#this array has one element for each layer (10) and contains the numner of neurons
#at each layer.
hidden_layer_sizes = [500] * 10


#this allows us to say which function to use at each layer
#use tanh non-liniarities for every layer
nonlinearities = ['tanh']*len(hidden_layer_sizes)

#this allows us to pick nonliniarities
act = {'relu':lambda x:np.maximum(0,x), 
       'tanh':lambda x:np .tanh(x)}


Hs = {}

for i in range(len(hidden_layer_sizes)):
    X = D if i == 0 else Hs[i-1] 
    fan_in = X.shape[1]
    fan_out = hidden_layer_sizes[i]
    W = np.random.rand(fan_in, fan_out) * 0.01  # use small random values to initialise the weights
    
    Htemp = np.dot(X, W)
    H = act[nonlinearities[i]](Htemp)
    Hs[i] = H
    
print('input layer has a mean of %f and std of %f' % (np.mean(D), np.std(D)) )

layer_means = [np.mean(H) for i,H in Hs.items()]
layer_stds = [np.std(H) for i,H in Hs.items()]
for i,H in Hs.items():
    print("Hidden layer %d has mean of %f and std of %f" % (i+1, layer_means[i], layer_stds[i]))


    
    