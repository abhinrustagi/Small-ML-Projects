from scipy.special import xlogy
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0) 

def relu(x):
    return x*(x>0)
    
def tanh(x):
    return (np.exp(2*x)-1)/(np.exp(2*x)+1)
    
def binaryCrossEntropy(trueOutputs, Predictions):
    error = xlogy(trueOutputs, Predictions+ 1e-15) + xlogy(1 - trueOutputs, 1 - Predictions + 1e-15)
    error = -np.sum(error)
    return error
        
def derivativeActivation(activation, valuesToChange):
    if activation == 'sigmoid':
        return valuesToChange*(1-valuesToChange)
    elif activation == 'tanh':
        return 1 - valuesToChange**2
    elif activation == 'relu':
        return np.ones(valuesToChange.shape)*(valuesToChange>=0)

class neural_network:
    layers = 0
    parameters = {}
    activations = {'sigmoid': sigmoid, 'relu': relu, 'softmax': softmax, 'tanh': tanh}
    
    def showParams(self, node=0):
        if node == 0:
            for i in range(self.layers):
                temp = self.parameters['layer'+str(i+1)]
                print("\nLayer "+str(i+1)+":\t"+str(temp['noOfNodes'])+ " Nodes")
                print("Activation Function: ",temp['activation'])
                print("Weights = ",temp['weights'])
                print("Bias = ", temp['bias'])
        else:
            temp = self.parameters['layer'+str(node)]
            print("\nLayer "+str(node)+":\t"+str(temp['noOfNodes'])+ " Nodes")
            print("Activation Function: ",temp['activation'])
            print("Weights = ",temp['weights'])
            print("Bias = ", temp['bias'])
            
    def info(self):
        print("Number of Layers:\t", self.layers)
        sumParams = 0
        for i in range(self.layers):
            temp = self.parameters['layer'+str(i+1)]
            print("\nLayer "+str(i+1)+":\t"+str(temp['noOfNodes'])+ " Nodes")
            print("Activation Function: ",temp['activation'])
            sumParams += temp['weights'].shape[0]*temp['weights'].shape[1]
            sumParams += temp['bias'].shape[0]*temp['bias'].shape[1]
        print("\nTotal number of parameters: ", sumParams)
    
    def predict(self, inputValues):
        cache = {}
        for i in range(self.layers):
            layer = self.parameters['layer'+str(i+1)]
            
            if i == 0:
                z = layer['weights'].dot(inputValues) + layer['bias']
                a = self.activations[layer['activation']](z)
                cache['layer'+str(i+1)] = {'z':z, 'a':a}
            else:
                z = layer['weights'].dot(cache['layer'+str(i)]['a']) + layer['bias']
                a = self.activations[layer['activation']](z)
                cache['layer'+str(i+1)] = {'z':z, 'a':a}
            
        return cache['layer'+str(self.layers)]['a']
    
    def forward(self, inputValues):
        cache = {}
        for i in range(self.layers):
            layer = self.parameters['layer'+str(i+1)]
            
            if i == 0:
                z = layer['weights'].dot(inputValues) + layer['bias']
                a = self.activations[layer['activation']](z)
                cache['layer'+str(i+1)] = {'z':z, 'a':a}
            else:
                z = layer['weights'].dot(cache['layer'+str(i)]['a']) + layer['bias']
                a = self.activations[layer['activation']](z)
                cache['layer'+str(i+1)] = {'z':z, 'a':a}
            
        return cache['layer'+str(self.layers)]['a'], cache
    
    def removeLayer(self):
        value = self.parameters.pop('layer'+str(self.layers))
        print("Removed Layer "+str(self.layers)+ ' with values : ', value)
        self.layers -= 1
    

    def addLayer(self, noOfNodes, activationFunction, **kwargs):
        self.layers += 1
        
        if self.layers == 1:
            if len(kwargs.items()) != 0:
                for key, value in kwargs.items():
                    if key == 'inputSize':
                        self.parameters['layer'+str(self.layers)] = {'weights':np.random.randn(noOfNodes, value), 'bias' : np.zeros((noOfNodes,1)), 'activation':activationFunction, 'noOfNodes':noOfNodes}
            else:
                print("ERROR: Cannot Compile Model, Input Size not given\nTry Giving an argument example: inputSize = 5")
                
        else:
           self.parameters['layer'+str(self.layers)] = {'weights':np.random.randn(noOfNodes, self.parameters['layer'+str(self.layers-1)]['noOfNodes']), 'bias' : np.zeros((noOfNodes,1)), 'activation':activationFunction, 'noOfNodes':noOfNodes}
        
        print("Added a layer with "+str(noOfNodes)+" nodes, and "+activationFunction+" Activation Function.")
        
    def backprop(self, inputs, outputs, epochs = 20, learning_rate = 0.2):
        cache = {}
        for i in range(1, epochs+1):
            predictions, forward_cache = self.forward(inputs)
            print("Epoch " + str(i) + " ||")
            loss = binaryCrossEntropy(outputs, predictions)
            print(loss)
            for j in range(self.layers, 0, -1):
                if j == self.layers:
                    da = -outputs/(predictions+ 1e-15) + ((1-outputs)/(1-predictions+ 1e-15))
                    dz = da * derivativeActivation(self.parameters['layer'+str(j)]['activation'], forward_cache['layer'+str(j)]['a'])
                   
                    dw = (dz.dot(forward_cache['layer'+str(j-1)]['a'].T))/inputs.shape[1]
                    db = np.sum(dz, axis = 1, keepdims = True)/inputs.shape[1]
                    da_1 = self.parameters['layer'+str(j)]['weights'].T.dot(dz)
                    cache['layer'+str(j)] = {'da':da, 'dz':dz, 'dw':dw, 'db':db}
                    cache['layer'+str(j-1)] = {'da':da_1}
                else:
                    if j > 1:
                        dz = cache['layer'+str(j)]['da']*derivativeActivation(self.parameters['layer'+str(j)]['activation'], forward_cache['layer'+str(j)]['a'])
                        dw = (dz.dot(forward_cache['layer'+str(j-1)]['a'].T))/inputs.shape[1]
                        db = np.sum(dz, axis = 1, keepdims = True)/inputs.shape[1]
                        da_1 = self.parameters['layer'+str(j)]['weights'].T.dot(dz)
                        cache['layer'+str(j)].update({'dz':dz, 'dw':dw, 'db':db})
                        cache['layer'+str(j-1)] = {'da':da_1}
                    else:
                        dz = cache['layer'+str(j)]['da']*derivativeActivation(self.parameters['layer'+str(j)]['activation'], forward_cache['layer'+str(j)]['a'])
                        dw = (dz.dot(inputs.T))/inputs.shape[1]
                        db = np.sum(dz, axis = 1, keepdims = True)/inputs.shape[1]
                        da_1 = self.parameters['layer'+str(j)]['weights'].T.dot(dz)
                        cache['layer'+str(j)].update({'dz':dz, 'dw':dw, 'db':db})
                    
            for k in range(1, self.layers+1):
                layer = self.parameters['layer'+str(k)]
                layer['weights'] -= learning_rate*cache['layer'+str(k)]['dw']
                layer['bias'] -= learning_rate*cache['layer'+str(k)]['db']
                