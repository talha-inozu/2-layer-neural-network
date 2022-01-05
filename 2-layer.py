import numpy as np
import h5py
import matplotlib.pyplot as plt


def initialize_parameters(n_x, n_h, n_y):

    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x)*0.02
    W2 = np.random.randn(n_y, n_h)*0.02
    b1 = np.zeros((n_h,1))
    b2 = np.zeros((n_y,1))
    
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
    
    return parameters 

def sigmoid(Z):

    sigmaZ = 1/(1+np.exp(-Z))
    cache = Z
    
    return sigmaZ, cache

def relu(Z):
    reluZ = np.maximum(0,Z)
    
    assert(reluZ.shape == Z.shape)
    
    cache = Z 
    return reluZ, cache

def forward_activation(A_prev, W, b, x):
  
    if x == 1:
        Z = np.dot(W, A_prev) + b
        linear_cache = (A_prev,W,b)
        A, activation_cache = sigmoid(Z)
    
    elif x == 0:
        Z = np.dot(W, A_prev) + b
        linear_cache = (A_prev,W,b)
        A, activation_cache = relu(Z)
        
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def linear_activation_forward(X, parameters):

    caches = []
     
    A, cache = forward_activation(X, parameters['W1'], parameters['b1'], 0)
    caches.append(cache)
        
    AL, cache = forward_activation(A, parameters['W2'], parameters['b2'], 1)
    caches.append(cache)
   
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches

def compute_cost(AL, y):
    m = y.shape[1]
    cost = (-1/m) * (np.dot(y,np.log(AL).T) + np.dot(1-y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)      
    assert(cost.shape == ())
    
    return cost


def activation_backward(dA_prev, cache, x):

    linear_cache, activation_cache = cache
    
    if x == 0:
        
        dZ = np.array(dA_prev, copy=True)
        dZ[activation_cache <= 0] = 0
        assert (dZ.shape == activation_cache.shape)

        A_prev, W, b = linear_cache
        m = A_prev.shape[1]
        dW = 1./m*np.dot(dZ, A_prev.T)
        db = 1./m*np.sum(dZ, axis = 1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
       
        
    elif x == 1:
        s = 1/(1+np.exp(-activation_cache))
        dZ = dA_prev * s * (1-s)
        assert (dZ.shape == activation_cache.shape)
        A_prev, W, b = linear_cache
        m = A_prev.shape[1]
        dW = 1./m*np.dot(dZ, A_prev.T)
        db = 1./m*np.sum(dZ, axis = 1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
    return dA_prev, dW, db


def linear_activation_backward(AL, Y, caches):

    grads = {}
    Y = Y.reshape(AL.shape)
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
   
    current_cache = caches[1]
    grads["dA2"], grads["dW2"], grads["db2"] = activation_backward(dAL, current_cache, 1)

    
    current_cache = caches[0]
    dA1, dW1, db1 = activation_backward(grads["dA2"],  current_cache, 0)
    grads["dA1"] = dA1
    grads["dW1"] = dW1
    grads["db1"] = db1
        

    return grads


def update_parameters(parameters, grads, learning_rate):

    parameters["W1"] =parameters["W1"]  - learning_rate * grads["dW1"]
    parameters["b1"] =parameters["b1"]  - learning_rate * grads["db1"]
    parameters["W2"] =parameters["W2"]  - learning_rate * grads["dW2"]
    parameters["b2"] =parameters["b2"]  - learning_rate * grads["db2"]

    return parameters

def predict(X,Y,parameters):
    m = X.shape[1]
    prediction = np.zeros((1,m))
    A, cache0 = forward_activation(X,parameters["W1"],parameters["b1"],0)
    AL,cache1 = forward_activation(A,parameters["W2"],parameters["b2"],1)
    for i in range(AL.shape[1]):
        prediction[0, i] = 1 if AL[0, i] > 0.5 else 0
        assert(prediction.shape == (1, m))
    
    accuracy = 100 - np.mean(np.abs(prediction - Y))*100
    return accuracy


def plot_loss(costs) :
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.show()


def two_layer_model(X, Y, layers_dims, learning_rate, num_iterations):
    np.random.seed(1)

    n_x,n_h,n_y= layers_dims
    parameters = initialize_parameters(n_x,n_h,n_y)

    cost_list = []

    for i in range(num_iterations):
        AL, caches = linear_activation_forward(X, parameters)
        cost = compute_cost(AL, Y)
        cost_list.append(cost) 
        grads = linear_activation_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)


    return parameters,cost_list


def load_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) 
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 

    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig




def main():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y= load_dataset()
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T 
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_set_x = train_set_x_flatten/255
    test_set_x = test_set_x_flatten/255
    n_x = 12288
    n_h = 7
    n_y = 1
    layers_dims = (n_x,n_h,n_y)

    parameters,costs = two_layer_model(train_set_x,train_set_y,layers_dims,0.05,1000)
    print("Test accuracy = % {} ".format(predict(test_set_x,test_set_y,parameters)))
    plot_loss(costs)

main()