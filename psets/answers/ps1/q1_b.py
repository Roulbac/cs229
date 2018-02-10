import numpy as np


## Load training data for X and Y ##
x = np.loadtxt('logistic_x.txt')
x = np.hstack([x, np.ones((len(x), 1))])
y = np.loadtxt('logistic_y.txt', ndmin=2)

# Initialize Theta as the nil vector
theta = np.zeros((len(x[0]), 1))

# x: 3x1 vector, y: scalar, theta: 3x1 vector
g_prime = lambda x, y: g(x,y) * (1 - g(x,y))
g = lambda x,y: 1/(1 + np.exp(-y * (theta.T).dot(x)) )


def grad_inner_term(tup):
    return (1 - g(tup[0], tup[1])) * tup[1] * tup[0]

def h_inner_term(tup) : 
    return g_prime(tup[0],tup[1]) * np.outer(tup[0], tup[0])

def compute_h():
    h_inner_terms = np.apply_along_axis(h_inner_term, 1, list(zip(x,y)) )
    h = 1/len(x) * h_inner_terms.sum(0)
    return h

# Computing gradient
def compute_gradient():
    gradient_inner_terms = np.apply_along_axis(grad_inner_term, 1, list(zip(x,y)))
    gradient = -1/len(x) * gradient_inner_terms.sum(0, keepdims=True)
    return gradient.T
# Newton loop

h = compute_h()
gradient = compute_gradient()
while( np.linalg.norm(gradient) > 0.0001 ):
    theta -= np.linalg.inv(h).dot(gradient)
    h = compute_h()
    gradient = compute_gradient()
print( theta )
    

