import pennylane as qml
from pennylane import numpy as np
dev = qml.device('lightning.qubit', wires=1)
def S(x):
    """Data-encoding circuit block."""
    qml.RZ(x, wires=0)
    '''for i in range(len(x)-1): #WE CHANGED THIS WATCH OUT
        qml.RZ(x[i], wires=0)'''
        
def single_block(x,theta):
    for i, x_ in enumerate(x):
        W(theta[i])
        S(x_)
    W(theta[len(x)])
    
def multi_block(x,theta,depth):
    for i in range(depth):
        single_block(x, theta[i])
        
def W(theta):
    """Trainable circuit block."""
    qml.RZ(theta[0], wires=0)
    qml.RX(theta[1], wires=0)
    qml.RY(theta[2], wires=0)
 
@qml.qnode(dev, diff_method = "adjoint")
def serial_model(weights, x):
    depth = 3
    qml.Hadamard(wires=0)
    '''for theta in weights[:-1]:  #FOR THETA IN ALL THE WEIGHTS UP TO (BUT NOT INCLUDING) THE LAST ONE 
        W(theta)
        S(x)
    # (L+1)'th unitary.
    W(weights[-1]) #Now this is the last W gate as the input is the last element in the weight array.'''
    multi_block(x, weights, depth)
    return qml.expval(qml.PauliZ(wires=0)) #Expectation value of the what??
