import pennylane as qml
from pennylane import numpy as np
dev = qml.device('lightning.qubit', wires=1)
def S(x,theta):
    """Data-encoding circuit block."""
    qml.RZ(x + theta[1], wires=0)

        
def single_block(x,theta):
    for i, x_ in enumerate(x):
        W(theta[i])
        S(x_,theta[i])
    W(theta[len(x)])
    
def multi_block(x,theta,depth):
    for i in range(depth):
        single_block(x, theta[i])

def W(theta):
    """Trainable circuit block."""
    qml.RY(theta[0], wires=0)

 
@qml.qnode(dev, diff_method = "adjoint")
def serial_model(weights, x):
    depth = 3
    multi_block(x, weights, depth)
    return qml.expval(qml.PauliZ(wires=0)) 
