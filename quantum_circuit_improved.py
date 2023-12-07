import pennylane as qml
from pennylane import numpy as np
dev = qml.device('lightning.qubit', wires=1)
def S(x,theta):
    """Data-encoding circuit block."""
    '''for i in range(len(x)-1): #WE CHANGED THIS WATCH OUT
        qml.RZ(x[i] + theta[1], wires=0)'''
    qml.RZ(x+theta,wires=0)


def W(theta):
    """Trainable circuit block."""
    qml.RY(theta[0], wires=0)

def single_model(x,theta):
    for i,x_ in enumerate(x):
        W(theta[i])
        S(x_)
    W(theta[len(x)])
    
@qml.qnode(dev, interface="adjoint")
def serial_model(weights, x):
    for theta in weights[:-1]:  #FOR THETA IN ALL THE WEIGHTS UP TO (BUT NOT INCLUDING) THE LAST ONE 
        #print(theta)    
        W(theta)
        S(x,theta)

    W(weights[-1]) #Now this is the last W gate as the input is the last element in the weight array.
    #THIS BELOW IS JUST TO TEST IF ITS CLASSIFYING...
    #qml.RY(np.pi/2,wires=0) 
    #qml.RZ(np.pi*(1-x[8]),wires=0)    
    #qml.RY(np.pi/2,wires=0)
    
    return qml.expval(qml.PauliZ(wires=0)) #Expectation value of the what??
