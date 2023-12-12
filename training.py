import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
import time

def pulsar_probability(sampled_data, weights, quantum_circuit):
    global expectation_value
    expectation_value = [quantum_circuit.serial_model(weights, x_) for x_ in sampled_data]
    probability_pulsar = (1 - np.array(expectation_value)) / 2
    return probability_pulsar

#Loss/Cost functions

def cross_entropy_loss(weights, data, quantum_circuit):
    #predictions = [quantum_circuit.serial_model(weights, x) for x in data]
    predictions = np.array([quantum_circuit.serial_model(weights, x) for x in data[:,:8]])
    predictions = (1-predictions)/2
    targets = np.array([row[-1] for row in data])
    targets = targets.reshape(predictions.shape)
    # Clip predicted probabilities to avoid log(0) issues
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    # Compute the cross-entropy loss
    loss = -np.sum(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    loss = loss / len(targets)
    return loss

def square_loss(weights, data,quantum_circuit):
    loss = 0
    predictions = [quantum_circuit.serial_model(weights, x) for x in data]
    predictions = np.array(predictions)
    predictions = (1-predictions)/2
    targets = np.array([row[-1] for row in data])
    for t, p in zip(targets, predictions):
        loss += (t - p) ** 2
    loss = loss / len(targets)
    return 0.5*loss #Normalizing distance

# Machine Learning Section. 
#Optimization using built-in PennyLane optimizer (Adam). stepsize=0.1: This parameter sets the learning rate for the Adam optimizer. The learning rate controls the step size during optimization and affects how quickly the model's parameters are updated
import time 

def training(epochs, initial_weights, sampled_train_data, loss_func_choice,quantum_circuit,set_number): #loss_func_choice 0 if we want square loss 1 if we want 
    optimizer = qml.AdamOptimizer(stepsize=0.1)
    loss_array = np.array([])
    q = 0
    # Define the loss function based on choice
    if loss_func_choice == 0:
        loss_function = square_loss
    else:
        loss_function = cross_entropy_loss
    
    start_time = time.perf_counter()
    for epoch in range(epochs):
        optimized_weights, loss= optimizer.step_and_cost(lambda w: loss_function(w, sampled_train_data, quantum_circuit), initial_weights)
        initial_weights = optimized_weights
        loss_array = np.append(loss_array, loss)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss:.8f}")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("Elapsed time (Set Number = {0}): {1}".format(set_number, elapsed_time))
    return optimized_weights,loss_array

