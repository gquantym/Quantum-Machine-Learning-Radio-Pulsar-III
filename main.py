import os
os.environ["OMP_NUM_THREADS"] = "1"
from pennylane import numpy as np
import time
import training
import data
import plot
import testing as t
def tezt(weights_crossEntropy,test_pulsar_samples,test_non_pulsar_samples,quantum_circuit,cross_entropy_choice):
    prob_pulsar_test = training.pulsar_probability(test_pulsar_samples[:,:8],weights_crossEntropy,quantum_circuit)
    prob_non_pulsar_test = training.pulsar_probability(test_non_pulsar_samples[:,:8],weights_crossEntropy,quantum_circuit)
    probabilities = np.concatenate((prob_pulsar_test, prob_non_pulsar_test))
    classifications = np.vstack((test_pulsar_samples,test_non_pulsar_samples))[:, 8]
    #global optimal_threshold, max_specificity, corresponding_sensitivity, thresholds, specificities, sensitivities
    optimal_threshold, max_specificity, corresponding_sensitivity, thresholds, specificities, sensitivities = t.find_optimal_threshold(probabilities, classifications)
    #t.plot_specificity_sensitivity_vs_threshold(thresholds, specificities, sensitivities, optimal_threshold, max_specificity, corresponding_sensitivity)
    print(f'Optimal Specificity Threshold: {optimal_threshold}')
    print(f'Max Specificity: {max_specificity}')
    print(f'Corresponding Sensitivity: {corresponding_sensitivity}')
    plot.test_probabilities(prob_pulsar_test,prob_non_pulsar_test,epochs,cross_entropy_choice,optimal_threshold,max_specificity,corresponding_sensitivity)
    return [max_specificity, corresponding_sensitivity]
    
def main(initial_weights,epochs,pulsar_samples,non_pulsar_samples,test_pulsar_samples,test_non_pulsar_samples,quantum_circuit):
    global weights_crossEntropy,probability_non_pulsar,probability_pulsar,train_data,opt_prob_non_pulsar,classifications,probabilities,prob_pulsar_test,prob_non_pulsar_test,threshold
    '''Initializing constants'''
    square_loss_choice = 0
    cross_entropy_choice = 1
    '''Initializing constants'''
    #Finding the probability of finding pulsars and non_pulsars
    #Note, the pulsar_samples and non_pulsar_samples have their 8th element as the classification
    #non_pulsar_samples = non_pulsar_samples[:,:8]
    #pulsar_samples = pulsar_samples[:,:8]
    probability_non_pulsar = training.pulsar_probability(non_pulsar_samples[:,:8],initial_weights,quantum_circuit)
    probability_pulsar = training.pulsar_probability(pulsar_samples[:,:8],initial_weights,quantum_circuit)
    plot.probabilities(probability_pulsar,probability_non_pulsar)
      
    #----------------------------NOW LETS TRAIN DATA----------------------------
    
    train_data = np.vstack((pulsar_samples, non_pulsar_samples))
    start_time = time.perf_counter()
    weights_crossEntropy,loss_crossEntropy = training.training(epochs, initial_weights, train_data, cross_entropy_choice,quantum_circuit)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)
    
    opt_prob_non_pulsar = training.pulsar_probability(non_pulsar_samples[:,:8],weights_crossEntropy,quantum_circuit)
    opt_prob_pulsar = training.pulsar_probability(pulsar_samples[:,:8],weights_crossEntropy,quantum_circuit)
    plot.optimized_probabilities(opt_prob_non_pulsar,opt_prob_pulsar,epochs,cross_entropy_choice)
    plot.loss_function(epochs,loss_crossEntropy,cross_entropy_choice)
    
    #----------------------------NOW LETS TEST DATA----------------------------
    '''
    thresholds = [tezt(weights_crossEntropy,test_pulsar_samples[i],test_non_pulsar_samples[i],quantum_circuit,cross_entropy_choice) for i in range(5)]
    mean_threshold = np.mean(thresholds)
    '''
    threshold = 0.9
    errors = [tezt(weights_crossEntropy[i],test_pulsar_samples[i],test_non_pulsar_samples[i],quantum_circuit,cross_entropy_choice) for i in range(5)]
    
    
    
    

initial_weights = 2 * np.pi * np.random.random(size=(3,9, 2))
epochs = 40

normalized_dataset = data.normalize()
    
pulsar_samples,non_pulsar_samples,test_pulsar_samples,test_non_pulsar_samples = data.sample_pulsars(normalized_dataset, train_size=100, test_size=1000, num_test_sets=5)#data.sample_pulsars(normalized_dataset,train_size=100,test_size=1000)

import toms_quantum_model as quantum_circuit
main(initial_weights,epochs,pulsar_samples,non_pulsar_samples,test_pulsar_samples,test_non_pulsar_samples,quantum_circuit)
