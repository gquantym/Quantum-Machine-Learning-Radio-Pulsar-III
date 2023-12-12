import os
os.environ["OMP_NUM_THREADS"] = "1"

from pennylane import numpy as np

import time
import training
import data
import plot
import testing as t


def errors(weights_crossEntropy,sampled_pulsars,sampled_non_pulsars,quantum_circuit,cross_entropy_choice,threshold):
    global specificity,sensitivity,prob_pulsar_test,specificity_sensitivity_acc,specificity,sensitivity
    
    print("test_pulsar_samples.size == ",test_pulsar_samples.shape)
    print("weights_crossEntropy.size == ",weights_crossEntropy.shape)
    
    prob_pulsar = [training.pulsar_probability(sampled_pulsars[i,:,:8],weights_crossEntropy[i],quantum_circuit) for i in range(num_sets)]
    prob_non_pulsar = [training.pulsar_probability(sampled_non_pulsars[i,:,:8],weights_crossEntropy[i],quantum_circuit) for i in range(num_sets)]
    probabilities = np.concatenate((prob_pulsar, prob_non_pulsar), axis = 1)
    classifications = np.concatenate((sampled_pulsars[:,:, 8],sampled_non_pulsars[:,:, 8]),axis=1)
    
    
    specificity_sensitivity_acc = [t.calculate_specificity_sensitivity(classifications[i], probabilities[i], threshold) for i in range(num_sets)]
    specificity,sensitivity,acc = zip(*specificity_sensitivity_acc)
    
    
    print(f'Specificity: {specificity}')
    print(f'Sensitivity: {sensitivity}')
    for i in range(num_sets):
        plot.test_probabilities(prob_pulsar[i],prob_non_pulsar[i],epochs,cross_entropy_choice,threshold,sensitivity[i],specificity[i],i)


def threshold_calculator(probabilities, true_labels):
    thresholds = np.linspace(0, 1, 100)
    specificities = []
    sensitivities = []
    train_accuracies = []
    
    for threshold in thresholds:
        #predicted_labels = (probabilities >= threshold).astype(int)
        #cm = confusion_matrix(true_labels, predicted_labels)
        specificity, sensitivity,train_acc = t.calculate_specificity_sensitivity(true_labels, probabilities, threshold)
        specificities.append(specificity)
        sensitivities.append(sensitivity)
        train_accuracies.append(train_acc)
    
    opt_threshold_index = np.argmax(specificities)
    opt_threshold_spec = thresholds[opt_threshold_index]
    opt_accuracy = train_accuracies[opt_threshold_index]
    
    opt_threshold_index_acc = np.argmax(train_accuracies)
    opt_threshold_acc = thresholds[opt_threshold_index_acc]
    opt_accuracy_acc = train_accuracies[opt_threshold_index_acc]
    
    return opt_threshold_spec,opt_threshold_acc,opt_accuracy,opt_accuracy_acc



    
def main(initial_weights,epochs,pulsar_samples,non_pulsar_samples,test_pulsar_samples,test_non_pulsar_samples,quantum_circuit):
    global weights_crossEntropy,loss_crossEntropy,probability_non_pulsar,probability_pulsar,train_data,opt_prob_non_pulsar,classifications,probabilities,prob_pulsar_test,prob_non_pulsar_test,threshold
    '''Initializing constants'''
    square_loss_choice = 0
    cross_entropy_choice = 1
    '''Initializing constants'''
    probability_non_pulsar = [training.pulsar_probability(non_pulsar_samples[i,:,:8],initial_weights,quantum_circuit) for i in range(num_sets)]
    probability_pulsar = [training.pulsar_probability(pulsar_samples[i,:,:8],initial_weights,quantum_circuit) for i in range(num_sets)]
    for i in range(num_sets):
        plot.probabilities(probability_pulsar[i],probability_non_pulsar[i],i)
    
    #----------------------------NOW LETS TRAIN DATA----------------------------
    train_data = np.concatenate((pulsar_samples, non_pulsar_samples), axis=1)

    
    weights_loss_values = [training.training(epochs, initial_weights, train_data[i], cross_entropy_choice,quantum_circuit,i) for i in range(num_sets)]
    weights_crossEntropy, loss_crossEntropy = zip(*weights_loss_values)
    
    weights_crossEntropy = np.array(weights_crossEntropy)
    loss_crossEntropy = np.array(loss_crossEntropy)
    
    '''# Using numpy.stack to combine weights_crossEntropy into a single 4D array
    weights_crossEntropy = np.stack(weights_crossEntropy_tuple, axis=3)
    
    # Converting loss_crossEntropy to a NumPy array
    loss_crossEntropy = np.array(loss_crossEntropy_tuple)'''
    
    opt_prob_non_pulsar = [training.pulsar_probability(non_pulsar_samples[i,:,:8],weights_crossEntropy[i],quantum_circuit)for i in range(num_sets)]
    opt_prob_pulsar = [training.pulsar_probability(pulsar_samples[i,:,:8],weights_crossEntropy[i],quantum_circuit) for i in range(num_sets)]
    
    train_classification = [train_data[i,:,8] for i in range(num_sets)]
    opt_probabilities = np.concatenate((opt_prob_pulsar, opt_prob_non_pulsar), axis=1)
    optimal_threshold_spec_optimal_threshold_acc = [threshold_calculator(opt_probabilities[i], train_classification[i]) for i in range(num_sets)]
    global opt_threshold_spec,opt_threshold_acc,opt_accuracy,opt_accuracy_acc
    opt_threshold_spec,opt_threshold_acc,opt_spec_acc,opt_acc_acc = zip(*optimal_threshold_spec_optimal_threshold_acc)
    

    global spec_threshold_mean,acc_threshold_mean,spec_acc_mean,acc_acc_mean
    spec_threshold_mean = np.mean(opt_threshold_spec)
    acc_threshold_mean = np.mean(opt_threshold_acc)
    spec_acc_mean = np.mean(opt_spec_acc)
    acc_acc_mean = np.mean(opt_acc_acc)
    
    
    for i in range(num_sets):
        plot.optimized_probabilities(opt_prob_non_pulsar[i],opt_prob_pulsar[i],epochs,cross_entropy_choice,i,opt_threshold_spec[i],opt_threshold_acc[i])
        plot.loss_function(epochs,loss_crossEntropy[i],cross_entropy_choice,i)
    
    #threshold = 0.9
    
    #for i in range(num_sets):
    #----------------------------NOW LETS TEST DATA----------------------------
    #errors(weights_crossEntropy,test_pulsar_samples,test_non_pulsar_samples,quantum_circuit,cross_entropy_choice,threshold)


    
num_weights = 1
initial_weights = 2 * np.pi * np.random.random(size=(3,9,3))
epochs = 150
normalized_dataset = data.normalize()
num_sets = 1
test_size = 1000
train_size = 100
    
pulsar_samples,non_pulsar_samples,test_pulsar_samples,test_non_pulsar_samples = data.sample_pulsars(normalized_dataset, train_size, test_size, num_sets)#data.sample_pulsars(normalized_dataset,train_size=100,test_size=1000)

import old_quantum_circuit as quantum_circuit
main(initial_weights,epochs,pulsar_samples,non_pulsar_samples,test_pulsar_samples,test_non_pulsar_samples,quantum_circuit)
