import os
os.environ["OMP_NUM_THREADS"] = "1"

from pennylane import numpy as np

import time
import training
import data
import plot
import testing as t


def errors(weights_crossEntropy,sampled_pulsars,sampled_non_pulsars,quantum_circuit,cross_entropy_choice,spec_threshold,acc_threshold,title,num_sets):
    #global specificity,sensitivity,prob_pulsar_test,specificity_sensitivity_acc,specificity,sensitivity
    #global mean_spec_accuracy,mean_acc_accuracy,std_spec_accuracy,std_acc_accuracy
    
    prob_pulsar = [training.pulsar_probability(sampled_pulsars[i,:,:8],weights_crossEntropy[i],quantum_circuit) for i in range(num_sets)]
    prob_non_pulsar = [training.pulsar_probability(sampled_non_pulsars[i,:,:8],weights_crossEntropy[i],quantum_circuit) for i in range(num_sets)]
    probabilities = np.concatenate((prob_pulsar, prob_non_pulsar), axis = 1)
    classifications = np.concatenate((sampled_pulsars[:,:, 8],sampled_non_pulsars[:,:, 8]),axis=1)
    
    specificity_sensitivity_acc = [t.calculate_specificity_sensitivity(classifications[i], probabilities[i], spec_threshold) for i in range(num_sets)]
    specificity,sensitivity,spec_acc = zip(*specificity_sensitivity_acc)
    
    specificity_sensitivity_acc = [t.calculate_specificity_sensitivity(classifications[i], probabilities[i], acc_threshold) for i in range(num_sets)]
    specificity,sensitivity,acc_acc = zip(*specificity_sensitivity_acc)
    
    mean_spec_accuracy = np.mean(spec_acc)
    mean_acc_accuracy = np.mean(acc_acc)
    std_spec_accuracy = np.std(spec_acc)
    std_acc_accuracy = np.std(acc_acc)
    
    
    #print(f'Specificity: {specificity}')
    #print(f'Sensitivity: {sensitivity}')
    for i in range(num_sets):
        plot.test_probabilities(prob_pulsar[i],prob_non_pulsar[i],epochs,cross_entropy_choice,spec_threshold,acc_threshold,i,title)
        
    return std_spec_accuracy,std_acc_accuracy



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

def sampling_error(mean_threshold_acc,mean_threshold_spec,initial_weights,weights_crossEntropy):
    train_size_error = 100
    test_size_error = 1000
    global pulsar_samples_error,non_pulsar_samples_error
    num_sets_error = 5
    pulsar_samples_error,non_pulsar_samples_error,test_pulsar_samples_error,test_non_pulsar_samples_error = data.sample_pulsars(normalized_dataset, train_size_error, test_size_error, num_sets_error)#data.sample_pulsars(normalized_dataset,train_size=100,test_size=1000)
    
    train_data = np.concatenate((pulsar_samples_error, non_pulsar_samples_error), axis=1)
    
    probability_non_pulsar = [training.pulsar_probability(non_pulsar_samples_error[i,:,:8],initial_weights,quantum_circuit) for i in range(num_sets_error)]
    probability_pulsar = [training.pulsar_probability(pulsar_samples_error[i,:,:8],initial_weights,quantum_circuit) for i in range(num_sets_error)]
    #ASK KAPPA IF HE WANTS TO PLOT THESE ABOVE
    #global array1,array2,array3,array4
    opt_prob_non_pulsar,opt_prob_pulsar,weights_crossEntropy, loss_crossEntropy = trained_probabilities(pulsar_samples_error, non_pulsar_samples_error,initial_weights,quantum_circuit,cross_entropy_choice,train_data,num_sets_error)
    
    spec_sampling_error,acc_sampling_error = errors(weights_crossEntropy,test_pulsar_samples_error,test_non_pulsar_samples_error,quantum_circuit,cross_entropy_choice,mean_threshold_spec,mean_threshold_acc,"Sampling Error",num_sets_error)
    return spec_sampling_error,acc_sampling_error








def initial_probabilites(initial_weights,pulsar_samples,non_pulsar_samples,quantum_circuit,num_sets):
    probability_non_pulsar = [training.pulsar_probability(non_pulsar_samples[i,:,:8],initial_weights,quantum_circuit) for i in range(num_sets)]
    probability_pulsar = [training.pulsar_probability(pulsar_samples[i,:,:8],initial_weights,quantum_circuit) for i in range(num_sets)]
    for i in range(num_sets):
        plot.probabilities(probability_pulsar[i],probability_non_pulsar[i],i)

def trained_probabilities(pulsar_samples, non_pulsar_samples,initial_weights,quantum_circuit,cross_entropy_choice,train_data,num_sets):
    '''all_opt_prob_non_pulsar = []
    all_opt_prob_pulsar = []
    all_weights_crossEntropy = []
    all_loss_crossEntropy = []
    for j in range(num_weights):'''
    weights_loss_values = [training.training(epochs, initial_weights, train_data[i], cross_entropy_choice,quantum_circuit,i) for i in range(num_sets)]
    weights_crossEntropy, loss_crossEntropy = zip(*weights_loss_values)
    
    weights_crossEntropy = np.array(weights_crossEntropy)
    loss_crossEntropy = np.array(loss_crossEntropy)
    
    opt_prob_non_pulsar = [training.pulsar_probability(non_pulsar_samples[i,:,:8],weights_crossEntropy[i],quantum_circuit)for i in range(num_sets)]
    opt_prob_pulsar = [training.pulsar_probability(pulsar_samples[i,:,:8],weights_crossEntropy[i],quantum_circuit) for i in range(num_sets)]
    
    '''all_opt_prob_non_pulsar.append(opt_prob_non_pulsar)
    all_opt_prob_pulsar.append(opt_prob_pulsar)
    all_weights_crossEntropy.append(weights_crossEntropy)
    all_loss_crossEntropy.append(loss_crossEntropy)'''
    
    return opt_prob_non_pulsar,opt_prob_pulsar,weights_crossEntropy, loss_crossEntropy#all_opt_prob_non_pulsar, all_opt_prob_pulsar, all_weights_crossEntropy, all_loss_crossEntropy

def mean_thresholds(opt_prob_pulsar,opt_prob_non_pulsar,train_data,num_sets):
    train_classification = [train_data[i,:,8] for i in range(num_sets)]
    opt_probabilities = np.concatenate((opt_prob_pulsar, opt_prob_non_pulsar), axis=1)
    optimal_threshold_spec_optimal_threshold_acc = [threshold_calculator(opt_probabilities[i], train_classification[i]) for i in range(num_sets)]
    #global opt_threshold_spec,opt_threshold_acc,opt_accuracy,opt_accuracy_acc
    opt_threshold_spec,opt_threshold_acc,opt_spec_acc,opt_acc_acc = zip(*optimal_threshold_spec_optimal_threshold_acc)
    

    #global spec_threshold_mean,acc_threshold_mean,spec_acc_mean,acc_acc_mean
    spec_threshold_mean = np.mean(opt_threshold_spec)
    acc_threshold_mean = np.mean(opt_threshold_acc)
    spec_acc_mean = np.mean(opt_spec_acc)
    acc_acc_mean = np.mean(opt_acc_acc)
    
    std_spec_acc = np.std(opt_spec_acc)
    std_acc_acc = np.std(opt_acc_acc)
    return spec_threshold_mean,acc_threshold_mean,spec_acc_mean,acc_acc_mean,std_spec_acc,std_acc_acc
    #--------------------  PUT THIS INTO MAIN -----------------------------------------------------------------
    '''for i in range(num_sets):
        plot.optimized_probabilities(opt_prob_non_pulsar[i],opt_prob_pulsar[i],epochs,cross_entropy_choice,i,opt_threshold_spec[i],opt_threshold_acc[i])
        plot.loss_function(epochs,loss_crossEntropy[i],cross_entropy_choice,i)'''
    
    


def initialization_error(mean_threshold_acc,mean_threshold_spec):
    #global pulsar_samples_error,non_pulsar_samples_error,initial_weights_error
    train_size_error = 100
    test_size_error = 1000
    num_sets = 1
    num_weights_error = 5
    initial_weights_error = 2 * np.pi * np.random.random(size=(num_weights_error,3,9,3))                              
    pulsar_samples_error,non_pulsar_samples_error,test_pulsar_samples_error,test_non_pulsar_samples_error = data.sample_pulsars(normalized_dataset, train_size_error, test_size_error, num_sets)#data.sample_pulsars(normalized_dataset,train_size=100,test_size=1000)
    
    train_data_error = np.concatenate((pulsar_samples_error, non_pulsar_samples_error), axis=1)
    
    probability_non_pulsar = [training.pulsar_probability(non_pulsar_samples_error[0,:,:8],initial_weights_error[j],quantum_circuit) for j in range(num_weights_error)]
    probability_pulsar = [training.pulsar_probability(pulsar_samples_error[0,:,:8],initial_weights_error[j],quantum_circuit) for j in range(num_weights_error)]
    #ASK KAPPA IF HE WANTS TO PLOT THESE ABOVE
    
    opt_prob_non_pulsar,opt_prob_pulsar,weights_crossEntropy,loss_crossEntropy = initialization_trained_probabilities(pulsar_samples_error, non_pulsar_samples_error,initial_weights_error,quantum_circuit,cross_entropy_choice,train_data_error,num_weights_error)
    
    std_spec_accuracy,std_acc_accuracy = errors_initialization(weights_crossEntropy,test_pulsar_samples_error,test_non_pulsar_samples_error,quantum_circuit,cross_entropy_choice,mean_threshold_spec,mean_threshold_acc,"Initialization Error",num_weights_error)
    return std_spec_accuracy,std_acc_accuracy

def initialization_trained_probabilities(pulsar_samples_error, non_pulsar_samples_error,initial_weights_error,quantum_circuit,cross_entropy_choice,train_data_error,num_weights):
    #global weights_init,loss_init
    '''all_opt_prob_non_pulsar = []
    all_opt_prob_pulsar = []
    all_weights_crossEntropy = []
    all_loss_crossEntropy = []
    for j in range(num_weights):'''
    weights_loss_values = [training.training(epochs, initial_weights_error[j], train_data_error[0], cross_entropy_choice,quantum_circuit,j) for j in range(num_weights)]
    weights_crossEntropy, loss_crossEntropy = zip(*weights_loss_values)
    
    
    weights_crossEntropy = np.array(weights_crossEntropy)
    loss_crossEntropy = np.array(loss_crossEntropy)
    
    weights_init = weights_crossEntropy 
    loss_init = loss_crossEntropy 
    opt_prob_non_pulsar = [training.pulsar_probability(non_pulsar_samples_error[0,:,:8],weights_crossEntropy[j],quantum_circuit)for j in range(num_weights)]
    opt_prob_pulsar = [training.pulsar_probability(pulsar_samples_error[0,:,:8],weights_crossEntropy[j],quantum_circuit) for j in range(num_weights)]
    
    '''all_opt_prob_non_pulsar.append(opt_prob_non_pulsar)
    all_opt_prob_pulsar.append(opt_prob_pulsar)
    all_weights_crossEntropy.append(weights_crossEntropy)
    all_loss_crossEntropy.append(loss_crossEntropy)'''
    
    return opt_prob_non_pulsar,opt_prob_pulsar,weights_crossEntropy, loss_crossEntropy#all_opt_prob_non_pulsar, all_opt_prob_pulsar, all_weights_crossEntropy, all_loss_crossEntropy

def errors_initialization(weights_crossEntropy,sampled_pulsars_test,sampled_non_pulsars_test,quantum_circuit,cross_entropy_choice,spec_threshold,acc_threshold,title,num_weights):

    #global a_pulsar,a_non_pulsar,a_probab,a_class
    
    prob_pulsar_test = [training.pulsar_probability(sampled_pulsars_test[0,:,:8],weights_crossEntropy[j],quantum_circuit) for j in range(num_weights)]
    a_pulsar = prob_pulsar_test
    prob_non_pulsar_test = [training.pulsar_probability(sampled_non_pulsars_test[0,:,:8],weights_crossEntropy[j],quantum_circuit) for j in range(num_weights)]
    a_non_pulsar = prob_non_pulsar_test
    probabilities = np.concatenate((prob_pulsar_test, prob_non_pulsar_test), axis = 1)
    a_probab = probabilities
    classifications = np.concatenate((sampled_pulsars_test[0,:, 8],sampled_non_pulsars_test[0,:, 8]),axis=0)
    a_class = classifications
    #global Aclass,Aprob, APULSAR,ANONPULSAR
    APULSAR=prob_pulsar_test
    ANONPULSAR=prob_non_pulsar_test
    Aclass = classifications
    Aprob = probabilities
    
    specificity_sensitivity_acc = [t.calculate_specificity_sensitivity(classifications, probabilities[j], spec_threshold) for j in range(num_weights)]
    specificity,sensitivity,spec_acc = zip(*specificity_sensitivity_acc)
    
    specificity_sensitivity_acc = [t.calculate_specificity_sensitivity(classifications, probabilities[j], acc_threshold) for j in range(num_weights)]
    specificity,sensitivity,acc_acc = zip(*specificity_sensitivity_acc)
    
    mean_spec_accuracy = np.mean(spec_acc)
    mean_acc_accuracy = np.mean(acc_acc)
    std_spec_accuracy = np.std(spec_acc)
    std_acc_accuracy = np.std(acc_acc)
    

    for j in range(num_weights):
        plot.test_probabilities(prob_pulsar_test[j],prob_non_pulsar_test[j],epochs,cross_entropy_choice,spec_threshold,acc_threshold,j,title)
        
    return std_spec_accuracy,std_acc_accuracy

def main(initial_weights,epochs,pulsar_samples,non_pulsar_samples,test_pulsar_samples,test_non_pulsar_samples,quantum_circuit,num_sets):
    global spec_threshold_mean,acc_threshold_mean,spec_acc_mean,acc_acc_mean,std_spec_acc,std_acc_acc,acc_sampling_error,spec_sampling_error,acc_initialization_error,spec_initialization_error
    
    
    initial_probabilites(initial_weights,pulsar_samples,non_pulsar_samples,quantum_circuit,num_sets)
    

    
    train_data = np.concatenate((pulsar_samples, non_pulsar_samples), axis=1)
    
    opt_prob_non_pulsar,opt_prob_pulsar,weights_crossEntropy,loss_crossEntropy = trained_probabilities(pulsar_samples, non_pulsar_samples,initial_weights,quantum_circuit,cross_entropy_choice,train_data,num_sets)
    spec_threshold_mean,acc_threshold_mean,spec_acc_mean,acc_acc_mean,std_spec_acc,std_acc_acc = mean_thresholds(opt_prob_pulsar,opt_prob_non_pulsar,train_data,num_sets)
    
    '''for i in range(num_sets):
        plot.optimized_probabilities(opt_prob_non_pulsar[i],opt_prob_pulsar[i],epochs,cross_entropy_choice,i,opt_threshold_spec[i],opt_threshold_acc[i])
        plot.loss_function(epochs,loss_crossEntropy[i],cross_entropy_choice,i)'''

    #std_spec_accuracy_train,std_acc_accuracy_train = errors(weights_crossEntropy,test_pulsar_samples,test_non_pulsar_samples,quantum_circuit,cross_entropy_choice,spec_threshold_mean,acc_threshold_mean,"",num_sets)
    #errors(weights_crossEntropy,test_pulsar_samples,test_non_pulsar_samples,quantum_circuit,cross_entropy_choice,acc_threshold_mean)
    
    acc_sampling_error,spec_sampling_error = sampling_error(acc_threshold_mean,spec_threshold_mean,initial_weights,weights_crossEntropy)


    acc_initialization_error,spec_initialization_error = initialization_error(acc_threshold_mean,spec_threshold_mean)




    print("Mean accurary threshold = ",acc_threshold_mean)
    print("Mean specificity threshold = ",spec_threshold_mean)
    print("Specificity accuracy mean = {0} ± {1}".format(spec_acc_mean,std_spec_acc))
    print("Accuracy accuracy mean = {0} ± {1}".format(acc_acc_mean,std_acc_acc))
    
    print("Accuracy Sampling Error = {}%".format(acc_sampling_error*100))
    print("Specificity Sampling Error = {}%".format(spec_sampling_error*100))

    print("Accuracy Initialization Error = {}%".format(acc_initialization_error*100))
    print("Specificity Initialization Error = {}%".format(spec_initialization_error*100))

square_loss_choice = 0
cross_entropy_choice = 1    
num_weights_global = 1
initial_weights_global = 2 * np.pi * np.random.random(size=(3,9,3))
epochs = 150
normalized_dataset = data.normalize()
num_sets_global = 29
test_size_global = 1000
train_size_global = 100
    
pulsar_samples_global,non_pulsar_samples_global,test_pulsar_samples_global,test_non_pulsar_samples_global = data.sample_pulsars(normalized_dataset, train_size_global, test_size_global, num_sets_global)#data.sample_pulsars(normalized_dataset,train_size=100,test_size=1000)

import old_quantum_circuit as quantum_circuit
main(initial_weights_global,epochs,pulsar_samples_global,non_pulsar_samples_global,test_pulsar_samples_global,test_non_pulsar_samples_global,quantum_circuit,num_sets_global)
