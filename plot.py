import numpy as np
import matplotlib.pyplot as plt

def optimized_probabilities(opt_prob_non_pulsar,opt_prob_pulsar,epoch,loss_func_choice,set_num,threshold_spec,threshold_acc):
    #Plotting time
    linspace_non_pulsar = np.linspace(0, len(opt_prob_non_pulsar), len(opt_prob_non_pulsar),dtype = int)/len(opt_prob_non_pulsar) #Linspace of samplesize 
    plt.scatter(linspace_non_pulsar, opt_prob_non_pulsar, c='blue', label = "Non-Pulsars") #Plots non-pulsars
    linspace_pulsar = np.linspace(0, len(opt_prob_pulsar), len(opt_prob_pulsar),dtype = int)/len(opt_prob_pulsar) #Linspace of samplesize 
    plt.scatter(linspace_pulsar, opt_prob_pulsar, c='red', label = "Pulsars")  #Plots pulsars
    titles = ["Square Loss","Cross Entropy Loss"]
    plt.title("After Optimization - {0} - Epoch {1} - Dataset Label {2}".format(titles[loss_func_choice],epoch,set_num))
    plt.xlabel("Samples")
    plt.ylabel("Probability")
    plt.legend(bbox_to_anchor=(1.3, 0.5))
    plt.axhline(y=threshold_spec, color='g', linestyle='--', label='Horizontal Line at probability = {:.1f}'.format(threshold_spec))
    plt.axhline(y=threshold_acc, color='orange', linestyle='--', label='Horizontal Line at probability = {:.1f}'.format(threshold_acc))
    plt.ylim(0, 1)
    plt.show()

def loss_function(epoch,loss_array,loss_func_choice,set_num):
    linspace_epoch = np.linspace(0, epoch, epoch,dtype = int) #Linspace of samplesize 
    titles = ["Square Loss","Cross Entropy Loss"]
    plt.plot(linspace_epoch, loss_array, c='green', label = "{}".format(titles[loss_func_choice]))  #Plots pulsars
    plt.title("{0} against - Epoch {1} - Dataset Label {2}".format(titles[loss_func_choice],epoch,set_num))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(bbox_to_anchor=(1, 0.5))
    plt.show()

def probabilities(probability_pulsar,probability_non_pulsar,set_num):
    linspace_non_pulsar = np.linspace(0, len(probability_non_pulsar), len(probability_non_pulsar),dtype = int)/len(probability_non_pulsar) #Linspace of samplesize 
    linspace_pulsar = np.linspace(0, len(probability_pulsar), len(probability_pulsar),dtype = int)/len(probability_pulsar) #Linspace of samplesize 
    print("linspace_non_pulsar = ",len(linspace_non_pulsar))
    print("linspace_pulsar = ",len(linspace_pulsar))
    print("Probability pulsar = ",len(probability_pulsar))
    print("Probability non_pulsar = ",len(probability_non_pulsar))
    plt.scatter(linspace_non_pulsar, probability_non_pulsar, c='blue', label = "Non-Pulsars") #Plots non-pulsars
    plt.scatter(linspace_pulsar, probability_pulsar, c='red', label = "Pulsars")  #Plots pulsars
    plt.title("Before Optimization - Dataset Label {0}".format(set_num))
    plt.xlabel("Samples")
    plt.ylabel("Probability")
    plt.legend(bbox_to_anchor=(1, 0.5))
    #plt.ylim(0, 1)
    plt.show()
    

def test_probabilities(pulsar_probability,non_pulsar_probability,epoch,loss_func_choice,threshold,sensitivity,specificity,set_num):
    #import testing
    # Create a boolean mask for class 0 and class 1
    linspace_pulsar = np.linspace(0, len(pulsar_probability), len(pulsar_probability),dtype = int)/len(pulsar_probability) #Linspace of samplesize 
    linspace_non_pulsar = np.linspace(0, len(non_pulsar_probability), len(non_pulsar_probability),dtype = int)/len(non_pulsar_probability)
    plt.scatter(linspace_non_pulsar, non_pulsar_probability, c='blue', label = "Non-Pulsars") #Plots non-pulsars
    plt.scatter(linspace_pulsar, pulsar_probability, c='red', label = "Pulsars")  #Plots pulsars
    #probabilities = np.concatenate((pulsar_probability,non_pulsar_probability))
    #sensitivity,specificity,threshold = testing.calculate_sensitivity_specificity(probabilities,classifications)
    #print("Sensitivity = {}%".format(sensitivity*100))
    #print("Specificity = {}%".format(specificity*100))
    
    plt.axhline(y=threshold, color='g', linestyle='--', label='Horizontal Line at probability = {:.1f}\nSensitivity = {:.1f}%\nSpecificity = {:.1f}%'.format(threshold,sensitivity*100,specificity*100))
    titles = ["Square Loss","Cross Entropy Loss"]
    plt.title("Test data using weights from {0} - Epoch {1} - Dataset Label {2}".format(titles[loss_func_choice],epoch,set_num))
    plt.xlabel("Samples")
    plt.ylabel("Probability")
    plt.legend(bbox_to_anchor=(1, 0.5))
    plt.ylim(0, 1)
    plt.show()
