import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

'''def calculate_specificity_sensitivity(conf_matrix):
    true_negatives = conf_matrix[0, 0]
    false_positives = conf_matrix[0, 1]
    false_negatives = conf_matrix[1, 0]
    true_positives = conf_matrix[1, 1]

    # Specificity calculation
    total_actual_negatives = true_negatives + false_positives
    specificity = true_negatives / total_actual_negatives

    # Sensitivity calculation
    total_actual_positives = true_positives + false_negatives
    sensitivity = true_positives / total_actual_positives
    
    return specificity, sensitivity'''


def find_optimal_threshold(probabilities, true_labels, threshold):
    specificities = []
    sensitivities = []


    predicted_labels = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(true_labels, predicted_labels)
    specificity, sensitivity = calculate_specificity_sensitivity(cm)[:2]
    
    specificities.append(specificity)
    sensitivities.append(sensitivity)


    max_specificity = specificities[threshold]
    corresponding_sensitivity = sensitivities[threshold]

    return max_specificity, corresponding_sensitivity,specificities,sensitivities

def plot_specificity_sensitivity_vs_threshold(thresholds, specificities, sensitivities, optimal_threshold, max_specificity, corresponding_sensitivity):
    plt.plot(thresholds, specificities, label='Specificity')
    plt.plot(thresholds, sensitivities, label='Sensitivity')
    plt.scatter(optimal_threshold, max_specificity, color='red', label='Optimal Specificity Threshold')
    plt.scatter(optimal_threshold, corresponding_sensitivity, color='blue', label='Corresponding Sensitivity')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Specificity and Sensitivity vs Threshold')
    plt.legend()
    plt.show()


'''----------------------------------------------------------------------------------------------------------------------'''
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_specificity_sensitivity(true_labels, predicted_probabilities, threshold):
    predicted_labels = (predicted_probabilities >= threshold).astype(int)
    cm = confusion_matrix(true_labels, predicted_labels)

    true_negatives = cm[0, 0]
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]
    true_positives = cm[1, 1]

    # Specificity calculation
    total_actual_negatives = true_negatives + false_positives
    specificity = true_negatives / total_actual_negatives

    # Sensitivity calculation
    total_actual_positives = true_positives + false_negatives
    sensitivity = true_positives / total_actual_positives
    
    '''
    print("true_negatives == ", true_negatives)
    print("true_positives == ", true_positives)
    print("total_actual_negatives == ", total_actual_negatives)
    print("total_actual_positives == ", total_actual_positives)
    print("len(true_labels) == ",len(true_labels))
    
    print("--------------------------------")'''
    acc = accurary(true_positives,true_negatives, len(true_labels))
    
    
    return specificity, sensitivity, acc 

def accurary(TP,TN,data_length):
    return (TP+TN)/data_length


