import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def calculate_specificity_sensitivity(conf_matrix):
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

    return specificity, sensitivity

def find_optimal_threshold(probabilities, true_labels):
    thresholds = np.linspace(0, 1, 100)
    specificities = []
    sensitivities = []

    for threshold in thresholds:
        predicted_labels = (probabilities >= threshold).astype(int)
        cm = confusion_matrix(true_labels, predicted_labels)
        specificity, sensitivity = calculate_specificity_sensitivity(cm)
        specificities.append(specificity)
        sensitivities.append(sensitivity)

    optimal_threshold_index = np.argmax(specificities)
    optimal_threshold = thresholds[optimal_threshold_index]
    max_specificity = specificities[optimal_threshold_index]
    corresponding_sensitivity = sensitivities[optimal_threshold_index]

    return optimal_threshold, max_specificity, corresponding_sensitivity, thresholds, specificities, sensitivities

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





