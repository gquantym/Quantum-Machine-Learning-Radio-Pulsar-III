# Test data time:
#This is where we will run a new randomly sampled dataset in the quantum model with the optimized weights from the loss functions.

#Let's begin with the optimized weights from the square loss function.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def testing_data(sampled_test_data):
    pulsar_test_data = sampled_test_data[np.where(sampled_test_data[:, -1] == 1)[0]]
    non_pulsar_test_data = sampled_test_data[np.where(sampled_test_data[:, -1] == 0)[0]]
    return pulsar_test_data,non_pulsar_test_data


def find_optimal_threshold(y_prob, y_true):
    """
    Find the optimal threshold that maximizes the F1 score.

    Parameters:
    - y_true: True labels (1 for pulsar, 0 for non-pulsar)
    - y_prob: Predicted probabilities

    Returns:
    - optimal_threshold: Optimal threshold for classification
    """
    thresholds = np.linspace(0, 1, num=11)
    f1_scores = [f1_score(y_true, (y_prob > t).astype(int)) for t in thresholds]
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    print("Optimal Threshold is == ",optimal_threshold)

    return optimal_threshold

'''
def calculate_sensitivity_specificity(probabilities,classifications):
    """
    Calculate Sensitivity and Specificity.

    Parameters:
    - true_positive: Number of true positive cases
    - false_negative: Number of false negative cases
    - false_positive: Number of false positive cases
    - true_negative: Number of true negative cases

    Returns:
    - sensitivity: Sensitivity (True Positive Rate)
    - specificity: Specificity
    """
    
    threshold = find_optimal_threshold(probabilities,classifications)
    
    # Create a boolean mask for class 0 and class 1
    class_0_mask = (classifications == 0)
    class_1_mask = (classifications == 1)
    # Use boolean indexing to get features for class 0 and class 1
    predictions_non_pulsar = probabilities[class_0_mask]
    predictions_pulsar = probabilities[class_1_mask]
    
    # Apply threshold to get binary predictions
    binary_predictions_pulsar = (predictions_pulsar < threshold).astype(int)
    binary_predictions_non_pulsar = (predictions_non_pulsar > threshold).astype(int)
    
    # Calculate confusion matrix
    true_positive = np.sum((binary_predictions_pulsar == 0))
    false_negative = np.sum((binary_predictions_pulsar == 1))
    false_positive = np.sum((binary_predictions_non_pulsar == 1))
    true_negative = np.sum((binary_predictions_non_pulsar == 0))

    sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) != 0 else 0
    # Convert to float if the result is a numpy array
    sensitivity = float(sensitivity) if isinstance(sensitivity, np.ndarray) else sensitivity
    specificity = float(specificity) if isinstance(specificity, np.ndarray) else specificity
    return sensitivity, specificity, threshold
'''
def calculate_sensitivity_specificity(prob_pulsar,prob_non_pulsar,classifications):
    """
    Calculate Sensitivity and Specificity.

    Parameters:
    - true_positive: Number of true positive cases
    - false_negative: Number of false negative cases
    - false_positive: Number of false positive cases
    - true_negative: Number of true negative cases

    Returns:
    - sensitivity: Sensitivity (True Positive Rate)
    - specificity: Specificity
    """
    probabilities = np.vstack((prob_pulsar,prob_non_pulsar))
    threshold = find_optimal_threshold(probabilities, classifications)
    print("Threshold = ",threshold)
    '''# Create a boolean mask for class 0 and class 1
    class_0_mask = (classifications == 0)
    class_1_mask = (classifications == 1)
    # Use boolean indexing to get features for class 0 and class 1
    predictions_non_pulsar = probabilities[class_0_mask]
    predictions_pulsar = probabilities[class_1_mask]
    '''
    # Apply threshold to get binary predictions
    binary_predictions_pulsar = (prob_pulsar > threshold).astype(int)  # Changed here
    binary_predictions_non_pulsar = (prob_non_pulsar > threshold).astype(int)  # Changed here

    # Calculate confusion matrix
    true_positive = np.sum((binary_predictions_pulsar == 1))
    false_negative = np.sum((binary_predictions_pulsar == 0))
    false_positive = np.sum((binary_predictions_non_pulsar == 1))
    true_negative = np.sum((binary_predictions_non_pulsar == 0))

    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) != 0 else 0
    sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0

    # Convert to float if the result is a numpy array
    sensitivity = float(sensitivity) if isinstance(sensitivity, np.ndarray) else sensitivity
    specificity = float(specificity) if isinstance(specificity, np.ndarray) else specificity
    return sensitivity, specificity, threshold






'''
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix,recall_score  
# Example data (replace with your actual data)
true_labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
predicted_probabilities = np.array([0.8, 0.3, 0.6, 0.2, 0.7, 0.4, 0.9, 0.1])

def objective_function(threshold, true_labels, predicted_probabilities):
    # Convert probabilities to binary predictions based on the threshold
    binary_predictions = (predicted_probabilities > threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, binary_predictions)
    
    # Define the objective function as the negative of specificity (minimizing false positives)
    return -1 * (cm[0, 0] / (cm[0, 0] + cm[0, 1])) if (cm[0, 0] + cm[0, 1]) != 0 else 0

# Set an initial guess for the threshold
initial_threshold = 0.5
def run():
    # Set an initial guess for the threshold
    initial_threshold = 0.5
    # Optimize the threshold using the minimize function from scipy
    result = minimize(objective_function, initial_threshold, args=(true_labels, predicted_probabilities), bounds=[(0, 1)])
    
    # Get the optimal threshold from the result
    optimal_threshold = result.x[0]
    
    # Convert probabilities to binary predictions based on the optimal threshold
    optimal_binary_predictions = (predicted_probabilities > optimal_threshold).astype(int)
    
    # Evaluate the performance metrics with the optimal threshold
    optimal_cm = confusion_matrix(true_labels, optimal_binary_predictions)
    optimal_sensitivity = recall_score(true_labels, optimal_binary_predictions)
    optimal_specificity = optimal_cm[0, 0] / (optimal_cm[0, 0] + optimal_cm[0, 1]) if (optimal_cm[0, 0] + optimal_cm[0, 1]) != 0 else 0
    
    print("Optimal Threshold:", optimal_threshold)
    print("Optimal Sensitivity:", optimal_sensitivity)
    print("Optimal Specificity:", optimal_specificity)'''
