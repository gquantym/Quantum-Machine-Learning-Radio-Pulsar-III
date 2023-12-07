import numpy as np
from sklearn.metrics import confusion_matrix

def find_optimal_threshold(y_prob, y_true):
    """
    Find the optimal threshold that maximizes specificity.

    Parameters:
    - y_true: True labels (0 for negative, 1 for positive)
    - y_prob: Predicted probabilities

    Returns:
    - optimal_threshold: Optimal threshold for classification
    """
    thresholds = np.linspace(0, 1, num=101)  # Adjust the number of thresholds as needed
    specificity_scores = []

    for threshold in thresholds:
        # Convert probabilities to binary predictions based on the threshold
        binary_predictions = (y_prob > threshold).astype(int)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, binary_predictions)

        # Calculate specificity
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) != 0 else 0
        specificity_scores.append(specificity)

    optimal_threshold = thresholds[np.argmax(specificity_scores)]
    print("Optimal Threshold is == ", optimal_threshold)

    return optimal_threshold

def calculate_sensitivity_specificity(y_prob, y_true):
    """
    Calculate Sensitivity and Specificity.

    Parameters:
    - y_true: True labels (0 for negative, 1 for positive)
    - y_prob: Predicted probabilities

    Returns:
    - sensitivity: Sensitivity (True Positive Rate)
    - specificity: Specificity
    - optimal_threshold: Optimal threshold for classification
    """
    optimal_threshold = find_optimal_threshold(y_prob, y_true)

    # Convert probabilities to binary predictions based on the optimal threshold
    binary_predictions = (y_prob > optimal_threshold).astype(int)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, binary_predictions)

    # Calculate sensitivity and specificity
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) != 0 else 0
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) != 0 else 0

    # Convert to float if the result is a numpy array
    sensitivity = float(sensitivity) if isinstance(sensitivity, np.ndarray) else sensitivity
    specificity = float(specificity) if isinstance(specificity, np.ndarray) else specificity

    return sensitivity, specificity, optimal_threshold
