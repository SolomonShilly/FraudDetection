# FraudDetection

## Description
The FraudDetection project implements a credit card fraud detection system using both custom and scikit-learn models. It includes:
- A custom Logistic Regression implementation
- A custom Neural Network implementation
- Comparisons with scikit-learn's Logistic Regression and Neural Network classifiers
- Evaluation metrics such as accuracy, precision, recall, and F1 score
- Precision-Recall curve visualization

## Installation
Python 3 should be installed
- pip install pandas numpy scikit-learn matplotlib

## Dataset
creditcard_2023.csv should be in the same project directory as the FraudDetection.py file

## Code Overview
Custom Logistic Regression
- Implements a binary logistic regression model from scratch
- Uses sigmoid function for predictions
- Updates weights and bias using gradient descent
Custom Neural Network
- Implements a basic feedforward neural network with one hidden layer
- Uses sigmoid activation function
- Performs forward and backward propagation for training
Evaluation Function
- Computes and prints accuracy, precision, recall, and F1 score
- Generates and saves a Precision-Recall curve plot
Cross-Validation
- Uses K-Fold Cross-Validation to split data into 5 folds
- Trains and evaluates both custom and scikit-learn models on each fold

## Results
### === Fold 1 ===
Logistic Regression Implementation:
- Accuracy: 0.9409
- Precision: 0.9935
- Recall: 0.8879
- F1 Score: 0.9377
  
Neural Network Implementation:
- Accuracy: 0.9559
- Precision: 0.9759
- Recall: 0.9351
- F1 Score: 0.9551
  
scikit-learn Logistic Regression:
- Accuracy: 0.9652
- Precision: 0.9771
- Recall: 0.9529
- F1 Score: 0.9648
  
scikit-learn Neural Network:
- Accuracy: 0.9988
- Precision: 0.9977
- Recall: 1.0000
- F1 Score: 0.9988

### === Fold 2 ===
Logistic Regression Implementation:
- Accuracy: 0.9405
- Precision: 0.9940
- Recall: 0.8861
- F1 Score: 0.9370
  
Neural Network Implementation:
- Accuracy: 0.9558
- Precision: 0.9783
- Recall: 0.9322
- F1 Score: 0.9547
  
scikit-learn Logistic Regression:
- Accuracy: 0.9653
- Precision: 0.9785
- Recall: 0.9515
- F1 Score: 0.9648
  
scikit-learn Neural Network:
- Accuracy: 0.9989
- Precision: 0.9982
- Recall: 0.9997
- F1 Score: 0.9989

### === Fold 3 ===
Logistic Regression Implementation:
- Accuracy: 0.9407
- Precision: 0.9935
- Recall: 0.8872
- F1 Score: 0.9373
  
Neural Network Implementation:
- Accuracy: 0.9566
- Precision: 0.9777
- Recall: 0.9344
- F1 Score: 0.9556
  
scikit-learn Logistic Regression:
- Accuracy: 0.9653
- Precision: 0.9777
- Recall: 0.9523
- F1 Score: 0.9648
  
scikit-learn Neural Network:
- Accuracy: 0.9989
- Precision: 0.9978
- Recall: 0.9999
- F1 Score: 0.9989

### === Fold 4 ===
Logistic Regression Implementation:
- Accuracy: 0.9397
- Precision: 0.9935
- Recall: 0.8854
- F1 Score: 0.9363
  
Neural Network Implementation:
- Accuracy: 0.9565
- Precision: 0.9772
- Recall: 0.9349
- F1 Score: 0.9556
  
scikit-learn Logistic Regression:
- Accuracy: 0.9648
- Precision: 0.9781
- Recall: 0.9509
- F1 Score: 0.9643
  
scikit-learn Neural Network:
- Accuracy: 0.9990
- Precision: 0.9982
- Recall: 0.9999
- F1 Score: 0.9990

### === Fold 5 ===
Logistic Regression Implementation:
- Accuracy: 0.9391
- Precision: 0.9932
- Recall: 0.8841
- F1 Score: 0.9355
  
Neural Network Implementation:
- Accuracy: 0.9561
- Precision: 0.9770
- Recall: 0.9341
- F1 Score: 0.9551
  
scikit-learn Logistic Regression:
- Accuracy: 0.9639
- Precision: 0.9773
- Recall: 0.9498
- F1 Score: 0.9634
  
scikit-learn Neural Network:
- Accuracy: 0.9986
- Precision: 0.9976
- Recall: 0.9996
- F1 Score: 0.9986
