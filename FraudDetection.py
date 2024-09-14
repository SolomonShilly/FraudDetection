import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the transactions from the csv file into a Pandas dataframe
data = pd.read_csv('creditcard_2023.csv')

# Implementation of Logistic Regression Model
# Setting learning rate = to 0.01
# We will train the model in 1000 iterations which is assigned to numIterations
# weights and bias will be assigned later
class LogisticRegression:
    def __init__(self, learningRate=0.01, numIterations=1000):
        self.learningRate = learningRate
        self.numIterations = numIterations
        self.weights = None
        self.bias = None

    # Implementation of Sigmoid method using self and z parameters
    # Sigmoid formula is 1/(1 + e^x)
    # Parameters will be plugged in the sigmoid formula and the answer will be returned
    # This answer will be a value between 0 and 1 to provide a probability value based on z
    def sigmoid(self, z):
        # Sigmoid function
        return 1 / (1 + np.exp(-z))

    # The fit method will train the logistic regression for the based on the best fit
    # x is the feature and y is the target label
    # m and n will be assigned based off of the shape of X
    # m is the training examples and n is the features
    # Weights will be assigned to an n sized matrix filled with zeros
    # Bias is assigned zero
    def fit(self, x, y):
        m, n = x.shape
        self.weights = np.zeros(n)
        self.bias = 0

        # the for loop will loop based on the value of 'numIterations'
        # The value of z is computed taking the dot product of X and weights, and adding bias to it
        # z is a raw prediction we will apply to the sigmoid method
        # yPred is the value returned from the sigmoid method, it is the prediction for y
        for _ in range(self.numIterations):
            z = np.dot(x, self.weights) + self.bias
            yPred = self.sigmoid(z)

            # Compute gradients of cost function with respect to the weights and biases
            # To calculate the gradients of the weights, find the average of the dot product of X transposed and the error of the predicted y compared to y
            # TO calculate the gradients of the biases, find the average of the sum of the error between the predicted y and y
            dw = (1 / m) * np.dot(x.T, (yPred - y))
            db = (1 / m) * np.sum(yPred - y)

            # Update parameters for weights and bias for next training iteration
            self.weights -= self.learningRate * dw
            self.bias -= self.learningRate * db

    # Predict y values based on X after training the model
    # The model is binary meaning 1 is positive (Flagged transaction) and 0 is negative (Regular transaction)
    # The model identifies any prediction of y above 0.5 as a flagged transaction
    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        yPred = self.sigmoid(z)
        return (yPred > 0.5).astype(int)

# Neural Network Model implementation
class NeuralNetwork:
    def __init__(self, layers, learningRate=0.01, numIterations=1000):
        self.layers = layers
        self.learningRate = learningRate
        self.numIterations = numIterations
        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize parameters
        parameters = {}
        for i in range(1, len(self.layers)):
            parameters[f"W{i}"] = np.random.randn(self.layers[i], self.layers[i - 1]) * 0.01
            parameters[f"b{i}"] = np.zeros((self.layers[i], 1))
        return parameters

    # Implementation of Sigmoid method using self and z parameters
    # Sigmoid formula is 1/(1 + e^x)
    # Parameters will be plugged in the sigmoid formula and the answer will be returned
    # This answer will be a value between 0 and 1 to provide a probability value based on z
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # We will use forward propagation to find z and activation values for each layer
    # Cache is a dictionary to store the value of z and activation
    # W is weight and b is bias
    # Z is the dot product of the weights and activation added to bias
    # Activation is calculated using z and the sigmoid function
    # All values will be stored in cache for future use
    def forward_propagation(self, x):
        cache = {"A0": x}
        for i in range(1, len(self.layers)):
            w = self.parameters[f"W{i}"]
            b = self.parameters[f"b{i}"]
            z = np.dot(w, cache[f"A{i - 1}"]) + b
            a = self.sigmoid(z)
            cache[f"Z{i}"] = z
            cache[f"A{i}"] = a
        return cache

    # We will calulate the error of the predicted A values using the compute cost function
    # Y is the true label and A is the predicted values
    # m will be equal to the shape of Y
    # m is the number of training examples assigned based on the shape of Y, the 1 in the brackets is number of outputs
    # cost variable calculates the cross-entropy cost
    # cost is calculate by taking the average of the sum of all predicted values
    # Y * np.log(A) measures cost when the label is close to 1
    # (1 - Y) * np.log(1 - A) measures cost when the label is close to 0
    # the squeeze function makes the output a scalar value
    def compute_cost(self, A, Y):
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return np.squeeze(cost)

    # We will use backward propagation to calculate the gradients of the cost function with respect to the weights and biases
    # The grads dictionary will be used to store updated training parameters (dw and db)
    # Assigning the number of training samples to m
    # We will use a for loop to iterate through the layers of the neural network
    # The reverse command is used to loop through the layers backwards
    # The gradients will be computed and saved to the grads dictionary as we loop through each layer
    def backward_propagation(self, cache, Y):
        # Backward propagation to update gradients
        grads = {}
        m = Y.shape[1]
        for i in reversed(range(1, len(self.layers))):
            A_prev = cache[f"A{i - 1}"]
            A = cache[f"A{i}"]
            dZ = A - Y
            dW = (1 / m) * np.dot(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            grads[f"dW{i}"] = dW
            grads[f"db{i}"] = db
        return grads

    # The update parameters method will update the values of the weights and biases based on the computed gradients
    def update_parameters(self, grads):
        # Update parameters using gradients
        for i in range(1, len(self.layers)):
            self.parameters[f"W{i}"] -= self.learningRate * grads[f"dW{i}"]
            self.parameters[f"b{i}"] -= self.learningRate * grads[f"db{i}"]

    # The fit method will loop through num_iterations times
    # The method will perform forward propagation, compute costs, compute gradients, and update parameters
    def fit(self, X, y):
        # Training the neural network
        for _ in range(self.numIterations):
            cache = self.forward_propagation(X)
            cost = self.compute_cost(cache[f"A{len(self.layers) - 1}"], y)
            grads = self.backward_propagation(cache, y)
            self.update_parameters(grads)

    # The predict method will perform forward propagation
    # Next it will retrieve the predicted activation value for the previous layer
    # We will return a flattened array using .reshape(-1)
    # All probabilities greater than 0.5 will be classified as 1
    def predict(self, X):
        # Predict function
        cache = self.forward_propagation(X)
        predictions = cache[f"A{len(self.layers) - 1}"]
        return (predictions > 0.5).astype(int).reshape(-1)

# Function to evaluate models based on the true and predicted values of y
# Accuracy, precision, recall, and f1 score will be calculated using the sklearn.metrics library
# All statistics will be printed
def evaluate_model(yTrue, yPred):
    accuracy = accuracy_score(yTrue, yPred)
    precision = precision_score(yTrue, yPred)
    recall = recall_score(yTrue, yPred)
    f1 = f1_score(yTrue, yPred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Estimate precision-recall pairs on precision-recall curve
    # Calculate precision-recall curve using true and predicted values of y
    # Precision recall curve is estimated using sklearn.metrics library
    # _ is the thresholds used to compute precision and recall
    precision, recall, _ = precision_recall_curve(yTrue, yPred)

    # Plot precision-recall curve using Matplotlib
    # Save the precision-recall curve as a png
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve')
    plt.savefig(f'pr_curve_{plt.gcf().number}.png')

# Initialize K-Fold Cross-Validation to split data into 5 folds
# Each will be used once as a validation set and 4 times as a training set
# The dataset will be randomly shuffled before being split into folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# We will drop the id and class columns
# ID provides us with no information and class is what we will predict
# Convert the dataframe into a numpy array using .values
X = data.drop(['id', 'Class'], axis=1).values
y = data['Class'].values

# Logistic Regression Model for binary classification
# Neural Network Model used to recognize patterns and make predictions
# X.shape[1] retrieves the number of features/columns from X
# We set the number of neurons to 16, higher value = more cost and overfitting risk but it will learn more complex patterns
lrModel = LogisticRegression(learningRate=0.01, numIterations=1000)
nnLayers = [X.shape[1], 16, 1]
nnModel = NeuralNetwork(layers=nnLayers, learningRate=0.01, numIterations=1000)

# Perform K-Fold Cross-Validation to train neural network and logistic regression
# Sci-kit learn implementations will be included as well
# We will loop through each fold
# The data will be split into training and test sets
for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    print(f"=== Fold {fold + 1} ===")
    xTrain, xTest = X[train_index], X[test_index]
    yTrain, yTest = y[train_index], y[test_index]

    # Normalize data by removing the mean and scaling to unit variance
    # We will than fit the scaler to transform the data
    scaler = StandardScaler()
    xTrainProcessed = scaler.fit_transform(xTrain)
    xTestProcessed = scaler.transform(xTest)

    # Train and evaluate Logistic Regression implementation based on the standardized training data and the target values of y
    # The model will learn the relationship between the two variables to predict Y
    # The model's output will be printed along with the performance metrics from the evaluate model function
    lrModel.fit(xTrainProcessed, yTrain)
    yPredLR = lrModel.predict(xTestProcessed)
    print("Logistic Regression Implementation:")
    evaluate_model(yTest, yPredLR)

    # Train and evaluate Neural Network implementation
    # We will transpose X to switch the features from columns to rows for neural network organization
    # YPredNN is the predicted value of the trained neural network
    # The model's output will be printed along with the performance metrics from the evaluate model function
    nnModel.fit(xTrainProcessed.T, yTrain.reshape(1, -1))
    yPredNN = nnModel.predict(xTestProcessed.T)
    print("Neural Network Implementation:")
    evaluate_model(yTest, yPredNN)

    # Initialize scikit-learn Logistic Regression and Neural Network by setting the max number of iterations and size of the neural network's layers
    skLRModel = SklearnLogisticRegression(max_iter=1000)
    skNNModel = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000)

    # Train and evaluate scikit-learn Logistic Regression using the same data and performance metrics from above
    skLRModel.fit(xTrainProcessed, yTrain)
    yPredSKLR = skLRModel.predict(xTestProcessed)
    print("scikit-learn Logistic Regression:")
    evaluate_model(yTest, yPredSKLR)

    # Train and evaluate scikit-learn Neural Network using the same data and performance metrics from above
    skNNModel.fit(xTrainProcessed, yTrain)
    yPredSKNN = skNNModel.predict(xTestProcessed)
    print("scikit-learn Neural Network:")
    evaluate_model(yTest, yPredSKNN)