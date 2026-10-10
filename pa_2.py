# Build a simple MLP class with 2 inputs, 3 hidden nodes and one output
# Using FowardPass and Backpropagation iteratively to make predicitons
# [30 Points]
class MLP():

  """ Parameters:

      numInputs   : Number of inputs to the MLP
      numHidden   : Number of nodes in the hidden layer
      numOutputs  : Number of outputs from the MLP
      LR          : Learning rate between 0.0 and 1.0.

      Attributes:

      w1          : Weights corresponding to the inputs to the MLP
                    1D-arrary of length (numInputs + 1) with w1[0] used for bias

      w2          : Weights corresponding to the hidden nodes of the MLP
                    1D-arrary of length (numHidden + 1) with w2[0] used for bias

      z1          : Weighted-Sum input to the hidden layer
                    1D-arrary of length (numHidden)

      z2          : Weighted-Sum input to the output layer
                    1D-arrary of length (numOutputs)

      h1          : Output of the activation function at hidden layer
                    1D-arrary of length (numHidden)

    """

  # TO DO: Initialize all the parameters and internal attributes [3 Points]
  def __init__(self, numInputs = 1, numHidden = 1, numOutputs = 1, LR = 0.1):

    self.numInputs = numInputs
    self.numHidden = numHidden
    self.numOutputs = numOutputs
    self.LR = LR

    # Input -> Hidden weights
    # +1 is for bias
    self.w1 = np.random.randn(numInputs + 1, numHidden) * 0.5

    # Hidden -> Output weights
    # +1 is for bias
    self.w2 = np.random.randn(numHidden + 1, numOutputs) * 0.5

    # Intermediate values
    self.z1 = np.zeros(numHidden)
    self.h1 = np.zeros(numHidden)
    self.z2 = np.zeros(numOutputs)

    # Output prediction
    self.y_pred = np.zeros(numOutputs)


  # TO DO: Define the Sigmoid Activation function [2 Point]
  def Sigmoid(self, z):

    return 1 / (1 + np.exp(-z))



      # Return

  # TO DO: Define the derivative of Sigmoid Activation function [2 Points]
  def delSigmoid(self, z):

     s = self.Sigmoid(z)
     return s * (1 - s)
      # Return

  # TO DO: Define the Binary Cross Entropy Cost function [3 Points]
  def BCECost(self, y_pred, y_true):
     epsilon = 1e-8
     y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
     cost = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
     return cost
      # Return

  # TO DO: Implement weighted sum of features and weights ( + bias) [2 Point]
  # Weighted sum
  def weightedSum(self, X, W):

      X_bias = np.insert(X, 0, 1)

      return np.dot(X_bias, W)
  # def weightedSum(self, X):



      # Return

  # TO DO: Implement function that performs ForwardPass of data through the network [5 Points]
  def forwardPass(self, X):

    # Input -> Hidden
    self.z1 = self.weightedSum(X, self.w1)

    # Hidden activation
    self.h1 = self.Sigmoid(self.z1)

    # Hidden -> Output
    h1_bias = np.insert(self.h1, 0, 1)

    self.z2 = np.dot(h1_bias, self.w2)

    # Output activation
    self.y_pred = self.Sigmoid(self.z2)

    return self.y_pred
  # def forwardPass(self,X):

    #define function to feedforward the network



      # Use above defined functions to calculate intermediate values of the network

      # Return (all necessary variables)

  # TO DO: Implement function for Backpropagation of error - calucation of derivates [5 Points]
  def backProp(self, X, y_pred, y_true):

    # Output layer error
    delta2 = self.y_pred - y_true

    # Hidden -> Output gradient
    h1_bias = np.insert(self.h1, 0, 1)

    dw2 = np.outer(h1_bias, delta2)

    # Hidden layer error
    w2_without_bias = self.w2[1:, :]

    delta1 = (
        np.dot(w2_without_bias, delta2)
        * self.delSigmoid(self.z1)
    )

    # Input -> Hidden gradient
    X_bias = np.insert(X, 0, 1)

    dw1 = np.outer(X_bias, delta1)

    return dw1, dw2
  # def backProp(self, y_pred,y_true):



  #   return dw1, dw2

      # Use above defined functions to calculate derivates in the network

      # Return (all necessary variables)


  # TO DO: Define the Prediction (for X with current weights) [2 Points]
  def predict(self, X):
     return self.forwardPass(X)

      # Return


  # TO DO: Define training function - update weights [3 Points]
  def train(self, X, y_true):

    # Forward pass
    y_pred = self.forwardPass(X)

    # Backpropagation
    dw1, dw2 = self.backProp(
        X,
        y_pred,
        y_true
    )

    # Update weights
    self.w1 -= self.LR * dw1
    self.w2 -= self.LR * dw2

    # Calculate cost
    cost = self.BCECost(y_pred, y_true)

    return cost
  # def train(self, X, y_true):
    # Forward propagation


      # Use above defined functions in each iteration


  # TO DO: Fit Training Set Using Perceptron Learning Algorithm [3 Points]
  """ Parameters:
    Input       : An array with shape [nSamples, nFeatures]
                  nSamples - number of samples
                  nFeatuers - number of featrures

    Output      : 1D-array of length [nSamples]
                  Binary classification of each sample

    nIterations : Number of iteration of learning aglorithm to run
                  over the training set
    """
  def fit(self, Input, Output, nIterations):
    costs = []

    for iteration in range(nIterations):

        total_cost = 0

        # Train using every sample
        for X, y in zip(Input, Output):

            cost = self.train(X, y)

            total_cost += np.mean(cost)

        # Average cost
        average_cost = total_cost / len(Input)

        costs.append(average_cost)

        # Print progress
        if iteration % 100 == 0:

            print(
                "Iteration:",
                iteration,
                "Cost:",
                average_cost
            )

    return costs

      # Use above defined functions in each iteration
      # Store cost after each iteration for plotting at the end




_------------------------------------------------
# TO DO: CODE FOR TRAINING AND TESTING MLP - XOR LOGIC GATE
# [3 Points]

# Plot the cost as a function of iterations
import numpy as np
import matplotlib.pyplot as plt

# XOR input
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# XOR output
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Create MLP
model = MLP(
    numInputs=2,
    numHidden=3,
    numOutputs=1,
    LR=0.5
)

# Train the MLP
nIterations = 5000

costs = model.fit(
    X,
    y,
    nIterations
)

# Test the trained model
print("\nXOR Predictions:")

for i in range(len(X)):

    probability = model.predict(X[i])

    prediction = (probability >= 0.5).astype(int)

    print(
        X[i],
        "Actual:",
        y[i],
        "Predicted:",
        prediction,
        "Probability:",
        probability
    )
