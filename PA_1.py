# Build Simple Perceptron Model (Rosenblatt) for Binary Classification
# [15 Points]
class Perceptron:

    """ Parameters:

        N  : Number of inputs to the perceptron
        LR : Learning rate between 0.0 and 1.0.

        Attributes:

        w : Weights corresponding to the inputs of the perceptron
            1D-arrary of length N+1 with w[0] used for bias
    """

    # TO DO: Initialize the weights and learning rate [1 Point]
    def __init__(self, N, LR = 0.01):
      self.W = [0.0]*(N+1)
      self.LR = LR



    # TO DO: Implement weighted sum of features and weights ( + bias) [2 Point]
    def weighted_sum(self, X):
      z = self.W[0]
      for i in range(len(X)):
        z+=self.W[i+1] * X[i]
      return z

      # Return


    # TO DO: Define the simple Step activation function [1 Point]
    def step_function(self, z):
      if(z>=0):
        return 1
      else :
        return 0

      # Return


    # TO DO: Define the Prediction (for X with current weights) [2 Points]
    def predict(self, X):
      z = self.weighted_sum(X)
      return self.step_function(z)

      # Return


    # TO DO: Define the Loss function [1 Point]
    def loss(self, y_pred, y_true):
      return y_true - y_pred

      # Return

    # TO DO: Define training function - update weights [4 Points]
    def train(self, X, y_true):
      y_pred = self.predict(X)
      error = self.loss(y_pred,y_true)
      self.W[0]  = self.W[0] + self.LR * error
      for i in range(len(X)):
        self.W[i+1] = self.W[i+1] + self.LR * error * X[i]





    # TO DO: Fit Training Set Using Perceptron Learning Algorithm [4 Points]
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
      for t in range(nIterations):
        for i in range(len(Input)):
          self.train(Input[i],Output[i])
      return self



      # Use above defined functions in each iteration



# TO DO: CODE FOR TRAINING AND TESTING PERCPETRON - AND GATE
# [3 Points]
Input = [[0,0],[0,1],[1,0],[1,1]]
Output = [0,0,0,1]
model = Perceptron(N=2, LR=0.1)

model.fit(Input, Output, nIterations=10)
for x in Input:
    print(x, "->", model.predict(x))

print("Final weights:", model.W)
