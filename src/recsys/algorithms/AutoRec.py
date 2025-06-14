from surprise import AlgoBase
from surprise import PredictionImpossible
import numpy as np
import tensorflow as tf

class Recommender(object):

    def __init__(self, visibleDimensions, epochs=200, hiddenDimensions=50, learningRate=0.1, batchSize=100):

        self.visibleDimensions = visibleDimensions
        self.epochs = epochs
        self.hiddenDimensions = hiddenDimensions
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.optimizer = tf.keras.optimizers.RMSprop(self.learningRate)
        
                
    def Train(self, X):
        
        self.initialize_weights_biases()
        for epoch in range(self.epochs):
            for i in range(0, X.shape[0], self.batchSize):
                epochX = X[i:i+self.batchSize]
                self.run_optimization(epochX)


            print("Trained epoch ", epoch)

    def GetRecommendations(self, inputUser):
                
        # Feed through a single user and return predictions from the output layer.
        rec = self.neural_net(inputUser)
        
        # It is being used as the return type is Eager Tensor.
        return rec[0]
    
    def initialize_weights_biases(self):
        # Create varaibles for weights for the encoding (visible->hidden) and decoding (hidden->output) stages, randomly initialized
        self.weights = {
            'h1': tf.Variable(tf.random.normal([self.visibleDimensions, self.hiddenDimensions])),
            'out': tf.Variable(tf.random.normal([self.hiddenDimensions, self.visibleDimensions]))
            }
        
        # Create biases
        self.biases = {
            'b1': tf.Variable(tf.random.normal([self.hiddenDimensions])),
            'out': tf.Variable(tf.random.normal([self.visibleDimensions]))
            }
    
    def neural_net(self, inputUser):

        #tf.set_random_seed(0)
        
        # Initialization of weights and biases was moved out to the initialize_weights_biases function above
        # This lets us avoid resetting them on every batch of training, which was a bug in earlier versions of
        # this script.
        
        # Create the input layer
        self.inputLayer = inputUser
        
        # hidden layer
        hidden = tf.nn.sigmoid(tf.add(tf.matmul(self.inputLayer, self.weights['h1']), self.biases['b1']))
        
        # output layer for our predictions.
        self.outputLayer = tf.nn.sigmoid(tf.add(tf.matmul(hidden, self.weights['out']), self.biases['out']))
        
        return self.outputLayer
    
    def run_optimization(self, inputUser):
        with tf.GradientTape() as g:
            pred = self.neural_net(inputUser)
            loss = tf.keras.losses.MSE(inputUser, pred)
            
        trainable_variables = list(self.weights.values()) + list(self.biases.values())
        
        gradients = g.gradient(loss, trainable_variables)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))


class AutoRec(AlgoBase):

    def __init__(self, epochs=100, hiddenDim=100, learningRate=0.01, batchSize=100, sim_options={}):
        AlgoBase.__init__(self)
        self.epochs = epochs
        self.hiddenDim = hiddenDim
        self.learningRate = learningRate
        self.batchSize = batchSize

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        numUsers = trainset.n_users
        numItems = trainset.n_items
        
        trainingMatrix = np.zeros([numUsers, numItems], dtype=np.float32)
        
        for (uid, iid, rating) in trainset.all_ratings():
            trainingMatrix[int(uid), int(iid)] = rating / 5.0
        
        # Create an RBM with (num items * rating values) visible nodes
        autoRec = Recommender(trainingMatrix.shape[1], hiddenDimensions=self.hiddenDim, learningRate=self.learningRate, batchSize=self.batchSize, epochs=self.epochs)
        autoRec.Train(trainingMatrix)

        self.predictedRatings = np.zeros([numUsers, numItems], dtype=np.float32)
        
        for uiid in range(trainset.n_users):
            if (uiid % 50 == 0):
                print("Processing user ", uiid)
            recs = autoRec.GetRecommendations([trainingMatrix[uiid]])
            
            for itemID, rec in enumerate(recs):
                self.predictedRatings[uiid, itemID] = rec * 5.0
        
        return self


    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
        
        rating = self.predictedRatings[u, i]
        
        if (rating < 0.001):
            raise PredictionImpossible('No valid prediction exists.')
            
        return rating
    