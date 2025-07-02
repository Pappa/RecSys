from surprise import AlgoBase
from surprise import PredictionImpossible
import numpy as np
import tensorflow as tf

class Recommender(object):

    def __init__(self, visible_dimensions, epochs=200, hidden_dimensions=50, learning_rate=0.1, batch_size=100):

        self.visible_dimensions = visible_dimensions
        self.epochs = epochs
        self.hidden_dimensions = hidden_dimensions
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
        
                
    def train(self, X):
        
        self.initialize_weights_biases()
        for epoch in range(self.epochs):
            for i in range(0, X.shape[0], self.batch_size):
                epochX = X[i:i+self.batch_size]
                self.run_optimization(epochX)


            print("Trained epoch ", epoch)

    def get_recommendations(self, input_user):
                
        # Feed through a single user and return predictions from the output layer.
        rec = self.neural_net(input_user)
        
        # It is being used as the return type is Eager Tensor.
        return rec[0]
    
    def initialize_weights_biases(self):
        # Create varaibles for weights for the encoding (visible->hidden) and decoding (hidden->output) stages, randomly initialized
        self.weights = {
            'h1': tf.Variable(tf.random.normal([self.visible_dimensions, self.hidden_dimensions])),
            'out': tf.Variable(tf.random.normal([self.hidden_dimensions, self.visible_dimensions]))
        }
        
        # Create biases
        self.biases = {
            'b1': tf.Variable(tf.random.normal([self.hidden_dimensions])),
            'out': tf.Variable(tf.random.normal([self.visible_dimensions]))
        }
    
    def neural_net(self, input_user):

        #tf.set_random_seed(0)
        
        # Initialization of weights and biases was moved out to the initialize_weights_biases function above
        # This lets us avoid resetting them on every batch of training, which was a bug in earlier versions of
        # this script.
        
        # Create the input layer
        self.input_layer = input_user
        
        # hidden layer
        hidden = tf.nn.sigmoid(tf.add(tf.matmul(self.input_layer, self.weights['h1']), self.biases['b1']))
        
        # output layer for our predictions.
        self.output_layer = tf.nn.sigmoid(tf.add(tf.matmul(hidden, self.weights['out']), self.biases['out']))
        
        return self.output_layer
    
    def run_optimization(self, input_user):
        with tf.GradientTape() as g:
            pred = self.neural_net(input_user)
            loss = tf.keras.losses.MSE(input_user, pred)
            
        trainable_variables = list(self.weights.values()) + list(self.biases.values())
        
        gradients = g.gradient(loss, trainable_variables)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))


class AutoRec(AlgoBase):

    def __init__(self, epochs=100, hidden_dim=100, learning_rate=0.01, batch_size=100, sim_options={}):
        AlgoBase.__init__(self)
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        n_users = trainset.n_users
        n_items = trainset.n_items
        
        training_matrix = np.zeros([n_users, n_items], dtype=np.float32)
        
        for (uid, iid, rating) in trainset.all_ratings():
            training_matrix[int(uid), int(iid)] = rating / 5.0
        
        # Create an RBM with (num items * rating values) visible nodes
        autoRec = Recommender(training_matrix.shape[1], hidden_dimensions=self.hidden_dim, learning_rate=self.learning_rate, batch_size=self.batch_size, epochs=self.epochs)
        autoRec.train(training_matrix)

        self.predicted_ratings = np.zeros([n_users, n_items], dtype=np.float32)
        
        for uiid in range(trainset.n_users):
            if (uiid % 50 == 0):
                print("Processing user ", uiid)
            recs = autoRec.get_recommendations([training_matrix[uiid]])
            
            for item_id, rec in enumerate(recs):
                self.predicted_ratings[uiid, item_id] = rec * 5.0
        
        return self


    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
        
        rating = self.predicted_ratings[u, i]
        
        if (rating < 0.001):
            raise PredictionImpossible('No valid prediction exists.')
            
        return rating
    