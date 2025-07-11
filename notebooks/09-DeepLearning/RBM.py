from surprise import AlgoBase
from surprise import PredictionImpossible
import numpy as np
import tensorflow as tf

class Recommender(object):

    def __init__(self, visible_dimensions, epochs=20, hidden_dimensions=50, rating_values=10, learning_rate=0.001, batch_size=100):

        self.visible_dimensions = visible_dimensions
        self.epochs = epochs
        self.hidden_dimensions = hidden_dimensions
        self.rating_values = rating_values
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
                
    def train(self, X):

        # Initialize weights randomly (earlier versions of thie code had this block inside make_graph, but that was a bug.)
        max_weight = -4.0 * np.sqrt(6.0 / (self.hidden_dimensions + self.visible_dimensions))
        self.weights = tf.Variable(tf.random.uniform([self.visible_dimensions, self.hidden_dimensions], minval=-max_weight, maxval=max_weight), tf.float32, name="weights")
        self.hidden_bias = tf.Variable(tf.zeros([self.hidden_dimensions], tf.float32, name="hidden_bias"))
        self.visible_bias = tf.Variable(tf.zeros([self.visible_dimensions], tf.float32, name="visible_bias"))

        for epoch in range(self.epochs):
            
            tr_x = np.array(X)
            for i in range(0, tr_x.shape[0], self.batch_size):
                epoch_x = tr_x[i:i+self.batch_size]
                self.make_graph(epoch_x)

            print("Trained epoch ", epoch)


    def get_recommendations(self, input_user):
        
        feed = self.make_hidden(input_user)
        rec = self.make_visible(feed)
        return rec[0]       

    def make_graph(self, input_user):
        
        # Perform Gibbs Sampling for Contrastive Divergence, per the paper we assume k=1 instead of iterating over the 
        # forward pass multiple times since it seems to work just fine
        
        # Forward pass
        # Sample hidden layer given visible...
        # Get tensor of hidden probabilities
        h_prob0 = tf.nn.sigmoid(tf.matmul(input_user, self.weights) + self.hidden_bias)
        # Sample from all of the distributions
        h_sample = tf.nn.relu(tf.sign(h_prob0 - tf.random.uniform(tf.shape(h_prob0))))
        # Stitch it together
        forward = tf.matmul(tf.transpose(input_user), h_sample)
        
        # Backward pass
        # Reconstruct visible layer given hidden layer sample
        v = tf.matmul(h_sample, tf.transpose(self.weights)) + self.visible_bias
        
        # Build up our mask for missing ratings
        v_mask = tf.sign(input_user) # Make sure everything is 0 or 1
        v_mask_3d = tf.reshape(v_mask, [tf.shape(v)[0], -1, self.rating_values]) # Reshape into arrays of individual ratings
        v_mask_3d = tf.reduce_max(v_mask_3d, axis=[2], keepdims=True) # Use reduce_max to either give us 1 for ratings that exist, and 0 for missing ratings
        
        # Extract rating vectors for each individual set of 10 rating binary values
        v = tf.reshape(v, [tf.shape(v)[0], -1, self.rating_values])
        v_prob = tf.nn.softmax(v * v_mask_3d) # Apply softmax activation function
        v_prob = tf.reshape(v_prob, [tf.shape(v)[0], -1]) # And shove them back into the flattened state. Reconstruction is done now.
        # Stitch it together to define the backward pass and updated hidden biases
        h_prob1 = tf.nn.sigmoid(tf.matmul(v_prob, self.weights) + self.hidden_bias)
        backward = tf.matmul(tf.transpose(v_prob), h_prob1)
    
        # Now define what each epoch will do...
        # Run the forward and backward passes, and update the weights
        weight_update = self.weights.assign_add(self.learning_rate * (forward - backward))
        # Update hidden bias, minimizing the divergence in the hidden nodes
        hidden_bias_update = self.hidden_bias.assign_add(self.learning_rate * tf.reduce_mean(h_prob0 - h_prob1, 0))
        # Update the visible bias, minimizng divergence in the visible results
        visible_bias_update = self.visible_bias.assign_add(self.learning_rate * tf.reduce_mean(input_user - v_prob, 0))

        self.update = [weight_update, hidden_bias_update, visible_bias_update]
        
    def make_hidden(self, input_user):
        hidden = tf.nn.sigmoid(tf.matmul(input_user, self.weights) + self.hidden_bias)
        self.make_graph(input_user)
        return hidden
    
    def make_visible(self, feed):
        visible = tf.nn.sigmoid(tf.matmul(feed, tf.transpose(self.weights)) + self.visible_bias)
        #self.make_graph(feed)
        return visible


class RBM(AlgoBase):

    def __init__(self, epochs=20, hidden_dim=100, learning_rate=0.001, batch_size=100, sim_options={}):
        AlgoBase.__init__(self)
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        n_users = trainset.n_users
        n_items = trainset.n_items
        
        training_matrix = np.zeros([n_users, n_items, 10], dtype=np.float32)
        
        for (uid, iid, rating) in trainset.all_ratings():
            adjusted_rating = int(float(rating)*2.0) - 1
            training_matrix[int(uid), int(iid), adjusted_rating] = 1
        
        # Flatten to a 2D array, with nodes for each possible rating type on each possible item, for every user.
        training_matrix = np.reshape(training_matrix, [training_matrix.shape[0], -1])
        
        # Create an RBM with (num items * rating values) visible nodes
        rbm = Recommender(training_matrix.shape[1], hidden_dimensions=self.hidden_dim, learning_rate=self.learning_rate, batch_size=self.batch_size, epochs=self.epochs)
        rbm.train(training_matrix)

        self.predicted_ratings = np.zeros([n_users, n_items], dtype=np.float32)
        for uiid in range(trainset.n_users):
            if (uiid % 50 == 0):
                print("Processing user ", uiid)
            recs = rbm.get_recommendations([training_matrix[uiid]])
            recs = np.reshape(recs, [n_items, 10])
            
            for item_id, rec in enumerate(recs):
                # The obvious thing would be to just take the rating with the highest score:                
                #rating = rec.argmax()
                # ... but this just leads to a huge multi-way tie for 5-star predictions.
                # The paper suggests performing normalization over K values to get probabilities
                # and take the expectation as your prediction, so we'll do that instead:
                normalized = self.softmax(rec)
                rating = np.average(np.arange(10), weights=normalized)
                self.predicted_ratings[uiid, item_id] = (rating + 1) * 0.5
        
        return self


    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
        
        rating = self.predicted_ratings[u, i]
        
        if (rating < 0.001):
            raise PredictionImpossible('No valid prediction exists.')
            
        return rating
    