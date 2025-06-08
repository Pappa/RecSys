# RecSys
Course material for [Recommender Systems Udemy course](https://www.udemy.com/course/building-recommender-systems-with-machine-learning-and-ai/).

# Course content

## 01 - Getting Started
- Udemy 101: Getting the Most From This Course
- Note: Alternate dataset download location
- [Activity] Install Anaconda, course materials, and create movie recommendations!
   - [01-GettingStarted/GettingStarted.ipynb](./notebooks/01-GettingStarted/GettingStarted.ipynb)
- Course Roadmap
- What Is a Recommender System?
- Types of Recommenders
- Understanding You through Implicit and Explicit Ratings
- Top-N Recommender Architecture
- [Quiz] Review the basics of recommender systems.

## 02 - Introduction to Python [Optional]
- [Activity] The Basics of Python
- Data Structures in Python
- Functions in Python
- [Exercise] Booleans, loops, and a hands-on challenge

## 03 - Evaluating Recommender Systems
- Train/Test and Cross Validation
- Accuracy Metrics (RMSE, MAE)
- Top-N Hit Rate - Many Ways
- Coverage, Diversity, and Novelty
- Churn, Responsiveness, and A/B Tests
- [Quiz] Review ways to measure your recommender.
- [Activity] Walkthrough of RecommenderMetrics.py
- [Activity] Walkthrough of TestMetrics.py
- [Activity] Measure the Performance of SVD Recommendations
   - [03-Evaluating/TestMetrics.ipynb](./notebooks/03-Evaluating/TestMetrics.ipynb)

## 04 - A Recommender Engine Framework
- Our Recommender Engine Architecture
- [Activity] Recommender Engine Walkthrough, Part 1
- [Activity] Recommender Engine Walkthrough, Part 2
- [Activity] Review the Results of our Algorithm Evaluation.
   - [04-Framework/RecsBakeOff.ipynb](./notebooks/04-Framework/RecsBakeOff.ipynb)

## 05 - Content-Based Filtering
- Content-Based Recommendations, and the Cosine Similarity Metric
- K-Nearest-Neighbors and Content Recs
- [Activity] Producing and Evaluating Content-Based Movie Recommendations
   - [05-ContentBased/ContentRecs.ipynb](./notebooks/05-ContentBased/ContentRecs.ipynb)
- A Note on Using Implicit Ratings.
- [Activity] Bleeding Edge Alert! Mise en Scene Recommendations
- [Exercise] Dive Deeper into Content-Based Recommendations

## 06 - Neighborhood-Based Collaborative Filtering
- Measuring Similarity, and Sparsity
- Similarity Metrics
- User-based Collaborative Filtering
- [Activity] User-based Collaborative Filtering, Hands-On
   - [06-CollaborativeFiltering/SimpleUserCF.ipynb](./notebooks/06-CollaborativeFiltering/SimpleUserCF.ipynb)
- Item-based Collaborative Filtering
- [Activity] Item-based Collaborative Filtering, Hands-On
   - [06-CollaborativeFiltering/SimpleItemCF.ipynb](./notebooks/06-CollaborativeFiltering/SimpleItemCF.ipynb)
- [Exercise] Tuning Collaborative Filtering Algorithms
- [Activity] Evaluating Collaborative Filtering Systems Offline
   - [06-CollaborativeFiltering/EvaluateUserCF.ipynb](./notebooks/06-CollaborativeFiltering/EvaluateUserCF.ipynb)
- [Exercise] Measure the Hit Rate of Item-Based Collaborative Filtering
- KNN Recommenders
- [Activity] Running User and Item-Based KNN on MovieLens
   - [06-CollaborativeFiltering/KNNBakeOff.ipynb](./notebooks/06-CollaborativeFiltering/KNNBakeOff.ipynb)
- [Exercise] Experiment with different KNN parameters.
- Bleeding Edge Alert! Translation-Based Recommendations

## 07 - Matrix Factorization Methods
- Principal Component Analysis (PCA)
- Singular Value Decomposition
- [Activity] Running SVD and SVD++ on MovieLens
   - [07-MatrixFactorization/SVDBakeOff.ipynb](./notebooks/07-MatrixFactorization/SVDBakeOff.ipynb)
- Improving on SVD
- [Exercise] Tune the hyperparameters on SVD
   - [07-MatrixFactorization/SVDTuning.ipynb](./notebooks/07-MatrixFactorization/SVDTuning.ipynb)
- Bleeding Edge Alert! Sparse Linear Methods (SLIM)

## 08 - Introduction to Deep Learning [Optional]
- Deep Learning Introduction
- Deep Learning Pre-Requisites
- History of Artificial Neural Networks
- [Activity] Playing with Tensorflow
- Training Neural Networks
- Tuning Neural Networks
- Activation Functions: More Depth
- Introduction to Tensorflow
- Important Tensorflow setup note!
- [Activity] Handwriting Recognition with Tensorflow, part 1
- [Activity] Handwriting Recognition with Tensorflow, part 2
- Introduction to Keras
- [Activity] Handwriting Recognition with Keras
- Classifier Patterns with Keras
- [Exercise] Predict Political Parties of Politicians with Keras
- Intro to Convolutional Neural Networks (CNN's)
- CNN Architectures
- [Activity] Handwriting Recognition with Convolutional Neural Networks (CNNs)
- Intro to Recurrent Neural Networks (RNN's)
- Training Recurrent Neural Networks
- [Activity] Sentiment Analysis of Movie Reviews using RNN's and Keras
- Tuning Neural Networks
- Neural Network Regularization Techniques
- Generative Adversarial Networks (GAN's)
- GAN's in Action
- [Activity] Generating images of clothing with Generative Adversarial Networks

## 09 - Deep Learning for Recommender Systems
- Intro to Deep Learning for Recommenders
- Restricted Boltzmann Machines (RBM's)
- [Activity] Recommendations with RBM's, part 1
- [Activity] Recommendations with RBM's, part 2
- [Activity] Evaluating the RBM Recommender
- [Exercise] Tuning Restricted Boltzmann Machines
- Exercise Results: Tuning a RBM Recommender
- Auto-Encoders for Recommendations: Deep Learning for Recs
- [Activity] Recommendations with Deep Neural Networks
- Clickstream Recommendations with RNN's
- [Exercise] Get GRU4Rec Working on your Desktop
- Exercise Results: GRU4Rec in Action
- Bleeding Edge Alert! Generative Adversarial Networks for Recommendations
- Tensorflow Recommenders (TFRS): Intro, and Building a Retrieval Stage
- Tensorflow Recommenders (TFRS): Building a Ranking Stage
- TFRS: Incorporating Side Features and Deep Retrieval
- TFRS: Multi-Task Recommenders, Deep & Cross Networks, ScaNN, and Serving
- Bleeding Edge Alert! Deep Factorization Machines
- Neural Collaborative Filtering (NCF)
- Introducing the LibRecommender Python package
- [Activity] Movie Recommendations with Neural Collaborative Filtering
- More Emerging Tech to Watch

## 10 - Scaling it Up
- WARNING: Don't install Java 16!
- [Activity] Introduction and Installation of Apache Spark
- Apache Spark Architecture
- [Activity] Movie Recommendations with Spark, Matrix Factorization, and ALS
- [Activity] Recommendations from 20 million ratings with Spark
- Amazon DSSTNE
- DSSTNE in Action
- Scaling Up DSSTNE
- AWS SageMaker and Factorization Machines
- SageMaker in Action: Factorization Machines on one million ratings, in the cloud
- Other Systems of Note (Amazon Personalize, RichRelevance, Recombee, and more)
- Recommender System Architecture

## 11 - Real-World Challenges of Recommender Systems
- The Cold Start Problem (and solutions)
- [Exercise] Implement Random Exploration
- Exercise Solution: Random Exploration
- Stoplists
- [Exercise] Implement a Stoplist
- Exercise Solution: Implement a Stoplist
- Filter Bubbles, Trust, and Outliers
- [Exercise] Identify and Eliminate Outlier Users
- Exercise Solution: Outlier Removal
- Fraud, The Perils of Clickstream, and International Concerns
- Temporal Effects, and Value-Aware Recommendations

## 12 - Case Studies
- Case Study: YouTube, Part 1
- Case Study: YouTube, Part 2
- Case Study: Netflix, Part 1
- Case Study: Netflix, Part 2

## 13 - Hybrid Approaches
- Hybrid Recommenders and Exercise
- Exercise Solution: Hybrid Recommenders

## 14 - Wrapping Up
- More to Explore
- Bonus Lecture: More courses to explore! 