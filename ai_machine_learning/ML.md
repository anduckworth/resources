# Hands On Machine Learning with Sci-Kit Learn and TensorFlow OREILLY

## Machine Learning Is Great For
  - Problems for which existing solutions require a lot of fine tuning or long lists of rules: one machine learning algorithm can often simplify code and perform better than the traditional approach.
  - Complex problems for which using a traditional approach yields no good solution: the best Machine Learning techniques can perhaps find a solution.
  - Fluctuating environments: a machine learning system can adapt to new data
  - Getting insights about complex problems and large amounts of data.

## Examples of Machine Learning
  - Analyzing images of products on a production line to automatically classify them
  - Detecting tumors in brain scans
  - Automatically classifying news articles
  - Automatically flagging offensive comments on discussion forums
  - Summarizing long documents automatically
  - Creating a chatbot or personal assistant
  - Forecasting your company's revenue next year, based on many performance metrics
      - This is a regression task (ie. predicting values) that may be tackled using any regression model such as a linear regression or polynomial regression model a regression SVM a regression random forest or an artificial neural network. RNN, CNN, ir transformers to help take into account sequences of past performance metrics.
  - Making your app react to voice commands
  - Detecting credit card fraud
  - Segmenting clinets based on their purchases so that you can design a different marketing strategy for each segment
      - Clustering
  - Representing a complex, high dimensional dataset ina clear and insightful diagram
  - Recommending a product that a client may be interested in, based on past purchases
  - Building an intelligent bot for a game


## Types of Machine Learning
  - Trained or not trained by a human (supervised, unsupervised, semisupervised, and reinforcement learning)
      - Supervised learning - the training set you feed the algorithm includes the desired solutions, called labels
          - Classification
          - Predict a target numeric value given a set of features called predictors. This sort of task is regression.
          - attribute is a data type; feature is attribute + value
          - K-Nearest Neighbors, Linear Regression, Logistic Regression, Support Vector Machines(SVMs), Decision Trees and Random Forests, Neural Networks
      - Unsupervised learning - the training data is unlabeled, the system tries to learn without a teacher
          - **Clustering**: K-means, DBSCAN, Heirachical cluster analysis (HCA)
          - Anomaly detection and novelty detection: one class SVM, isolation forest
          - Visualization and **dimensionality reduction**: principal component analysis (PCA), kernal PCA, locally linear embedding, t-distributed stochastic neighbor embedding (t-SNE)
          - Association rule learning: apriori, eclat
      - Supervised Learning - Some algorithms can deal with data that's partially labeled
      - Reinforcement learning - The learning system, called an agent in this contect, can observe the environment, select and perform actions, as shoow in rewards in return (or penalties in the form of negative rewards). It must then learn by itself what is th ebest strategy, called a policy, to get the most rward over time.
  - Learn incrementally on the fly or not (online vs. batch learning)
      - batch learning - the system is incapable of learning incrementally: it must be trained using all available data. Offline learning.
      - online learning - train the system incrementally by feeding it data instances sequentially, either individually or in small groups called mini-batches. Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrives.
          - Learning Rate - high learning rate = rapid adaptation to new data, but tends to quickly forget the old data; low learning rate = more inertia, learn more slowly, but less sensitive to noise in the new data or to sequences of outliers
          - Drops in performance can occur with bad data ingestion
  - Work by simply comparing new data points to known data points, or instead by detecting patterns in the training data and building a predictive model, much like scientists do (instance based vs. model based learning)
      - generalizing for future instances and performing well with these new instances
      - instance based learning - The system learns the examples by heart, then generalizes to new cases by using a similarity measure to compare them to the learned examples (or a subset of them)
      - model based learning - generalize from a set of examples by building a model of these examples and then using that model to make predictions
          - Model selection
          - Utility/Fitness function - measures how good your model is
          - Cost function - measures how bad your model is 
          - For linear regression problems, people typically use a cost function that measires the distance between the linear model's predictions and the training examples; the objective is to minimize the distance
          - Training - feed in yoyur training examples, and find the parameters that make the linear model fit best to your data
  - Combine these like thanos did with the gems

## Process
  - Study the data
  - Select a model
  - Train the model on the training data (ie, the learning algorithm searched for the model parameter values that minimize a cost function)
  - Apply the model to make predictions on new cases, inference

## Bad Data
  - Insufficient quantity of data
  - Nonrepresentative Training Data - it is crucial that your training data be representative of the new cases you want to generalize to
      - If the sample is too small, you will have **sampling noise** (nonrepresentative data as a result of chance
      - Very large samples can be nonrepresentative if the sampling method is flawed, **sampling bias**
  - Poor Quality Data
  - Irrelevant Features
      - Feature engineering - coming up with a good set of features to train on
      - Feature selection - selecting the most useful features to train on among existing features
      - Feature extraction - combining eexisting features to produce a more usefule one, ie. dimensionality reduction
      - Creating new features by gathering new data
 
 ## Bad Algorithms
  - Overfitting the training data - model performs well on the training data, but it does not generalize well
