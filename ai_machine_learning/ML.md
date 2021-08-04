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
      - Possible sollutions
      - simplify the model by selecting one with fewer parameters (linear model rather than a high-degree polynomial model) by reducing the number of attributes in the training data or by constraining the model
      - gather more training data
      - reduce the noise in the training data (**regularization** , degrees of freedom)
      - hyperparameter - a parameter in a learning algorithm (not of the model). Must be set prior to training and remains constant during training
  - Underfitting the training data - when your model is too simple to learn the underlying structure of the data
      - to fix
      - select a more powerful model, with more parameters
      - feed better features to the learning algorithm (feature engineering)
      - reduce the constraints on the model


## Testing and Validating
  - Send model into production and test performance
  - Split data into a training and testing set (80% training, 20% testing)
  - The error rate on new cases is called the **generalization error** (or out of sample error). By evaluating your model on the test set, you get an estimate of this error. This value will tell you how well your model will perform on instances it has never seen before.
  - If the training error is low and the generalization error is high, it means that the model is overfitting the training data

## Hyperparameter Tuning and Model Selection
  - Holdout Validation - hold out part of the training set to evaluate several candidate models and select the best one
  - Validation set - the new testing set 
  - train multiple models with various hyperparameters on the reduced training set (full training set - validation set)
  - evaluate final model on the test set to get an estimate of the generalization error
  - perform cross validation using many small validation sets



1. How would you define machine learning? (Buildung systems that learn from data rather than having to hardcode for new data)
2. Can you name four types of problems where it shines? (Tumor detection, Chatbots, Predicting future revenues, automatically classifying news articles)
3. What is a labeled training set? (a data set that has the solutions it desires labeled for each instance)
4. What are the two most common supervised tasks? (classification, regression)
5. Can you name four common unsupervised tasks? (clustering, anomaly/novelty detection, visualization/dimensionality reduction, association rule learning)
6. What type of machine learning algorithm would you use to allow a robot to walk in various unknown terrains? (reinforcement learning)
7. What type of algorithm would you use to segment your customers into multiple groups? (clustering)
8. would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem? (supervised learning)
9. What is an online learning system? (Trains the system incrementally by feeding it data sequentially, learning fast and on the fly)
10. What is out of core learning? (online learning can also be used to train systems on huge datasets that cannot fit in the machine's main memory. the algorithm loads part of the data, runs a training step on the data, and repeats the process until it has run on all of the data.)
11. What type of learning algorithm relies on a similarity measure to make predictions? (Instanced based learning)
12. What is the difference between a model parameter and a learning algorithms hyperparameter? (A hyper parameter is part of the structure of the algorithm and must be set prior to running the model)
13. What do model based learning algorithms search for? What is the most common strategy they used to succeed? How do they make predictions? (Build a model based off examples, fit the model to the training set, and use the model to make predictions, possibly on a test set)
14. Can you name four of the main challenges in machine learning? (bad data, nonrepresentative data, uninformative features, excessively simple models that underfit the training data, and excessively complex models that overfit the data)
15. If your model performs great on the training data but generalizes poorly to new instances what is happening? Can you name three possible solutions? (The model is overfit to the training data. Simplify the model, gather more training data, reduce the noise in the training data)
16. What is a test set, and why would you want to use it? (A test set is a portion of the original data set which is *held out* and used to test the model which was trained with the training model. This would be used to avoid making errors in a live production of the model, and to test for generalization errors.)
17. What is the purpose of a validation set? (The validation set is important for selecting the model that performs best on it from the holdout validation and tune hyperparameters.)
18. What is the train-dev set, when do you need it, and how do you use it? (The train-dev set is an additional *holdout* that can be evaluated on after the model is trained on the training set. The train-dev set is used when there is a risk of mismatch between the training data and the data used in the validation and test datasets. The train-dev set, is a part of the training set that's held out(the model is not trained on it). The model is trainded on the rest of the training set, and evaluated on both the train-dev set and the validation set. If the model performs well on the training set but not on the train-dev set, then the model is likely overfitting the training set. If it performs well on both the training set and the train-dev set, but not on the validation set, then there is probably a significant data mismatch between the training data and the validation +test data, and you should try to improve the training data to make it look more like the validation + test data)
19. What can go wrong if you tune hyperparameters using the test set? (overfit the test set and the generalization error will be overly optimistic)
