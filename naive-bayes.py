# Import scikit-learn dataset library
from sklearn import datasets, metrics, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Pandas
import pandas as pd
import numpy as np

# %% -------------------------------------------------------------------------------------------------------------------
"""
This section uses the scikit-learn wine dataset to fit a Gaussian NB model.
Predictor values are continuous.
"""
# Load dataset
wine = datasets.load_wine()

# print the names of the 13 features
print("Features: ", wine.feature_names)

# print the label type of wine(class_0, class_1, class_2)
print("Labels: ", wine.target_names)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3,
                                                    random_state=109)  # 70% training and 30% test

# Create a Gaussian Classifier
gnb = GaussianNB()

# Train the model using the training sets
gnb.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = gnb.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred))

# %% -------------------------------------------------------------------------------------------------------------------

"""
This section would use a more complex dataset from UCI ML repos
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

Following the blog on:
http://dataaspirant.com/2017/02/20/gaussian-naive-bayes-classifier-implementation-python/
"""

dftrain = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None)
dftest = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', header=None,
                     skiprows=1)

dftrain.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                   'marital_status', 'occupation', 'relationship',
                   'race', 'sex', 'capital_gain', 'capital_loss',
                   'hours_per_week', 'native_country', 'income']


# Label enconding

dftrain.describe(include='all')

"""le = preprocessing.LabelEncoder()
workclass_cat = le.fit_transform(adult_df.workclass)
education_cat = le.fit_transform(adult_df.education)
marital_cat = le.fit_transform(adult_df.marital_status)
occupation_cat = le.fit_transform(adult_df.occupation)
relationship_cat = le.fit_transform(adult_df.relationship)
race_cat = le.fit_transform(adult_df.race)
sex_cat = le.fit_transform(adult_df.sex)
native_country_cat = le.fit_transform(adult_df.native_country)"""
