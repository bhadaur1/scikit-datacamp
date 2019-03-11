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
desc = dftrain.describe(include='all')
adult_df = dftrain.copy(deep=True)

# Data imputation by most common (top) value coming from dataframe.describe
# keep in mind that if a deep copy is not made, it would change the original dataframe
for value in ['workclass', 'education',
              'marital_status', 'occupation',
              'relationship', 'race', 'sex',
              'native_country', 'income']:
    adult_df[value].replace([' ?'], [adult_df.describe(include='all')[value][2]], inplace=True)

# Define a label encoder and encode all categorical variables
le = preprocessing.LabelEncoder()
workclass_cat = le.fit_transform(adult_df.workclass)
education_cat = le.fit_transform(adult_df.education)
marital_cat = le.fit_transform(adult_df.marital_status)
occupation_cat = le.fit_transform(adult_df.occupation)
relationship_cat = le.fit_transform(adult_df.relationship)
race_cat = le.fit_transform(adult_df.race)
sex_cat = le.fit_transform(adult_df.sex)
native_country_cat = le.fit_transform(adult_df.native_country)
income_cat = le.fit_transform(adult_df.income)

# Append the new columns to the dataframe
adult_df['workclass_cat'] = workclass_cat
adult_df['education_cat'] = education_cat
adult_df['marital_cat'] = marital_cat
adult_df['occupation_cat'] = occupation_cat
adult_df['relationship_cat'] = relationship_cat
adult_df['race_cat'] = race_cat
adult_df['sex_cat'] = sex_cat
adult_df['native_country_cat'] = native_country_cat
adult_df['income'] = income_cat

# Drop the categorical columns as they have already been encoded and appended
dummy_fields = ['workclass', 'education', 'marital_status',
                'occupation', 'relationship', 'race',
                'sex', 'native_country']
adult_df.drop(dummy_fields, axis=1, inplace=True)

# Reindex columns
adult_df = adult_df.reindex(['age', 'workclass_cat', 'fnlwgt', 'education_cat',
                             'education_num', 'marital_cat', 'occupation_cat',
                             'relationship_cat', 'race_cat', 'sex_cat', 'capital_gain',
                             'capital_loss', 'hours_per_week', 'native_country_cat',
                             'income'], axis=1)

# Scale the features
num_features = ['age', 'workclass_cat', 'fnlwgt', 'education_cat', 'education_num',
                'marital_cat', 'occupation_cat', 'relationship_cat', 'race_cat',
                'sex_cat', 'capital_gain', 'capital_loss', 'hours_per_week',
                'native_country_cat']

scaled_features = {}
for each in num_features:
    mean, std = adult_df[each].mean(), adult_df[each].std()
    scaled_features[each] = [mean, std]
    adult_df.loc[:, each] = (adult_df[each] - mean) / std

# Test-train split
features = adult_df.values[:, :14]
target = adult_df.values[:, 14]
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.33,
                                                                            random_state=10)
# Initialize model
clf = GaussianNB()
clf.fit(features_train, target_train)
target_pred = clf.predict(features_test)

print("Accuracy score = {}".format(metrics.accuracy_score(target_test, target_pred, normalize=True)))
print("AUC score = {}".format(metrics.roc_auc_score(target_test, target_pred)))

# %% Now repeat with this sklearn pipelines