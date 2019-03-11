# Import scikit-learn dataset library
from sklearn import datasets, metrics, preprocessing, svm, naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Pipeline related stuff
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Category encoder
import category_encoders as ce

# Pandas
import pandas as pd
import numpy as np

# Misc
import itertools


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

# Define numeric and categorical features
categorical_features = ['workclass', 'education',
                        'marital_status', 'occupation',
                        'relationship', 'race', 'sex',
                        'native_country']

numeric_features = ['age', 'fnlwgt',
                    'education_num', 'capital_gain',
                    'capital_loss', 'hours_per_week']


#%%
"""
Brute force implementation without Pipelines
"""

# Label enconding
desc = dftrain.describe(include='all')
adult_df = dftrain.copy(deep=True)

# Data imputation by most common (top) value coming from dataframe.describe
# Followed by target encoding
# keep in mind that if a deep copy is not made, it would change the original dataframe
# Define a label encoder and encode all categorical variables

le = preprocessing.LabelEncoder()

# We add income -- the response variable in here too
for value in itertools.chain(categorical_features, ['income']):
    adult_df[value].replace([' ?'], [adult_df.describe(include='all')[value][2]], inplace=True)
    adult_df[value+'_cat'] = le.fit_transform(adult_df[value])

# Drop the categorical columns as they have already been encoded and appended
adult_df.drop(categorical_features, axis=1, inplace=True)

# Reindex columns
adult_df = adult_df.reindex(['age', 'workclass_cat', 'fnlwgt', 'education_cat',
                             'education_num', 'marital_status_cat', 'occupation_cat',
                             'relationship_cat', 'race_cat', 'sex_cat', 'capital_gain',
                             'capital_loss', 'hours_per_week', 'native_country_cat',
                             'income_cat'], axis=1)

# Scale the features manually
num_features = ['age', 'workclass_cat', 'fnlwgt', 'education_cat', 'education_num',
                'marital_status_cat', 'occupation_cat', 'relationship_cat', 'race_cat',
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
print("Confusion matrix = \n{}".format(metrics.confusion_matrix(target_test, target_pred)))

# %% Now repeat with this sklearn pipelines

# Define imputation and scaling strategy
numeric_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scalar', preprocessing.StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(missing_values=' ?', strategy='most_frequent')),
        ('targetencode', ce.TargetEncoder()),
        #('labelencode', ce.OrdinalEncoder()),
        #('binaryencode', ce.BinaryEncoder()),
        #('scalar', preprocessing.StandardScaler())
    ]
)

# Assemble transformation tasks
preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', numeric_transformer, numeric_features),
        ('categorical', categorical_transformer, categorical_features)
    ]
)

# Create model pipeline -- that will first run the preprocessor and then apply model
pplmodel = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('GaussianNB', naive_bayes.GaussianNB())
        #('svm-linear', svm.SVC(gamma=0.001, C=100., kernel='linear'))
    ]
)

# Split into features and response
dfXall = dftrain[numeric_features+categorical_features]
yall = preprocessing.LabelEncoder().fit_transform(dftrain['income'])

# If we want to use label encoder
"""le = preprocessing.LabelEncoder()
for value in categorical_features:
    dfXall[value].replace([' ?'], [dfXall.describe(include='all')[value][2]], inplace=True)
    dfXall[value] = le.fit_transform(dfXall[value])
# dfXall[categorical_features] = dfXall[categorical_features].apply(preprocessing.LabelEncoder().fit_transform)
"""
# Test-train split and resetting index
X_train, X_test, y_train, y_test = train_test_split(dfXall, yall, test_size=0.33,
                                                    random_state=10)
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

pplmodel.fit(X_train, y_train)
y_pred_train = pplmodel.predict(X_train)
y_pred_test = pplmodel.predict(X_test)

print("Accuracy score = {}".format(metrics.accuracy_score(y_test, y_pred_test, normalize=True)))
print("AUC score = {}".format(metrics.roc_auc_score(y_test, y_pred_test)))
print("Confusion matrix = \n{}".format(metrics.confusion_matrix(y_test, y_pred_test)))
