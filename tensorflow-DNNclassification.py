# Import scikit-learn dataset library
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split

# Pandas
import pandas as pd
import numpy as np

# Misc
import itertools

# TensorFlow library
import tensorflow as tf

# %% -------------------------------------------------------------------------------------------------------------------

"""
This section would use a more complex dataset from UCI ML repos
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

Following the blog on:
http://dataaspirant.com/2017/02/20/gaussian-naive-bayes-classifier-implementation-python/
"""

dftrain = pd.read_csv('adult.train', header=None)
dftest = pd.read_csv('adult.test', header=None, skiprows=1)

dftrain.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                   'marital_status', 'occupation', 'relationship',
                   'race', 'sex', 'capital_gain', 'capital_loss',
                   'hours_per_week', 'native_country', 'income']

dftest.columns = dftrain.columns

del dftrain["fnlwgt"]
del dftest["fnlwgt"]

# Define numeric and categorical features
categorical_features = ['workclass', 'education',
                        'marital_status', 'occupation',
                        'relationship', 'race', 'sex',
                        'native_country']

numeric_features = ['age', # 'fnlwgt',
                    'education_num', 'capital_gain',
                    'capital_loss', 'hours_per_week']
# %% -------------------------------------------------------------------------------------------------------------------

# Make a deep copy
adult_df = dftrain.copy(deep=True)

# Imputation
for value in itertools.chain(categorical_features, ['income']):
    adult_df[value].replace([' ?'], [adult_df.describe(include='all')[value][2]], inplace=True)
#%%
# Test train split
X = adult_df[numeric_features + categorical_features]
y = pd.Series(preprocessing.LabelEncoder().fit_transform(adult_df['income']))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

#%%
# Tensorflow initialize
tf_feat_col = []
for col in categorical_features:
    temp = tf.feature_column.categorical_column_with_hash_bucket(col, hash_bucket_size=len(adult_df[col].unique()))
    tf_feat_col.append(
        tf.feature_column.embedding_column(temp, dimension=len(adult_df[col].unique())))

for col in numeric_features:
    tf_feat_col.append(tf.feature_column.numeric_column(col))
# %%
# Input function
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=100, num_epochs=10000, shuffle=True)

# Define model as Linear Classifier
model = tf.estimator.DNNClassifier(
    hidden_units=[100, 50],
    feature_columns=tf_feat_col,
    n_classes=2,
    activation_fn=tf.nn.sigmoid,
    optimizer=lambda: tf.train.AdamOptimizer(
        learning_rate=tf.train.exponential_decay(learning_rate=0.001,
                                                 global_step=tf.train.get_global_step(),
                                                 decay_steps=100,
                                                 decay_rate=0.96)))

# Train the model
model.train(input_fn=input_func, steps=2800)

# Evaluate the model
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
results = model.evaluate(eval_input_func)

out = list(model.predict(eval_input_func))  # This is a list of dict, model.predict gives a generator
y_pred_test = np.array(list(out[i]['class_ids'][0] for i in range(len(out))))

print("Accuracy score = {}".format(metrics.accuracy_score(y_test, y_pred_test, normalize=True)))
print("AUC score = {}".format(metrics.roc_auc_score(y_test, y_pred_test)))
print("Confusion matrix = \n{}".format(metrics.confusion_matrix(y_test, y_pred_test)))

print("Classification report = \n{}".format(metrics.classification_report(y_test, y_pred_test)))

