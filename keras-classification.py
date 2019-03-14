import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split

import category_encoders as ce

# Misc
import itertools

# %% -------------------------------------------------------------------------------------------------------------------

"""
This section would use a more complex dataset from UCI ML repos
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
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


# Label encoding the response
y = pd.Series(preprocessing.LabelEncoder().fit_transform(adult_df['income'])).to_numpy()
# Target encoding the features
te = ce.TargetEncoder()
adult_df[categorical_features] = te.fit_transform(adult_df[categorical_features], y)

X = adult_df[numeric_features + categorical_features].to_numpy()

#%%
# Test train split and model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

model = Sequential()
model.add(Dense(4, activation='tanh', input_dim=13))
model.add(Dense(4, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=50)

#%%
# predictions = model.predict(X_test)
# y_pred_test = np.array(list(predictions[i, 0] for i in range(0, len(predictions))))

y_pred_test = model.predict(X_test).flatten('C')
y_pred_test[y_pred_test > 0.5] = 1
y_pred_test[y_pred_test <= 0.5] = 0

print("Accuracy score = {}".format(metrics.accuracy_score(y_test, y_pred_test, normalize=True)))
print("AUC score = {}".format(metrics.roc_auc_score(y_test, y_pred_test)))
print("Confusion matrix = \n{}".format(metrics.confusion_matrix(y_test, y_pred_test)))
print("Classification report = \n{}".format(metrics.classification_report(y_test, y_pred_test)))
