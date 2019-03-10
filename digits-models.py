"""
This exercise is on classification using the digits data stored in scikit-learn datasets library.
The first part is to plot the dataset, then we move to use PCA and visualize the lower dimensional dataset.
Then under the assumptions that the labels are not provided, we use a Kmeans algorithm to cluster data.
Later we use a SVM with grid search to find optimal hyper-parameters

Mostly followed the tutorial by Karlijn Willems at datacamp

https://www.datacamp.com/community/tutorials/machine-learning-python

"""
# Data manipulation libs
import pandas as pd
import numpy as np

# Plotting libs
import matplotlib.pyplot as plt

# ML libs
from sklearn import datasets, svm, cluster, metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.manifold import Isomap

digits = datasets.load_digits()
# digits = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", header=None)

# ------------------------------- Data exploration -----------------------------------------
# print(digits.DESCR)
print('keys = {}'.format(digits.keys()))
print('target = {}'.format(digits.target))

# Figure size (width, height) in inches
fig = plt.figure(figsize=(6, 6))

# Adjust the subplots
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images
for i in range(64):
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    # Display an image at the i-th position
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

# Show the plot
plt.show()

# ---------------------------- Data projection into subspace and visualization -----------------------------------------

# Create a Randomized PCA model that takes two components
randomized_pca = PCA(n_components=2, svd_solver='randomized')

# Fit and transform the data to the model
reduced_data_rpca = randomized_pca.fit_transform(digits.data)

# Create a regular PCA model
pca = PCA(n_components=2)

# Fit and transform the data to the model
reduced_data_pca = pca.fit_transform(digits.data)

# Scatter plot to visualize the PCA data
colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
for i in range(len(colors)):
    x = reduced_data_rpca[:, 0][digits.target == i]
    y = reduced_data_rpca[:, 1][digits.target == i]
    plt.scatter(x, y, c=colors[i])
plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("PCA Scatter Plot")
plt.show()

# --------------------------------------- K Means to cluster data ------------------------------------------------------

data = scale(digits.data)

X_train, X_test, y_train, y_test, images_train, images_test = \
    train_test_split(data, digits.target, digits.images, test_size=0.25, random_state=42)

n_samples, n_features = X_train.shape

clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)
clf.fit(X_train)

"""
Visualize centroids -- does it resemble to the numbers????
"""
# Figure size in inches
fig = plt.figure(figsize=(8, 3))

# Add title
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

# For all labels (0-9)
for i in range(10):
    # Initialize subplots in a grid of 2X5, at i+1th position
    ax = fig.add_subplot(2, 5, 1 + i)
    # Display images
    ax.imshow(clf.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
    # Don't show the axes
    plt.axis('off')

# Show the plot
plt.show()

# ---------------------   Do predictions and study metrics -----------------------------------

# Predict the labels for `X_test`
y_pred = clf.predict(X_test)

# Print out the first 100 instances of `y_pred`
print("y_pred = {}".format(y_pred[:100]))

# Print out the first 100 instances of `y_test`
print("y_test = {}".format(y_test[:100]))

# Study the shape of the cluster centers
clf.cluster_centers_.shape

print('% 9s' % 'inertia    homo   compl  v-meas     ARI AMI  silhouette')
print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
      % (clf.inertia_,
         metrics.homogeneity_score(y_test, y_pred),
         metrics.completeness_score(y_test, y_pred),
         metrics.v_measure_score(y_test, y_pred),
         metrics.adjusted_rand_score(y_test, y_pred),
         metrics.adjusted_mutual_info_score(y_test, y_pred),
         metrics.silhouette_score(X_test, y_pred, metric='euclidean')))

# Create an isomap and fit the `digits` data to it
X_iso = Isomap(n_neighbors=10).fit_transform(X_train)

# Compute cluster centers and predict cluster index for each sample
clusters = clf.fit_predict(X_train)

# Create a plot with subplots in a grid of 1X2
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Adjust layout
fig.suptitle('Predicted Versus Training Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)

# Add scatterplots to the subplots
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=clusters)
ax[0].set_title('Predicted Training Labels')
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
ax[1].set_title('Actual Training Labels')

# Show the plots
plt.show()

# --------------- Support Vector Machine model --------------------------

# Create the SVC model
svc_model = svm.SVC(gamma=0.001, C=100., kernel='linear')

# Fit the data to the SVC model
svc_model.fit(X_train, y_train)

y_pred = svc_model.predict(X_test)

# Print the classification report of `y_test` and `predicted`
print(metrics.classification_report(y_test, y_pred))

# Print the confusion matrix
print(metrics.confusion_matrix(y_test, y_pred))

# -----------------  Grid search ---------------------------

# Split the `digits` data into two equal sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.5, random_state=0)

# Set the parameter candidates
parameter_candidates = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

# Create a classifier with the parameter candidates
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)

# Train the classifier on training data
clf.fit(X_train, y_train)

# Print out the results
print('Best score for training data:', clf.best_score_)
print('Best `C`:', clf.best_estimator_.C)
print('Best kernel:', clf.best_estimator_.kernel)
print('Best `gamma`:', clf.best_estimator_.gamma)


# Apply the classifier to the test data, and view the accuracy score
clf.score(X_test, y_test)

# Train and score a new classifier with the grid search parameters
svm.SVC(C=10, kernel='rbf', gamma=0.001).fit(X_train, y_train).score(X_test, y_test)
