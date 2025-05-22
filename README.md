5. Implement the non-parametric Locally Weighted Regression algorithm in order to fit data points.
Select appropriate data set for your experiment and draw graphs.
import numpy as np
import matplotlib.pyplot as plt
def gaussian_kernel(x, x_query, tau):
"""Compute Gaussian weights for Locally Weighted Regression."""
return np.exp(-np.square(x - x_query) / (2 * tau ** 2))
def locally_weighted_regression(X, y, x_query, tau):
"""Perform Locally Weighted Regression at a query point x_query."""
m = X.shape[0]
X_bias = np.c_[np.ones(m), X] # Add bias term
x_query_bias = np.array([1, x_query]) # Bias term for the query point
# Compute weights using Gaussian kernel
W = np.diag(gaussian_kernel(X, x_query, tau))
# Compute theta using weighted least squares: theta = (X'WX)^(-1) X'Wy
theta = np.linalg.pinv(X_bias.T @ W @ X_bias) @ (X_bias.T @ W @ y)
return x_query_bias @ theta # Predicted value at x_query
# Generate synthetic dataset
np.random.seed(42)
X = np.linspace(-3, 3, 100)
y = np.sin(X) + np.random.normal(0, 0.2, X.shape)
# Fit LWR model
tau = 0.5 # Bandwidth parameter
X_test = np.linspace(-3, 3, 200)
y_pred = np.array([locally_weighted_regression(X, y, x, tau) for x in X_test])
# Plot results
plt.scatter(X, y, color='blue', label="Training Data")
plt.plot(X_test, y_pred, color='red', label="LWR Fit (tau=0.5)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Locally Weighted Regression")
plt.show()

7) Develop a program to load the Titanic dataset. Split the data into training and test sets. Train a
decision tree classifier. Visualize the tree structure. Evaluate accuracy, precision, recall, and F1-score.
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,
classification_report
import matplotlib.pyplot as plt
# Step 1: Load the Titanic dataset
df = sns.load_dataset('titanic')
# Step 2: Data preprocessing
# Select relevant features and drop rows with missing values
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
df = df[features + ['survived']].dropna()
# Convert categorical columns to numeric
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
X = df[features]
y = df['survived']
# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 4: Train a Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)
# Step 5: Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=features, class_names=["Not Survived", "Survived"], filled=True)
plt.title("Decision Tree for Titanic Survival")
plt.show()
# Step 6: Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)f1 = f1_score(y_test, y_pred)print("Evaluation Metrics:")print(f"Accuracy : {accuracy:.2f}")print(f"Precision : {precision:.2f}")print(f"Recall : {recall:.2f}")print(f"F1 Score : {f1:.2f}")print("\nClassification Report:\n", classification_report(y_test, y_pred))OUTPUT:Evaluation Metrics:Accuracy : 0.71Precision : 0.72Recall : 0.54F1 Score : 0.62Classification Report:precision recall f1-score support0 0.70 0.84 0.76 801 0.72 0.54 0.62 63accuracy 0.71 143macro avg 0.71 0.69 0.69 143weighted avg 0.71 0.71 0.70 1438)

 Develop a program to implement the Naive Bayesian classifier considering Iris dataset for training.Compute the accuracy of the classifier, considering the test data.PROGRAM:from sklearn.model_selection import train_test_splitfrom sklearn.naive_bayes import GaussianNBfrom sklearn.metrics import accuracy_score, classification_report, confusion_matrix# Load the Iris datasetdf = pd.read_excel("irisdataset.xlsx")X = iris.data # Featuresy = iris.target # Target classes# Split the dataset into training and test sets (80% training, 20% testing)X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)# Initialize the Naive Bayes classifier (GaussianNB for continuous features)model = GaussianNB()# Train the classifiermodel.fit(X_train, y_train)# Predict on test datay_pred = model.predict(X_test)# Calculate accuracyaccuracy = accuracy_score(y_test, y_pred)# Display resultsprint("Predicted Labels:", y_pred)print("Actual Labels :", y_test)print("\nAccuracy of Naive Bayes classifier: {:.2f}%".format(accuracy * 100))print("\nClassification Report:\n", classification_report(y_test, y_pred))print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))OUTPUT:Predicted Labels: [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 2 2 1 1 2 0 2 0 2 2 2 2 2 0 0 0 0 1 0 0 2 10 0 0 2 1 1 0 0 1 1 2 1 2 1 2 1 0 2 1 0 0 0 1]Actual Labels : [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0 0 0 1 0 0 2 10 0 0 2 1 1 0 0 1 2 2 1 2 1 2 1 0 2 1 0 0 0 1]Accuracy of Naive Bayes classifier: 96.67%Classification Report:precision recall f1-score support0 1.00 1.00 1.00 231 0.95 0.95 0.95 192 0.94 0.94 0.94 18accuracy 0.97 60macro avg 0.96 0.96 0.96 60weighted avg 0.97 0.97 0.97 60Confusion Matrix:[[23 0 0][ 0 18 1][ 0 1 17]]9.

 Develop a program to implement k-means clustering using Wisconsin Breast Cancer data set andvisualize the clustering result.import pandas as pdimport numpy as npimport matplotlib.pyplot as pltimport seaborn as snsfrom sklearn.datasets import load_breast_cancerfrom sklearn.cluster import KMeansfrom sklearn.preprocessing import StandardScalerfrom sklearn.decomposition import PCA# Load the datasetdata = load_breast_cancer()df = pd.DataFrame(data.data, columns=data.feature_names)# Standardize the datascaler = StandardScaler()scaled_data = scaler.fit_transform(df)# Apply KMeans clusteringkmeans = KMeans(n_clusters=2, random_state=42)kmeans.fit(scaled_data)labels = kmeans.labels_# Add cluster labels to the original dataframedf['Cluster'] = labels# Reduce dimensions using PCA for visualizationpca = PCA(n_components=2)pca_result = pca.fit_transform(scaled_data)df['PCA1'] = pca_result[:, 0]df['PCA2'] = pca_result[:, 1]# Plot the clustering resultplt.figure(figsize=(10, 6))sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=100)plt.title("K-Means Clustering on Breast Cancer Data (PCA Projection)")plt.xlabel("Principal Component 1")plt.ylabel("Principal Component 2")plt.legend(title="Cluster")plt.grid(True)plt.show()