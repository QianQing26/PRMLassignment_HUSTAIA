import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Softmax import SoftmaxClassifier
import matplotlib.pyplot as plt
import optimizer

np.random.seed(26)

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create a Softmax classifier
clf = SoftmaxClassifier()

# Train the classifier on the training set
losses, _ = clf.fit(
    X_train, y_train, optimizer=optimizer.RMSprop(), batch_size=64, num_epochs=100
)

# Predict the labels of the test set
y_pred = clf.predict(X_test)

# Compute the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()
plt.show()
