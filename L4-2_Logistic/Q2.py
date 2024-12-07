from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from LogisitcRegression import logistic
import optimizer as opti

# Generate data
np.random.seed(1919)
X1 = np.random.multivariate_normal([-5, 0], np.eye(2), size=200)
Y1 = np.ones(X1.shape[0])
X2 = np.random.multivariate_normal([0, 5], np.eye(2), size=200)
Y2 = -np.ones(X2.shape[0])
X = np.vstack((X1, X2))
Y = np.hstack((Y1, Y2))

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# define the logistic regression model and optimizer
model = logistic(X_train, Y_train)
optimizer = opti.Adam()

# train the model on the training set
model.train(optimizer=optimizer, batch_size=16, num_epoch=20, plot=True)

# evaluate the model on the testing set
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)
model.visualize_prob(X, Y, show=True)
model.plot2classes(X, Y, show=True)
