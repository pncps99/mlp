from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target

# 6 neurons in the hidden layer
clf = MLP(size=6, lr=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)

epochs = 300

loss = clf.train(X_train, y_train, epochs)

print(f'Accuracy: {accuracy_score(y_test, clf.predict(X_test))*100}%')

plt.plot(np.linspace(1, epochs, epochs), loss, label='Sum of Square Error')
plt.title(f'Evolution of Sum of Squares Error on {epochs} epochs')
plt.xlabel('Epoch')
plt.legend()
plt.show()
