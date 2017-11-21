"""
Matplotlib charts.

:author Sam O
"""

import matplotlib.pyplot as plt
epochs = [10, 20, 50, 100]
training_cost = [0.21, 0.0095, 0.0014, 0.00030]
validation_cost = [0.56, 0.25, 0.19, 0.20]
avg_validation_accuracy = [0.78, 0.92, 0.96, 0.95]

plt.plot(epochs, training_cost)
plt.title('Training Cost vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Training Cost')
plt.show()

plt.plot(epochs, validation_cost)
plt.title('Validation Cost vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Validation Cost')
plt.show()

plt.plot(epochs, avg_validation_accuracy)
plt.title('Avg Validation Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Avg Validation Accuracy')
plt.show()
