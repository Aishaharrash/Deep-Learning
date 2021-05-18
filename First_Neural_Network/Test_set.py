import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

Test_labels = []
Test_samples = []

for i in range(50):
    # the 5% of younger individuals who did experience side effects
    random_younger = randint(13, 64)
    Test_samples.append(random_younger)
    Test_labels.append(1)

    # the 5% of older individuals who did not experience side effects 
    random_older = randint(65, 100)
    Test_samples.append(random_older)
    Test_labels.append(0)

for i in range(1000):
    # the 95% of younger indviduals who did not experience side effects
    random_younger = randint(13, 64)
    Test_samples.append(random_younger)
    Test_labels.append(0)

    # the 95% of older individuals who did experience the side effects
    random_older = randint(65, 100)
    Test_samples.append(random_older)
    Test_labels.append(1)

# for i in Test_labels:
#     print(i)
# for i in Test_samples:
#     print(i)

#the next step is to take these two lists and processing them

Test_labels = np.array(Test_labels)
Test_samples = np.array(Test_samples)
Test_labels, Test_samples = shuffle(Test_labels, Test_samples)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_Test_samples = scaler.fit_transform(Test_samples.reshape(-1,1))

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Activation, Dense 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.metrics import categorical_crossentropy 

model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

# Here I am going to use the model to predict on test data.
predictions = model.predict(x = scaled_Test_samples, batch_size = 10, verbose =0)
for i in predictions:
    print(i)

rounded_predictions = np.argmax(predictions, axis = -1) 
for i in rounded_predictions:
    print(i)

# Here I will use a confusion matrix to visualize prediction results from a neural network during inference.
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true = Test_labels, y_pred = rounded_predictions)
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    # This function prints and plots the confusion matrix.
    # Normalization can be applied by setting `normalize=True`.

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("confusion matrix, without normalization")
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], 
                 horizontalalignment="center", 
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm_plot_labels = ['no_side_effects', 'had_side_effects']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')













