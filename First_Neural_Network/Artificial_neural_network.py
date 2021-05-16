import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

Train_labels = []
Train_samples = []

for i in range(50):
    # the 5% of younger individuals who did experience side effects
    random_younger = randint(13, 64)
    Train_samples.append(random_younger)
    Train_labels.append(1)

    # the 5% of older individuals who did not experience side effects 
    random_older = randint(65, 100)
    Train_samples.append(random_older)
    Train_labels.append(0)

for i in range(1000):
    # the 95% of younger indviduals who did not experience side effects
    random_younger = randint(13, 64)
    Train_samples.append(random_younger)
    Train_labels.append(0)

    # the 95% of older individuals who did experience the side effects
    random_older = randint(65, 100)
    Train_samples.append(random_older)
    Train_labels.append(1)

# for i in Train_labels:
#     print(i)
# for i in Train_samples:
#     print(i)

#the next step is to take these two lists and processing them

Train_labels = np.array(Train_labels)
Train_samples = np.array(Train_samples)
Train_labels, Train_samples = shuffle(Train_labels, Train_samples)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(Train_samples.reshape(-1,1))

# for i in scaled_train_samples:
#     print(i)


# building an ANN

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
model.summary()

# In this step, we will train the neural network on the data that we have created and processed.
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy']) #preparing the model for training.
model.fit(x=scaled_train_samples, y = Train_labels, validation_split=0.1, batch_size = 10, epochs = 30, shuffle = True, verbose = 2) #training the model.
