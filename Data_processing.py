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


