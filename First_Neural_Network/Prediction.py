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

Test_labels = np.array(Test_labels)
Test_samples = np.array(Test_samples)
Test_labels, Test_samples = shuffle(Test_labels, Test_samples)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_Test_samples = scaler.fit_transform(Test_samples.reshape(-1,1))

predictions = model.predict(x=scaled_Test_samples, batch_size =10, verbose=0)
for i in predictions:
    print(i)
rounded_predictions = np.argmax(predictions, axis=-1)
for i in rounded_predictions:
    print(i)




