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
model.fit(x=scaled_train_samples, y = train_labels, validation_split=0.1, batch_size = 10, epochs = 30, shuffle = True, verbose = 2) #training the model.

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
