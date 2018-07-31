from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

import numpy as np

model = Sequential()


# 0 = Iris-setosa
# 1 = Iris-versicolor
# 2 = Iris-virginica

model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(
  optimizer='adam', 
  loss='categorical_crossentropy',
  metrics=['accuracy']
)

data = np.genfromtxt('iris.csv', delimiter=',')

x_train = data[1:, :4]
y_train = to_categorical(data[1:, 4])

perm = np.random.permutation(y_train.shape[0])
x_train = x_train[perm]
y_train = y_train[perm]

model.fit(
  x_train, 
  y_train,
  epochs=100, 
  validation_split=0.2
)

predict_data = np.array([
  [4.9, 3.0, 1.5, 0.2], #0 Iris-setosa
  [5.7, 3.0, 4.5, 1.2], #1 Iris-versicolor
  [7.2, 3.2, 6.4, 2.3]  #2 Iris-virginica
])

output = model.predict(predict_data)

np.set_printoptions(suppress=True)

print("")
print(output)

output = model.predict_classes(predict_data)

print("")
print(output)
