
from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np

# 10.5, 5, 9.5, 12 => 18.5

model = Sequential()

model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

x_train = np.array([
  [1, 2, 3, 4],
  [4, 6, 1, 2],
  [10, 9, 10, 11],
  [10, 12, 9, 13],
  [99, 100, 101, 102],
  [105, 111, 109, 102],
  [3, 7, 4, 1],
  [1, 8, 3, 7],
  [12, 15, 11, 9],
  [9, 15, 10, 11],
  [110, 99, 105, 101],
  [97, 101, 100, 105]
])

y_train = np.array([
  [2.5],
  [3.25],
  [10.0],
  [11.0],
  [100.5],
  [106.75],
  [3.75],
  [4.75],
  [11.75],
  [11.25],
  [103.75],
  [100.75]
])

perm = np.random.permutation(y_train.size)
x_train = x_train[perm]
y_train = y_train[perm]

model.fit(
  x_train, 
  y_train,
  epochs=100, 
  batch_size=2, 
  verbose=1,
  validation_split=0.2
)

