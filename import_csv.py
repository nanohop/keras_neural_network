import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()

model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
  optimizer='adam', 
  loss='binary_crossentropy',
  metrics=['accuracy']
)

data = np.genfromtxt('high_low.csv', delimiter=',')

x_train = data[1:, :4]
y_train = data[1:, 4]


model.fit(
  x_train, 
  y_train,
  epochs=100, 
  validation_split=0.2
)
