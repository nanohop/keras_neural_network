from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# 10.5, 5, 9.5, 12 => 18.5

model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(
  optimizer='adam', 
  loss='mean_squared_error'
)
