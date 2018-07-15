from keras.models import load_model
import numpy as np
from keras.utils.np_utils import to_categorical

model = load_model('iris.h5')

model.summary()

data = np.genfromtxt('iris.csv', delimiter=',')
data = np.random.permutation(data[1:, :])

x_train = data[:, :4]
y_train = to_categorical(data[:, 4])


model.fit(
  x_train, 
  y_train,
  epochs=100, 
  validation_split=0.2
)
