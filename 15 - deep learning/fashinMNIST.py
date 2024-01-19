import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Conv2D,Input, Dropout
from keras.models import Model


fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test,y_test) = fashion_mnist.load_data()
x_train,y_test = x_train/255.0, x_test /255.0
print("x_train.shape:", x_train.shape)


x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(x_train.shape)

K = len(set(y_train))
print("number of classes", K)

i = Input(shape = x_train[0].shape[:-1])
x = Conv2D(32, (3,3), strides = 2, activation = 'relu')(i)
x = Conv2D(64, (3,3), strides = 2, activation = 'relu')(x)
x = Conv2D(128,(3,3), strides = 2, activation = 'relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation = 'softmax')(x)

model = Model(i,x)

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)


# PLOT LOSS PER INTERATION
import matplotlib.pyplot as plt
plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'],label='val_loss')
plt.legend()