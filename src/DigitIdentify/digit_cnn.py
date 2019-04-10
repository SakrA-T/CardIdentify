import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import utils

"""
数字识别cnn模型
"""

batch_size = 256
num_classes = 11
epochs = 20

# input image dimensions
img_rows, img_cols = 46, 30

(x_train, y_train), (x_test, y_test) = utils.load_data_3()

print('origin')
print('x_train: ', x_train.shape, ' y_train: ', y_train.shape)
print('x_test: ', x_test.shape, 'y_test: ', y_test.shape)

# channel last default
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('reshape x')
print('x_train: ', x_train.shape)
print('x_test: ', x_test.shape)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('cate y')
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)

# Test accuracy: 0.9983770855873811
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.1)

model.save('model_cnn.h5')
print('model saved')

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])



