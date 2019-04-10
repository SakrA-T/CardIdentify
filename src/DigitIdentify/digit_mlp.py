import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import utils

batch_size = 256
num_classes = 11
epochs = 20

(x_train, y_train), (x_test, y_test) = utils.load_data_3()

print('origin')
print('x_train: ', x_train.shape, ' y_train: ', y_train.shape)
print('x_test: ', x_test.shape, 'y_test: ', y_test.shape)

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

print('reshape x')
print('x_train: ', x_train.shape)
print('x_test: ', x_test.shape)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('cate y')
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)

# Test accuracy: 0.97
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

model.save('model_mlp.h5')
print('model saved')

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
