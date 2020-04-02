import numpy as np
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
model = Sequential()
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

y_trainOneHot = np_utils.to_categorical(y_train)
y_testOneHot = np_utils.to_categorical(y_test)

X_train2D = X_train.reshape(60000, 28*28).astype('float32')
X_test2D = X_test.reshape(10000, 28*28).astype('float32')

X_train_norm = X_train2D/255
X_test_norm = X_test2D/255

train_history = model.fit(x=X_train_norm, y=y_trainOneHot, validation_split=0.2, epochs=10, batch_size=800, verbose=2)

scores = model.evaluate(X_test_norm, y_testOneHot)

print()
print('測試資料準確度: {:2.1f}%'.format(scores[1]*100.0))

X_input = X_test_norm[0:10,:]
predictions = model.predict_classes(X_input)
print(predictions)

plt.imshow(X_test[4])
plt.show()