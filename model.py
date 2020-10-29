import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,BatchNormalization,Activation,MaxPooling2D,regularizers
from keras.utils import to_categorical
from skimage.filters import threshold_local

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential() 
model.add(Conv2D(16, (5, 5),input_shape=(28,28,1), padding="same", kernel_initializer="he_normal")) 
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu')) 
model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")) 
model.add(BatchNormalization(axis=-1)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Conv2D(64, (5, 5), kernel_initializer="he_normal")) 
model.add(BatchNormalization(axis=-1)) 
model.add(Activation('relu')) 
model.add(Conv2D(128, (3, 3), kernel_initializer="he_normal")) 
model.add(BatchNormalization(axis=-1)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Conv2D(256, (3, 3), kernel_initializer="he_normal")) 
model.add(BatchNormalization(axis=-1)) 
model.add(Activation('relu')) 
model.add(Conv2D(128, (2, 2), kernel_initializer="he_normal")) 
model.add(BatchNormalization(axis=-1)) 
model.add(Activation('relu')) 
model.add(Conv2D(64, (1, 1), kernel_initializer="he_normal")) 
model.add(BatchNormalization(axis=-1)) 
model.add(Activation('relu')) 
model.add(Conv2D(10, (1, 1), kernel_initializer="he_normal", kernel_regularizer = regularizers.l2(0.01))) 
model.add(Flatten()) 
model.add(Activation('sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy']) 
model.fit(X_train,y_train,epochs=120,validation_data=(X_test,y_test))
model.save('cnn.h5')
