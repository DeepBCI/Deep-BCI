import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten,  MaxPooling2D, Conv2D
from keras.callbacks import TensorBoard


#bindsnet library
https://github.com/BindsNET/bindsnet

(X_train,y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000,28,28,1).astype('float32')
X_test = X_test.reshape(10000,28,28,1).astype('float32')
# 연습데이터 60000개와 테스트 데이터 10000개로 구성됨
#이 mnist데이터의 연습 입력(x)은 47040000(28*28*60000)길이의 1차원 배열임
#이 mnist데이터의 테스트 입력(x)은 7840000(28*28*10000)길이의 1차원 배열임
#cnn을 진행하기 위해 1차원배열을 28*28인 2차원 배열로 바꿔줌

X_train /= 255
X_test /= 255
#배열의 한 칸에는 0~255사이의 숫자로 명도를 표현함
#정규화를 위해 이를 0~1사이의 숫자로 표현

n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)
#라벨링인 y데이터는 60000
#y값에는 0부터 10사이의 숫자가 저장되어있는데 이를 one-hot-coding으로 변환
#ex) keras.utils.to_categorical(3,10)은 [0,0,0,1,0,0,0,0,0,0]를 의미

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)) )
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))
#드롭아웃
model.add(Flatten())
#dense는 1차원만받아서 1차원으로 만들어줌
model.add(Dense(128, activation='relu'))
#128칸1차원배열이 출력, 활성화함수
model.add(Dropout(0.5))

model.add(Dense(n_classes, activation='softmax'))
#결과값dense, 0~1을 표현하기위한 softmax
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=15, verbose=1)
