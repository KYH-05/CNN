#-----------------------------------------------------------------------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
#-----------------------------------------------------------------------------------------------------------------------
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('Cat_and_Dog/training_set',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
test_set = test_datagen.flow_from_directory('Cat_and_Dog/test_set',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
#-----------------------------------------------------------------------------------------------------------------------
model=Sequential()

model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',kernel_initializer='he_normal',input_shape=(64,64,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=400, kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

model.fit(training_set,batch_size=32,epochs=30)

model.evaluate(test_set)


#-----------------------------------------------------------------------------------------------------------------------
#ImageDataGenerator 공부/CNN개선 방식 공부
#모델 저장하고 실제 사진에 대해 테스트해보기
#그래프

#1. 처음 이미지 resize값이 부적절하여 이미지의 내용을 담아내지 못함
#2. 과적합(Dropout이나 이미지 보강, 과하게 복잡한 신경망 등)
#3. 적절하지 못한 하이퍼 파라미터
#4. adam 기법 세부적으로 조정(rate를 올린다든지)
#5. 사실 더 반복하면 됨