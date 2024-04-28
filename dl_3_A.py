from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

#https://www.kaggle.com/datasets/noulam/tomato (Link to the dataset)

#Originally New Plant Diseases Dataset(Augmented) contains two folders [train & valid] each containing 10 directories i.e. 10 classes 
# 1. Tomato___Bacterial_spot 
# 2. Tomato___Early_blight 
# 3. Tomato___Late_blight 
# 4. Tomato___Leaf_Mold 
# 5. Tomato___Septoria_leaf_spot 
# 6. Tomato___Spider_mites Two-spotted_spider_mite 
# 7. Tomato___Target_Spot 
# 8. Tomato___Tomato_Yellow_Leaf_Curl_Virus 
# 9. Tomato___Tomato_mosaic_virus 
# 10. Tomato___healthy [Train - 18345 Total Images, Valid - 4585 Total Images]

#But for training & validating we have taken only three classes namely 
# 1. Tomato___Bacterial_spot 
# 2. Tomato___Early_blight 
# 3. Tomato___healthy each class containing 200 images i.e. total images - 600 [Train - 600 Total Images, Valid - 600 Total Images]

#Path to the New Plant Diseases Dataset(Augmented)
train_dir = r'D:\DL Practical\New Plant Diseases Dataset(Augmented)\train'
val_dir = r'D:\DL Practical\New Plant Diseases Dataset(Augmented)\valid'

img_size = 224
batch_size = 32

"""Preprocessing"""

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_size, img_size),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(val_dir,
                                                target_size=(img_size, img_size),
                                                batch_size=batch_size,
                                                class_mode='categorical')

list(train_generator.class_indices)

"""Building our Model"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

model = Sequential()

model.add((Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3))))
model.add(BatchNormalization())
model.add((MaxPooling2D(2,2)))
model.add((Conv2D(64, (3,3), activation='relu')))
model.add(BatchNormalization())
model.add((MaxPooling2D(2,2)))
model.add((Conv2D(64, (3,3), activation='relu')))
model.add(BatchNormalization())
model.add((MaxPooling2D(2,2)))
model.add((Conv2D(128, (3,3), activation='relu')))
model.add(BatchNormalization())
model.add((MaxPooling2D(2,2)))

model.add((Flatten()))

model.add((Dense(128, activation='relu')))
model.add((Dropout(0.2)))
model.add((Dense(64, activation='relu')))
model.add((Dense(train_generator.num_classes, activation='softmax')))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

"""Training our Model"""

model.fit(train_generator, epochs=30, validation_data=val_generator)

"""Evaluating our Model"""

loss, accuracy = model.evaluate(val_generator)
print("Loss :",loss)
print("Accuracy (Test Data) :",accuracy*100)

"""Testing our Model"""

import numpy as np
img_path =r'D:\DL Practical\New Plant Diseases Dataset(Augmented)\valid\Tomato___Early_blight\28d03063-a772-4136-80fd-3bbff0fffa41___RS_Erly.B 7370.JPG'
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

prediction = model.predict(img_array)
class_names=['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy']

predicted_class = np.argmax(prediction)
print(prediction)
print(predicted_class)
print('Predicted class:', class_names[predicted_class])