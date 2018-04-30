import csv
import cv2
import numpy as np
import sklearn

# read log data
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from random import shuffle
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# load image from row/index
def load_image(index, sample):
    return cv2.imread('data/IMG/' + sample[index].split('/')[-1])

# flip image
def flip_input(image, angle):
    processed_image = cv2.flip(image,1)
    processed_angle = angle*-1.0
    return (processed_image, processed_angle)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            correction = 0.2
            for batch_sample in batch_samples:

                # load center image / angle
                center_image = load_image(0, batch_sample) 
                center_angle = float(batch_sample[3])
                # flip center image / angle
                center_flipped = flip_input(center_image, center_angle)
                images.extend([center_image, center_flipped[0]])
                angles.extend([center_angle, center_flipped[1]])
              
                # load left image / angle
                left_image = load_image(1, batch_sample)
                left_angle = center_angle + correction
                # flip left image /angle 
                left_flipped = flip_input(left_image, left_angle)
                images.extend([left_image, left_flipped[0]])
                angles.extend([left_angle, left_flipped[1]])

                # load right image / angle
                right_image = load_image(2, batch_sample)
                right_angle = center_angle - correction
                # load right image / angle
                right_flipped = flip_input(right_image, right_angle)
                images.extend([right_image, right_flipped[0]])
                angles.extend([right_angle, right_flipped[1]])

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Conv2D, MaxPooling2D

keep_prob = 0.5
num_epochs = 5

model = Sequential()
# Normalize image
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# Crop image
model.add(Cropping2D(cropping=((70,25),(0,0))))
# Convolution 5x5 Layers
model.add(Conv2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(48,5,5,subsample=(2,2),activation='relu'))
# Convolution 3x3 Layers
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
model.add(Dropout(keep_prob))
model.add(Flatten())
# Full-Connected Layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

from keras.models import Model
import matplotlib.pyplot as plt

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=num_epochs, verbose=1)
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
