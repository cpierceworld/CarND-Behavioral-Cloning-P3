import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

# Read in training meta-data from CSV file
lines = []
with open("./data/driving_log.csv") as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		lines.append(line)

# Load training images and steering angle ground truth
images = []
measurements = []
for line_cnt, line in enumerate(lines):
	filename = line[0]
	current_path = './data/' + filename
	image = np.array(cv2.imread(current_path))
	if line_cnt % 10:
		# lower brightness on every 10th sample
		image = image - 50
		image[image < 0] = 0

	images.append(image)

	# used flipped image to double traing size
	images.append(np.fliplr(image))

	measurement = float(line[3])
	measurements.append(measurement)
	measurements.append(-1.0 * measurement)

# Conver training data to numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)

# Shuffle training data
X_train, y_train = shuffle(X_train, y_train)

#Import Keras related modules
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras import backend as K

# Define the model: 4 convolution layes, 4 fully connected layers.
# Use Dropout between layers to counteract overfitting.
model = Sequential()
model.add(Cropping2D(cropping=((30,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(24, (5,5), strides=(2, 2), activation="relu"))
model.add(Dropout(0.4))
model.add(Conv2D(36, (5,5), strides=(2, 2), activation="relu"))
model.add(Dropout(0.4))
model.add(Conv2D(48, (5,5), strides=(2, 2), activation="relu"))
model.add(Dropout(0.4))
model.add(Conv2D(64, (3,3), strides=(1, 1), activation="relu"))
#model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), strides=(2, 1), activation="relu"))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(25, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))

# Use "mean square error" for loss function and "Adam" optimizer
model.compile(loss='mse', optimizer='adam')

# Train the model
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=20)

# Save the model
model.save('model.h5')
K.clear_session()
