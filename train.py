# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy
import math
from sklearn.preprocessing import LabelEncoder

# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("matches.csv", skiprows=1, delimiter=',', usecols=(2, 4, 5, 10), dtype='str')
cities = {}
teams = {}
cid = 1
tid = 1
for row in range(len(dataset)):
	if dataset[row][0] not in cities.keys():
		cities[dataset[row][0]] = cid
		cid += 1
	if dataset[row][1] not in teams.keys():
		teams[dataset[row][1]] = tid
		tid += 1
newDataset = numpy.zeros(shape=(len(dataset), 4))
for row in range(len(dataset)):
	for col in range(4):
		if dataset[row][col] in cities.keys():
			newDataset[row][col] = cities[dataset[row][col]]
		else:
			newDataset[row][col] = teams[dataset[row][col]]
print len(teams)

# split into input (X) and output (Y) variables
X = newDataset[:501, 0:3]
Y = newDataset[:501, 3]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y, num_classes = len(teams) + 1)

# create model
model = Sequential()
model.add(Dense(12, input_dim=3, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(len(teams) + 1, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, dummy_y, epochs=5000, batch_size=10)

# evaluate the model
scores = model.evaluate(X, dummy_y)
testX = newDataset[501:, 0:3]
predictions = model.predict(testX)
count = 0
wrong = 0
total = 0
for i in range(len(predictions)):
	new = [int(round(j)) for j in predictions[i].tolist()]
	if 1 in new:
		for j in teams.keys():
			if teams[j] == new.index(1) + 1:
				if j == dataset[501 + i][3]:
					count += 1
	else:
		wrong += 1
	total += 1
print total, count, wrong
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
