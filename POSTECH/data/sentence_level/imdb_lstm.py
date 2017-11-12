from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import pickle

max_features = 10000
maxlen = 6200  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
fi = open('qualvec_dev.txt','r');
ar = pickle.load(fi);
x_test = [];
for i in range(0,1000):
	t = [];
	for j in range(len(ar[i])):
		t.extend(ar[i][j][0]);
	x_test.append(t);

fi = open('qualvec_train.txt','r');
ar = pickle.load(fi);
x_train = [];
for i in range(0,25000):
	t = [];
	for j in range(len(ar[i])):
		t.extend(ar[i][j][0]);
	x_train.append(t);

fi = open('train.hter','r');
fi = fi.read();
fi = fi.split();
y_train = [];
for i in range(0,25000):
	y_train.append(float(fi[i]));


fi = open('train.hter','r');
fi = fi.read();
fi = fi.split();
y_test = [];
for i in range(0,1000):
	y_test.append(float(fi[i]));

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(50,activation='tanh'));
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          validation_data=(x_test, y_test))
score = model.predict_on_batch(x_test)
print('Test score:', score)
#print('Test accuracy:', acc)
