import numpy as np
import pypianoroll
import tensorflow as tf
import keras
from Generator import Generator
from Parser import Parser

print("load training")
training_generator = Generator('../lpd',file_name="train_names.txt",subset=0.5)
print("load validation")
validation_generator = Generator('../lpd_valid',file_name="test_names.txt")

model = keras.Sequential()
model.add(keras.layers.CuDNNLSTM(50,input_shape=(200,128),return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.CuDNNLSTM(50,input_shape=(200,128)))
model.add(keras.layers.Dense(200))
model.add(keras.layers.Dense(128,activation='linear'))

model.compile(loss='mse',optimizer='adam')

model.fit_generator(generator=training_generator,
					validation_data=validation_generator,
                    use_multiprocessing=True,
                    epochs=1,
                    workers=2)


files = Parser('../lpd_valid',file_name='test_names.txt',subset=0.001)
init = files.tracks[100].reshape(1,200,128)
clip = np.empty([200,128])
for i in range(200):
    prediction = model.predict(init)
    clip[i,] = prediction
    init = np.roll(init,(0,-1,0))
    init[0,199,] = prediction

midi_file = 'midi/out2.mid'
pypianoroll.Multitrack(tracks=[pypianoroll.Track(pianoroll=clip)]).write(midi_file)
