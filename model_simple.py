import numpy as np
import pypianoroll
import tensorflow as tf
import keras
from Generator import Generator
from Parser import Parser


training_generator = Generator('../lpd')
validation_generator = Generator('../lpd_valid')

model = keras.Sequential()
model.add(keras.layers.LSTM(300,input_shape=(200,128),return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(200,input_shape=(200,128)))
model.add(keras.layers.Dense(200))
model.add(keras.layers.Dense(128,activation='linear'))

model.compile(loss='mse',optimizer='adam')

model.fit_generator(generator=training_generator,
					validation_data=validation_generator,
                    use_multiprocessing=True,
                    epochs=1,
                    workers=2)


files = Parser('lpd',subset=0.001,lazy=False)

init = files.tracks[2100].reshape(1,200,128)
clip = np.empty([200,128])
for i in range(200):
    prediction = model.predict(init)
    clip[i,] = prediction
    init = np.roll(init,(0,-1,0))
    init[0,199,] = prediction

midi_file = 'midi/out.mid'
pypianoroll.Multitrack(tracks=[pypianoroll.Track(pianoroll=clip)]).write(midi_file)