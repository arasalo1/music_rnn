import numpy as np
import pypianoroll as pp
import tensorflow as tf
import keras
from Generator import Generator
from Parser import Parser
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt

sl = 100
b = 32
print("load training")
training_generator = Generator('../lpd',file_name="train_names.txt",sequence_length=sl,batch_size=b)
print("load validation")
validation_generator = Generator('../lpd_valid',file_name="test_names.txt",sequence_length=sl,batch_size=b)

model = keras.Sequential()
model.add(keras.layers.CuDNNLSTM(130,input_shape=(sl,128),return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.CuDNNLSTM(130,input_shape=(200,128)))
model.add(keras.layers.Dense(130))
model.add(keras.layers.Dense(128,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

model.fit_generator(generator=training_generator,
					validation_data=validation_generator,
                    use_multiprocessing=True,
                    epochs=1,
                    workers=10)

model.save("simple6.h5")

files = Parser('../lpd_valid',file_name='test_names.txt',sequence_length=sl,subset=0.002)
init = files.tracks[21].reshape(1,sl,128)
init = (init>0).astype(int)
print(init)
clip = np.empty([200,128])
for i in range(200):
    print("Predicting: %i"%i,end='\r')
    prediction = model.predict(init)
    clip[i,] = prediction
    init = np.roll(init,(0,-1,0))
    init[0,-1,] = prediction

print("\nsaving")
np.save('out2.npy',clip)
midi_file = 'out2.mid'
pp.Multitrack(tracks=[pp.Track(pianoroll=clip)]).write(midi_file)

fig,ax = pp.Track(pianoroll=clip).plot()
fig.savefig("song.png")
print("finished")
