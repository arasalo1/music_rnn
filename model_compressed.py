import numpy as np
import pypianoroll as pp
import tensorflow as tf
import keras
from Generator_compressed import Generator_compressed
from Parser import Parser
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt

sl = 100
b = 32
print("load training")
training_generator = Generator_compressed('../lpd',file_name="train_names.txt",sequence_length=sl,batch_size=b,subset=0.3)
print("load validation")
validation_generator = Generator_compressed('../lpd_valid',file_name="test_names.txt",sequence_length=sl,batch_size=b)

model = keras.Sequential()
model.add(keras.layers.CuDNNLSTM(150,input_shape=(sl,1),return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.CuDNNLSTM(150))
model.add(keras.layers.Dense(130))
model.add(keras.layers.Dense(1,activation='softmax'))

model.compile(loss='mse',optimizer='adam')

model.fit_generator(generator=training_generator,
					validation_data=validation_generator,
                    use_multiprocessing=True,
                    epochs=1,
                    workers=10)

model.save("compressed.h5")

print("Create a sample")
files = validation_generator.__getitem__(21)
init = files[0][4:5]
le = 300
clip = np.empty([le,1])
for i in range(le):
    print("Predicting: %i"%i,end='\r')
    prediction = model.predict(init)[0,]
    clip[i,] = prediction
    init = np.roll(init,-1,axis=1)
    init[0,-1,] = prediction

print("\nsaving")
np.save('out2.npy',clip)
#midi_file = 'out2.mid'
#pp.Multitrack(tracks=[pp.Track(pianoroll=clip)]).write(midi_file)

#fig,ax = pp.Track(pianoroll=clip).plot()
#fig.savefig("song.png")
#print("finished")
