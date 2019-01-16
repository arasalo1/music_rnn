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
training_generator = Generator_compressed('../lpd',file_name="train_names.txt",sequence_length=sl,batch_size=b)
print("load validation")
validation_generator = Generator_compressed('../lpd_valid',file_name="test_names.txt",sequence_length=sl,batch_size=b)

model = keras.Sequential()
model.add(keras.layers.CuDNNLSTM(128,input_shape=(sl,128),return_sequences=True))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.CuDNNLSTM(512,return_sequences=True))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.CuDNNLSTM(256))
model.add(keras.layers.Dense(256,activation='elu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(128,activation='elu'))
model.add(keras.layers.Dense(128,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam')

history = model.fit_generator(generator=training_generator,
					validation_data=validation_generator,
                    use_multiprocessing=True,
                    epochs=15,
                    workers=10)

model.save("compressed7.h5")

fig,ax = plt.subplots()
ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
ax.set_title('Loss')
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
ax.legend(['train', 'test'], loc='upper right')
fig.savefig("loss2.png")

print("Create a sample")
files = validation_generator.__getitem__(21)
init = files[0][4:5]
le = 300
clip = np.empty([le,128])
for i in range(le):
    print("Predicting: %i"%i,end='\r')
    pred = model.predict(init)[0,]
    prediction = np.zeros(128)
    prediction[np.argmax(pred)] = 1
    #prediction[0] = 0
    prediction[127] = 0
    clip[i,] = prediction
    init = np.roll(init,-1,axis=1)
    init[0,-1,] = prediction

print("\nsaving")
np.save('out2.npy',clip)
keras.backend.clear_session()
#midi_file = 'out2.mid'
#pp.Multitrack(tracks=[pp.Track(pianoroll=clip)]).write(midi_file)

#fig,ax = pp.Track(pianoroll=clip).plot()
#fig.savefig("song.png")
#print("finished")
