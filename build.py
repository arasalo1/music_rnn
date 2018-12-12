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

print("loading model")
model = keras.models.load_model("simple6.h5")
print("model loaded")
validation_generator = validation_generator = Generator('../lpd_valid',file_name="test_names.txt",sequence_length=sl,batch_size=1)
files = validation_generator.__getitem__(21)
init = files[0][4:5]
#init = (init>0).astype(float)
#print(init)
le = 1000
clip = np.empty([le,128])
for i in range(le):
    print("iteration: %i"%i, end='\r')
    prediction = (model.predict(init)[0,]>0.01).astype(float)
    clip[i,] = prediction
    init = np.roll(init,-1,axis=1)
    init[0,-1,] = prediction

print("\nFinished predicting")
np.save('out2.npy',clip)
midi_file = 'out2.mid'
pp.Multitrack(tracks=[pp.Track(pianoroll=clip)]).write(midi_file)

fig,ax = pp.Track(pianoroll=clip).plot()
fig.savefig("song.png")
