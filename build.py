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
files = Parser('../lpd_valid',file_name='test_names.txt',sequence_length=sl,subset=0.002)
init = files.tracks[21].reshape(1,sl,128)
init = (init>0).astype(int)
#print(init)
le = 600
clip = np.empty([600,128])
for i in range(600):
    print("iteration: %i"%i, end='\r')
    prediction = model.predict(init)
    clip[i,] = prediction[0,]
    init = np.roll(init,(0,-1,0))
    init[0,-1,] = prediction

print("\nFinished predicting")
np.save('out2.npy',clip)
midi_file = 'out2.mid'
pp.Multitrack(tracks=[pp.Track(pianoroll=clip)]).write(midi_file)

fig,ax = pp.Track(pianoroll=clip).plot()
fig.savefig("song.png")
