import numpy as np
import pypianoroll as pp
import tensorflow as tf
import keras
from Generator import Generator
from Parser import Parser
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys

args = sys.argv
index = int(args[1])

sl = 300

print("loading model")
model = keras.models.load_model("simple7.h5")
print("model loaded")
validation_generator = Generator('../lpd_valid',file_name="test_names.txt",sequence_length=sl,batch_size=1)
files = validation_generator.__getitem__(index)
init = files[0][4:5]
#init = (init>0).astype(float)
print(init.shape)
le = 200
clip = np.empty([le,128])
for i in range(le):
    print("iteration: %i"%i, end='\r')
    #print(model.predict(init).shape)
    prediction = (model.predict(init)[0,]>0.01).astype(float)
    #pred = model.predict(init)[0,]
    #print(pred.shape)
    #pred = pred[0,]
    #prediction = np.zeros(128)
    #prediction[np.argmax(pred)] = 1
    #prediction = model.predict(init)[0,]
    #print(prediction)
    clip[i,] = prediction
    init = np.roll(init,-1,axis=1)
    init[0,-1,] = prediction

print("\nFinished predicting")
np.save('out2.npy',clip)
#midi_file = 'out2.mid'
#pp.Multitrack(tracks=[pp.Track(pianoroll=clip)]).write(midi_file)

#fig,ax = pp.Track(pianoroll=clip).plot()
#fig.savefig("song.png")
