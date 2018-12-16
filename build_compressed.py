import numpy as np
import pypianoroll as pp
import tensorflow as tf
import keras
from Generator_compressed import Generator_compressed
from Parser import Parser
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys

sl = 100

print("loading model")
model = keras.models.load_model("compressed3.h5")
print("model loaded")
validation_generator = Generator_compressed('../lpd_valid',file_name="test_names.txt",sequence_length=sl,batch_size=1)
init = np.array([])
end = False
index = 0
while not end:
  files = validation_generator.__getitem__(index)
  for i in range(files[0].shape[0]):
    if files[0][i:i+1].size != 0:
      init = files[0][i:i+1]
      end = True
      break
  index += 1

if init.size == 0:
  exit()
#init = (init>0).astype(float)
#print(files[1])
print(np.argmax(init[0,],axis=1))
le = 400
clip = np.empty([le,128])
for i in range(le):
    print("iteration: %i"%i, end='\r')
    #prediction = (model.predict(init)[0,]>0.01).astype(float)
    pred = model.predict(init)[0,]
    prediction = np.zeros(128)
    prediction[np.argmax(pred)] = 1
    #prediction[0] = 0
    prediction[127] = 0
    #print(prediction)
    clip[i,] = prediction
    init = np.copy(np.roll(init,-1,axis=1))
    init[0,-1,] = prediction

print("\nFinished predicting")
np.save('out2.npy',clip)
print(np.argmax(clip,axis=1))
keras.backend.clear_session()
#midi_file = 'out2.mid'
#pp.Multitrack(tracks=[pp.Track(pianoroll=clip)]).write(midi_file)

#fig,ax = pp.Track(pianoroll=clip).plot()
#fig.savefig("song.png")
