import numpy as np
import music21
import glob
import pypianoroll as pp
import matplotlib.pyplot as plt
import math
import keras

class Generator_compressed(keras.utils.Sequence):

  def __init__(self,root,file_name=None,sequence_length=200,subset=1.0,
                batch_size=32,shuffle=True):
    self.root = root
    self.file_name = file_name
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.sequence_length = sequence_length
    self.subset = subset
    if self.file_name is not None:
      self.files = self.list_files_from_file()
    else:
      self.files = self.list_files()
    if subset != 1.0:
      self.files = self.files[:int(subset*len(self.files))]
    self.samples = 0
    # find total number of samples
    #self.__num_samples()
    self.on_epoch_end()


  def on_epoch_end(self):

    self.indexes = np.arange(len(self.files))
    if self.shuffle:
        np.random.shuffle(self.indexes)


  def __len__(self):
    return int(math.floor(len(self.files)/self.batch_size))

  def __getitem__(self,index):

    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    # Generate data
    X, y = self.__load(indexes)

    return X, y

  def list_files_from_file(self):
    with open(self.file_name,'r') as f:
      files = [line.rstrip('\n') for line in f]
    if self.subset != 1.0:
      return files[:int(self.subset*len(files))]
    else:
      return files

  def list_files(self):
    files = glob.glob(self.root + '/**/*.npz', recursive=True)
    # restrict to subset of files if subset != 1.0
    if self.subset != 1.0:
      return files[:int(self.subset*len(files))]
    else:
      return files

  # load given file
  def __load(self,index_list):
    # take only the first track that is not a drum track
    # TODO maybe consider taking all tracks
    # currently trying to find the melody
    X = list()
    labels = list()
    for i in index_list:
      tmp = pp.load(self.files[i]).tracks
      for j in tmp:
        # be smarter here
        if not j.is_drum and j.name.lower() not in ["bass","bckvocals"]:
          roll = j.pianoroll
          start = 0
          end = self.sequence_length
          r_len = roll.shape[0]
          while(end+1<r_len and roll[start:end].sum() != 0):
            current = roll[start:end]
            current = (128-np.argmax(current[:,::-1],axis=1))-1
            X.append(current.reshape(self.sequence_length,1))
            # we are trying to predict the next note
            label_current = roll[end+1]
            label_current = (128-np.argmax(label_current[::-1]))-1
            labels.append(label_current)
            start = end
            end += self.sequence_length
          break

    return np.array(X), np.array(labels)

  # calculate the number of samples
  def __num_samples(self):
    for i in self.files:
      tmp = pp.load(i).tracks
      for j in tmp:
        # be smarter here
        if not j.is_drum and j.name.lower() not in ["bass","bckvocals"]:
          roll = j.pianoroll
          self.samples += math.floor(roll.shape[0]/self.sequence_length)
          break  
