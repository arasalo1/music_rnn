import numpy as np
import music21
import glob
import pypianoroll as pp
import matplotlib.pyplot as plt

class Parser:

  def __init__(self,root,file_name=None,sequence_length=200,subset=1.0,lazy=False):
    self.root = root
    self.file_name = file_name
    self.sequence_length = sequence_length
    self.subset = subset
    self.lazy = lazy
    if self.file_name is not None:
      self.files = self.list_files_from_file()
    else:
      self.files = self.list_files()
    print("Number of files: %i"%len(self.files))
    if lazy:
      self.tracks = self.__load_lazy()
    else:
      self.tracks = self.__load()

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

  # load samples lazily
  def __load_lazy(self):
    # take only the first track that is not a drum track
    # TODO maybe consider taking all tracks
    # currently trying to find the melody
    for i in self.files:
      tmp = pp.load(i).tracks
      for j in tmp:
        # be smarter here
        if not j.is_drum and j.name.lower() not in ["bass","bckvocals"]:
          roll = j.pianoroll
          start = 0
          end = self.sequence_length
          r_len = roll.shape[0]
          while(end<r_len and roll[start:end] != 0):
            yield roll[start:end]
            start = end
            end += self.sequence_length
          break

  # load samples into memory
  def __load(self):
    size = len(self.files)
    tracks_ = list()
    counter = 1
    for i in self.files:
      print("Loading track %i/%i"%(counter,size),end='\r')
      tmp = pp.load(i).tracks
      for j in tmp:
        # be smarter here
        if not j.is_drum and j.name.lower() not in ["bass","bckvocals"]:
          roll = j.pianoroll
          start = 0
          end = self.sequence_length
          r_len = roll.shape[0]
          while(end<r_len):
            tracks_.append(roll[start:end])
            start = end
            end += self.sequence_length
          break  
      counter += 1
    print("\nLoading finished with %i samples"%(len(tracks_)))
    return tracks_

  # allow listening the piece
  def play(self,index):

    # currently only works if lazy = False
    if self.lazy:
      print("Currently only works for in memory (lazy=False)")
      return

    def midi_play(midi_file):
        mi = music21.midi.MidiFile()
        mi.open(midi_file)
        mi.read()
        mi.close()
        out = music21.midi.translate.midiFileToStream(mi)
        out.show('midi')

    if index>=len(self.tracks):
      index = 0
      print("Index out of range!")
      print("Playing the first song.")   
    
    # create a tmp file
    midi_file = 'midi/tmp.mid'
    pp.Multitrack(tracks=[pp.Track(pianoroll=self.tracks[index])]).write(midi_file)
    midi_play(midi_file)

    
  # visualzie single track sample
  def plot(self,index):

    # currently only works if lazy = False
    if self.lazy:
      print("Currently only works for in memory (lazy=False)")
      return

    if index>=len(self.tracks):
      index = 0
      print("Index out of range!")
      print("Visualizing the first sample.")

    pp.Track(pianoroll=self.tracks[index]).plot()
    plt.show()
