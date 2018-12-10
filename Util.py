import glob
import pathlib

root = '/u/16/arasalo1/unix/music/lpd/'
root2 = '/u/16/arasalo1/unix/music/lpd_valid/'

def read(root):
  p = pathlib.Path(root)
  return p.glob('**/*.npz')

print("Creating train names")
with open('train_names.txt','w') as f:
  for i in read(root):
    f.write(str(i)+'\n')

print("Creating test names")
with open('test_names.txt','w') as f:
  for i in read(root2):
    f.write(str(i)+'\n')
