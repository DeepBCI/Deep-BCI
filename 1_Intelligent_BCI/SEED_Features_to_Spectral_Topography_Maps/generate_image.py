import numpy as np
from utils import cart2sph, pol2cart, gen_images
import math as m
import dataset as db
import scipy.misc
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2

def azim_proj(pos):

  [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
  return pol2cart(float(az), m.pi/2.0 - float(elev))


if __name__ == '__main__':
  interpolation = 'cubic' # 'linear', etc.
  f = open('seed_locs_3d.txt') # 3D Coordinates
  lines = f.readlines()
  locs_3d = [l[:-1].split() for l in lines]
  locs_3d = [(float(elm[0]), float(elm[1]), float(elm[2])) for elm in locs_3d]
  locs_2d = []
  for e in locs_3d:
    locs_2d.append(azim_proj(e))
  locs_2d = np.array(locs_2d)


  f = open('filenames.txt') # Name of .mat files
  dname = f.readlines()
  dname = [elm[:-1] for elm in dname]
  for i in range(len(dname)):
    images = []
    data = sio.loadmat('SEED/Features/'+dname[i]+'.mat') # features

    for c in range(1,15+1): # 15 Clips
      cdata = data['psd_LDS'+str(c)] # For SEED features
      for j in range(cdata.shape[1]):
        image = gen_images(locs_2d, cdata[:,j,:], 32, interpolation=interpolation)
        images.append(image)

    save_np = images # Reshape, Data Split here.
    np.save('save.npy',save_np)

