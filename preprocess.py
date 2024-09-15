from skimage.filters import (threshold_otsu, threshold_niblack, threshold_sauvola)
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.spatial.distance import cdist  
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import random
import math
import time
import cv2
import os
import re

def preprocess_image(inp_img, img_res_limit=700):
  h, w = inp_img.shape[:2]

  # Setting the scaling factor for inputting images
  scale_factor = max(h / img_res_limit, w / img_res_limit)

  # If either dimension is greater thean the allowed size, resize.
  if scale_factor > 1:
    new_dim = (int(w / scale_factor), int(h / scale_factor))
    out_img = cv2.resize(inp_img, new_dim, interpolation=cv2.INTER_AREA)

  # If the image is smaller than the limit, return the original image.
  return inp_img