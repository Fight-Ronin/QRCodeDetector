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

def find_unit_vector(pt1, pt2):
    diff_vector = np.subtract(pt1, pt2)
    unit_vector = diff_vector / np.linalg.norm(diff_vector)
    return unit_vector

def calculate_point(main_pt, unit_vector, scale_factor):
    return main_pt + np.multiply(unit_vector, scale_factor)

def qr_bbox_triplet(out_img, finder_triplets, finder_descs):
    sqrt_18 = math.sqrt(18)
    qr_bbox_triplets = []
    
    for triplet in finder_triplets:
        descs = [finder_descs[i] for i in triplet]
        corner_pt, side_pt1, side_pt2 = [np.array(desc[0]) for desc in descs]
        module_sizes = [desc[1] for desc in descs]

        # Calculate main and secondary points for side points
        side_pt1_main = calculate_point(side_pt1, find_unit_vector(side_pt1, side_pt2), module_sizes[1] * sqrt_18)
        side_pt2_main = calculate_point(side_pt2, find_unit_vector(side_pt2, side_pt1), module_sizes[2] * sqrt_18)

        # Calculate displacement vectors and unit vectors for the corner point
        disp_vec = find_unit_vector(corner_pt, side_pt1) + find_unit_vector(corner_pt, side_pt2)
        disp_unit = disp_vec / np.linalg.norm(disp_vec)
        corner_pt_main = calculate_point(corner_pt, disp_unit, module_sizes[0] * sqrt_18)

        # Calculate secondary points for corner adjustments
        side_pt1_second = calculate_point(side_pt1, find_unit_vector(side_pt1, corner_pt), module_sizes[1] * 3)
        side_pt2_second = calculate_point(side_pt2, find_unit_vector(side_pt2, corner_pt), module_sizes[2] * 3)
        corner_pt_pt1 = calculate_point(corner_pt, find_unit_vector(corner_pt, side_pt1), module_sizes[0] * 3)
        corner_pt_pt2 = calculate_point(corner_pt, find_unit_vector(corner_pt, side_pt2), module_sizes[0] * 3)

        # Draw the points
        for pt in [side_pt1_main, side_pt2_main, corner_pt_main]:
            cv2.circle(out_img, tuple(pt.astype(int)), 7, (0, 255, 255), -1)
        for pt in [side_pt1_second, side_pt2_second, corner_pt_pt1, corner_pt_pt2]:
            cv2.circle(out_img, tuple(pt.astype(int)), 1, (255, 0, 255), -1)

        qr_bbox_triplets.append([corner_pt_main, side_pt1_main, side_pt2_main, side_pt1_second, side_pt2_second])

    return qr_bbox_triplets, out_img