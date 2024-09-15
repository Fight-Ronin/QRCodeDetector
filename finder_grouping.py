from skimage.filters import (threshold_otsu, threshold_niblack, threshold_sauvola)
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.spatial.distance import cdist  
import matplotlib.pyplot as plt
from functools import partial
from itertools import combinations
import numpy as np
import random
import math
import time
import cv2
import os
import re

def group_finder_locations(out_img, finder_descs, side_tol=0.05, hypot_tol=0.05):
    centroids = [(desc[0][0], desc[0][1]) for desc in finder_descs]
    distance_matrix = cdist(centroids, centroids)

    pairings = []

    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            for k in range(j + 1, len(centroids)):
                # Calculate distances and identify the hypotenuse
                sides = [distance_matrix[i][j], distance_matrix[j][k], distance_matrix[i][k]]
                max_idx = np.argmax(sides)
                sides_idx = [0, 1, 2]
                sides_idx.remove(max_idx)

                hypot = sides[max_idx]
                sideA, sideB = sides[sides_idx[0]], sides[sides_idx[1]]

                # Check if it forms a right-angled triangle within tolerances
                if not(abs(sideA - sideB) < side_tol * max(sideA, sideB) and
                       abs(math.sqrt(sideA**2 + sideB**2) - hypot) < hypot_tol * hypot):
                    continue

                # Reconstruct the pairing according to their roles in the triangle
                side_opp_point_idx = [k, i, j]
                pairing = (side_opp_point_idx[max_idx], side_opp_point_idx[sides_idx[0]], side_opp_point_idx[sides_idx[1]])

                pairings.append(pairing)

                # Debug drawing
                cv2.line(out_img, tuple(map(int, centroids[i])), tuple(map(int, centroids[j])), (0, 0, 255), 3)
                cv2.line(out_img, tuple(map(int, centroids[j])), tuple(map(int, centroids[k])), (0, 0, 255), 3)
                cv2.line(out_img, tuple(map(int, centroids[i])), tuple(map(int, centroids[k])), (0, 0, 255), 3)

    return pairings, out_img