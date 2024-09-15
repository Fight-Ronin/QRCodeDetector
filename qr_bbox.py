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

def find_slope_and_intercept(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    if x1 == x2:
        return None, x1  # Vertical line: no slope, x-intercept is x

    slope = (y2 - y1) / (x2 - x1)
    y_intercept = y1 - slope * x1

    return slope, y_intercept

def find_intersection_point(line1, line2):
    slope1, y_intercept1 = find_slope_and_intercept(line1[0], line1[1])
    slope2, y_intercept2 = find_slope_and_intercept(line2[0], line2[1])

    if slope1 is None:  # First line is vertical
        return (y_intercept1, slope2 * y_intercept1 + y_intercept2) if slope2 is not None else None
    if slope2 is None:  # Second line is vertical
        return (y_intercept2, slope1 * y_intercept2 + y_intercept1)
    if slope1 == slope2:  # Parallel lines
        return None

    x_intersection = (y_intercept2 - y_intercept1) / (slope1 - slope2)
    y_intersection = slope1 * x_intersection + y_intercept1

    return (x_intersection, y_intersection)

def qr_bbox(bin_img, out_img, qr_bbox_triplets):
    unordered_bboxes = []
    for triplet in qr_bbox_triplets:
        line1, line2 = [triplet[1], triplet[3]], [triplet[2], triplet[4]]
        fourth_point = find_intersection_point(line1, line2)
        if fourth_point:  # Check for valid intersection
            cv2.circle(out_img, tuple(map(int, fourth_point)), 7, (0, 255, 255), -1)
            unordered_bboxes.append(triplet[:3] + [fourth_point])

    final_bboxes = []
    for bbox in unordered_bboxes:
        center = np.mean(bbox, axis=0)
        cv2.circle(out_img, tuple(map(int, center)), 7, (255, 0, 255), -1)
        angles = [(np.arctan2(pt[0] - center[0], pt[1] - center[1]), pt) for pt in bbox]
        ordered_bbox = np.array([pt for _, pt in sorted(angles)])
        final_bboxes.append(ordered_bbox)

    return final_bboxes, out_img

