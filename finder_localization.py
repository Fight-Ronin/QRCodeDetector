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

# Finder Localization Methods
def finder_localization_centroid(img, aspect_filter=0.3, area_ratio_threshold=(1, 4), centroid_closeness=4):
    # Ensure image is 8-bit integer
    img_int = np.uint8(img * 255)
    output = cv2.connectedComponentsWithStats(255 - img_int, 4, cv2.CV_32S)
    numLabels, labelled_img, stats, centroids = output

    # Initialize the output image
    output_img = np.stack([img_int] * 3, axis=-1)

    # Pre-filtering based on area, aspect ratio
    valid_labels = [i for i in range(1, numLabels) if stats[i, cv2.CC_STAT_AREA] >= 9]
    aspect_ratios = stats[valid_labels, cv2.CC_STAT_WIDTH] / stats[valid_labels, cv2.CC_STAT_HEIGHT]
    valid_aspect = (aspect_ratios >= (1 - aspect_filter)) & (aspect_ratios <= (1 + aspect_filter))
    valid_labels = np.array(valid_labels)[valid_aspect]

    # Prepare valid_components data structure more efficiently
    valid_components = [[*centroids[i], stats[i, cv2.CC_STAT_AREA],
                         [stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                          stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH],
                          stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]], i]
                        for i in valid_labels]

    # Sort valid components by centroid
    valid_components.sort(key=lambda x: (x[0], x[1]))

    # Finder Pattern Matching
    possible_finder_descriptors = []
    for i in range(len(valid_components) - 1):
        for j in range(i + 1, len(valid_components)):
            big, small = (i, j) if valid_components[i][2] > valid_components[j][2] else (j, i)
            if not validate_component_pair(valid_components[big], valid_components[small], area_ratio_threshold, centroid_closeness):
                continue
            draw_finder_patterns(output_img, valid_components[big], valid_components[small])
            possible_finder_descriptors.append((valid_components[big], valid_components[small]))

    # Calculate new centroids and module width for possible finder patterns
    output_finder_descriptors = calculate_new_centroids(labelled_img, possible_finder_descriptors, output_img)

    return output_finder_descriptors, output_img

def validate_component_pair(big_desc, small_desc, area_ratio_threshold, centroid_closeness):
    # Validate based on area ratio and bounding box containment
    area_ratio = big_desc[2] / small_desc[2]
    if not (area_ratio_threshold[0] < area_ratio <= area_ratio_threshold[1]):
        return False
    if (big_desc[3][0] >= small_desc[3][0] or big_desc[3][1] >= small_desc[3][1] or
            big_desc[3][2] <= small_desc[3][2] or big_desc[3][3] <= small_desc[3][3]):
        return False
    # Validate centroid closeness
    if math.sqrt((big_desc[0] - small_desc[0]) ** 2 + (big_desc[1] - small_desc[1]) ** 2) >= centroid_closeness:
        return False
    return True

def draw_finder_patterns(output_img, big_desc, small_desc):
    cv2.rectangle(output_img, (big_desc[3][0], big_desc[3][1]), (big_desc[3][2], big_desc[3][3]), (255, 0, 0), 5)
    cv2.rectangle(output_img, (small_desc[3][0], small_desc[3][1]), (small_desc[3][2], small_desc[3][3]), (255, 0, 0), 5)

def calculate_new_centroids(labelled_img, possible_finder_descriptors, output_img):
    output_finder_descriptors = []
    for b_desc, s_desc in possible_finder_descriptors:
        try:
            b_obj_img = np.uint8(labelled_img == b_desc[4]) * 255
            s_obj_img = np.uint8(labelled_img == s_desc[4]) * 255

            b_M, s_M = cv2.moments(b_obj_img), cv2.moments(s_obj_img)
            centroid = [(b_M["m10"] + s_M["m10"]) / (b_M["m00"] + s_M["m00"]),
                        (b_M["m01"] + s_M["m01"]) / (b_M["m00"] + s_M["m00"])]
            module_width = math.sqrt((b_desc[2] + s_desc[2]) / 33)

            cv2.circle(output_img, [int(x) for x in centroid], 5, (0, 255, 0), -1)
            output_finder_descriptors.append((centroid, module_width, b_desc, s_desc))
        except ZeroDivisionError:
            continue
    return output_finder_descriptors