from skimage.filters import (threshold_otsu, threshold_niblack, threshold_sauvola)
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.spatial.distance import cdist  
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import webbrowser
import random
import math
import time
import cv2
import os
import re

from binarization import *
from finder_grouping import *
from finder_localization import *
from preprocess import *
from qr_bbox import *
from qr_triplet import *
from temporal_stability import *
from qr_decoder import *

def process_image(inp_img, preprocess_module, binarization_module, finder_localization_module, finder_pattern_grouper_module, qr_bbox_triplet_module, qr_bbox_module):
    # Pre-process Image
    out_img = preprocess_module(inp_img)

    # Binarize Input Image
    out_img = binarization_module(out_img)

    # Construct the out_matrix
    img_int = (out_img * 255).astype(np.uint8)
    output = np.zeros((img_int.shape[0], img_int.shape[1], 4), dtype=np.uint8)
    output[:, :, 0] = img_int
    output[:, :, 1] = img_int
    output[:, :, 2] = img_int
    output[:, :, 3] = img_int

    # Mark Finder Locations
    finder_locations, debug_img1 = finder_localization_module(out_img)

    if len(finder_locations) < 3:
        return None, output, []

    # Group Finder Locations
    grouped_finder_patterns, debug_img2 = finder_pattern_grouper_module(debug_img1, finder_locations)

    if len(grouped_finder_patterns) == 0:
        return None, output, []

    # Construct 3-point QR bounding boxes
    qr_bbox_triplets, debug_img3 = qr_bbox_triplet_module(debug_img2, grouped_finder_patterns, finder_locations)

    # Find 4th point of QR bounding box
    qr_bbox_out, debug_img4 = qr_bbox_module(out_img, debug_img3, qr_bbox_triplets)

    # Decode and categorize QR codes using the newly integrated function
    decoded_qr_contents = decode_and_categorize_qr_codes(qr_bbox_out, inp_img)


    # Draw out the final QR boxes
    # for bbox in qr_bbox_out:
    #     cv2.polylines(output, [bbox.astype(np.int32)], True, (0, 255, 0, 255), 4)

    return qr_bbox_out, output, decoded_qr_contents

# Config for the system
vid = cv2.VideoCapture(0)
pp_func = partial(preprocess_image, img_res_limit=700)
bin_func = partial(binarization_suvola, ws=85)
finder_func = partial(finder_localization_centroid, aspect_filter=0.3, area_ratio_threshold=(1, 4), centroid_closeness=4)
grouper_func = partial(group_finder_locations, side_tol=0.2, hypot_tol=0.15)
trip_bbox_func = partial(qr_bbox_triplet)
qr_bbox_func = partial(qr_bbox)

# Run the event loop
state_dim = (2 + 2) * 4 # bbox coords, velocities
measurement_dim = 2 * 4 # bbox coords
kalman_filter = initialize_kalman_filter(state_dim, measurement_dim)
step = 0

previous_bboxes = []  # Initialize an empty list to store the previous frame's QR code bounding boxes
displayed_qr_data = set()

while True:
    ret, frame = vid.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # The process_image function now directly returns the decoded QR content along with bounding boxes and processed image
    bboxes, bbox_img, decoded_qr_contents = process_image(gray, pp_func, bin_func, finder_func, grouper_func, trip_bbox_func, qr_bbox_func)

    # Reset bbox_img with the original frame to clear previous drawings
    bbox_img = frame.copy()

    # Handle initial state setup with the Kalman filter
    if step == 0 and bboxes:
        kalman_filter.statePost[:measurement_dim] = bboxes[0].reshape(measurement_dim, 1)[:measurement_dim].astype(np.float32)

    # For subsequent steps, predict and correct Kalman filter state based on new detections
    if step > 0:
        kalman_filter.predict()
        if bboxes:
            kalman_filter.correct(bboxes[0].reshape(measurement_dim, 1)[:measurement_dim].astype(np.float32))
            smoothed_bounding_box = kalman_filter.statePost[:measurement_dim].reshape(-1, 2)
            # Optionally draw smoothed bounding box here if needed

    # Decode QR codes found in the bounding boxes
    # Extract QR code data for the current frame
    current_detected_qr_data = set(content['data'] for content in decoded_qr_contents)

    # Process and display data for each detected QR code in the current frame
    for content in decoded_qr_contents:
        if content['data'] not in displayed_qr_data:
            print(f"Decoded QR Data: {content['data']}")
            displayed_qr_data.add(content['data'])
            # Example action: open URLs
            if content['type'] == 'URL':
                webbrowser.open(content['data'])

    # Display QR code data on the image for every QR code detected in the current frame
    for i, content in enumerate(decoded_qr_contents):
        bbox = bboxes[i]
        x, y, w, h = cv2.boundingRect(bbox.astype(np.int32))
        text_position = (x, y - 10)
        cv2.putText(bbox_img, f"{content['type']}: {content['data']}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Optionally, clear displayed_qr_data if no QR codes are detected to reset the state for new detections
    if not decoded_qr_contents:
        displayed_qr_data.clear()

    # Draw current QR code bounding boxes
    if bboxes:
        for bbox in bboxes:
            cv2.polylines(bbox_img, [bbox.astype(np.int32)], True, (0, 255, 0, 255), 4)

    cv2.imshow('Qrcode Detection Window', bbox_img)
    
    step += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()