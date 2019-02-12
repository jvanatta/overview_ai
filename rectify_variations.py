#!/usr/bin/env python
# Python 3
import glob
import os
import sys

import cv2 as cv
import numpy as np

''' Adapted from Kinght's answer here:
    https://stackoverflow.com/questions/8667818/opencv-c-obj-c-detecting-a-sheet-of-paper-square-detection
    Masking in saturation works well for the two base PCB images I tested on, but it's certainly not infallible.
    Consistency of the lighting, background, and PCB color is important. Masking on value also works.
'''
def find_corners(image, channel_threshold, dilate_erode_amount):
    hue_channel, sat_channel, val_channel = cv.split(cv.cvtColor(image, cv.COLOR_BGR2HSV))
    _, threshed = cv.threshold(sat_channel, channel_threshold, 255, cv.THRESH_BINARY)
    # A dilate/erode pass helps remove the holes in the mask from the components on the PCB
    threshed = cv.dilate(threshed, cv.getStructuringElement(cv.MORPH_RECT, (dilate_erode_amount, dilate_erode_amount)))
    threshed = cv.erode(threshed, cv.getStructuringElement(cv.MORPH_RECT, (dilate_erode_amount, dilate_erode_amount)))

    found_contours = sorted(cv.findContours(threshed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2], key=cv.contourArea)
    largest_contour = found_contours[-1]

    arclen = cv.arcLength(largest_contour, True)
    approx = cv.approxPolyDP(largest_contour, 0.02 * arclen, True)

    if len(approx) < 4:
        print("ERROR: Couldn't find enough corner points")

    return approx


''' Sort coordinates found from cv.findContours() into a known order: TL, TR, BR, BL by Euclidean distance.
    The coordinates closest to (0,0) will be selected as the TL point, etc. 
    This isn't completely robust, extreme enough perspective transforms could result in the same point
    getting chosen twice.
'''
def sort_corners(unsorted_corners, image_height, image_width):

    def distance_cost(a_tuple, b_tuple):
        return (a_tuple[0] - b_tuple[0])**2 + (a_tuple[1] - b_tuple[1])**2

    distance_costs = [[], [], [], []]
    for corner in unsorted_corners:
        new_coord = (corner[0][0], corner[0][1])
        distance_costs[0].append(distance_cost((0, 0), new_coord))
        distance_costs[1].append(distance_cost((image_width, 0), new_coord))
        distance_costs[2].append(distance_cost((image_width, image_height), new_coord))
        distance_costs[3].append(distance_cost((0, image_height), new_coord))

    sorted_corners = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        sorted_corners[i, ::] = unsorted_corners[distance_costs[i].index(min(distance_costs[i]))]

    return sorted_corners


''' Detect the corners of a directory of images of colorful planar quadrangles. For example, pictures of rectangular PCBs.
    Warp them to a consistent view. The detected corners of the quadrangles will become the true corners of the new,
    rectangular image.
    
    A clear separation from the background for the majority of the quadrangles is necessary. Detection is done in saturation,
    it could be easily modified to detect based on value or hue.
'''
if __name__ == '__main__':
    # These random values normally generate good output, but there's a possibility a 'perfect storm'
    # will result in the image being transformed beyond the edge of padding.
    # If so, the corner recovery will fail.
    source_files = "variations"
    output_x_size = 600
    output_y_size = 450
    detection_mask_threshold = 80
    dilate_erode_amount = 80

    images_to_process = []
    if os.path.isdir(source_files):
        for filename in glob.glob("{0}/*.jpg".format(source_files)):
            images_to_process.append(filename)
    else:
        images_to_process.append(source_files)

    print("Starting work on {0} image files...".format(len(images_to_process)))

    for image_filename in images_to_process:
        input_image = cv.imread(image_filename)
        if input_image is None:
            sys.exit("Bad input, check your filename: {0}".format(input_image))

        # Detect the corners in the input image, then sort them to the same TL, TR, BR, BL order as the output
        detected_corners = sort_corners(find_corners(input_image, detection_mask_threshold, dilate_erode_amount), input_image.shape[1], input_image.shape[0])
        print("\n \n")
        print(input_image.shape[1], input_image.shape[0])
        print(detected_corners)
        output_corners = np.array([
            [0, 0],
            [output_x_size - 1, 0],
            [output_x_size - 1, output_y_size - 1],
            [0, output_y_size - 1]], dtype="float32")

        # Map the detected PCB corner coordinates to the output image corners
        correction_matrix = cv.getPerspectiveTransform(detected_corners, output_corners)
        corrected_image = cv.warpPerspective(input_image, correction_matrix, (600, 450))

        while True:
            k = cv.waitKey(1)
            # Press escape or q for next image
            if k == 'q' or k == 27 or k == 'Q' or k == 1048603 or k == 1048689:
                break

            for corner in detected_corners:
                cv.circle(input_image, (corner[0], corner[1]), 10, (100, 0, 40), 3)
            cv.imshow("modified", input_image)
            cv.imshow("corrected", corrected_image)

    print("All done!")



