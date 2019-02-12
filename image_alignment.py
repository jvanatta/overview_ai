#!/usr/bin/env python
# Python 3
import random
import sys

import cv2 as cv
import numpy as np

def random_hue_shift(image, max_amount):
    hue_channel, sat_channel, val_channel = cv.split(cv.cvtColor(image, cv.COLOR_BGR2HSV))
    # There's some weirdness here if this goes above 75 or so, wherein the hue shifts appear visually
    # discontinuous.
    shift_amount = random.randrange(0, int(max_amount))
    adjustment = np.ones(hue_channel.shape, dtype=np.uint8) * shift_amount

    return cv.cvtColor(cv.merge([hue_channel + adjustment, sat_channel, val_channel]), cv.COLOR_HSV2BGR)


def random_rotation_matrix(max_amount_deg, center_point):
    return cv.getRotationMatrix2D(center_point, random.uniform(-1 * max_amount_deg, max_amount_deg), 1)


def random_scale(original_size, max_amount):
    scale_amount = 1.0 + random.uniform(-1 * max_amount, max_amount)

    return int(round(original_size[0] * scale_amount)), int(round(original_size[1] * scale_amount))


def random_distort(max_affine, max_projective):
    distortion_array = np.array([
        [1 + random.uniform(-1 * max_affine, max_affine), random.uniform(-1 * max_affine, max_affine), 0],
        [random.uniform(-1 * max_affine, max_affine), 1 + random.uniform(-1 * max_affine, max_affine), 0],
        [random.uniform(-1 * max_projective, max_projective), random.uniform(-1 * max_projective, max_projective), 1]
    ], dtype="float32")

    return distortion_array

def find_corners(image):
    # Adapted from Kinght's answer here:
    # https://stackoverflow.com/questions/8667818/opencv-c-obj-c-detecting-a-sheet-of-paper-square-detection
    dilate_erode_amount = 80
    channel_threshold = 80
    hue_channel, sat_channel, val_channel = cv.split(cv.cvtColor(image, cv.COLOR_BGR2HSV))
    _, threshed = cv.threshold(sat_channel, channel_threshold, 255, cv.THRESH_BINARY)
    threshed = cv.dilate(threshed, cv.getStructuringElement(cv.MORPH_RECT, (dilate_erode_amount, dilate_erode_amount)))
    threshed = cv.erode(threshed, cv.getStructuringElement(cv.MORPH_RECT, (dilate_erode_amount, dilate_erode_amount)))

    found_contours = sorted(cv.findContours(threshed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2], key=cv.contourArea)
    largest_contour = found_contours[-1]

    arclen = cv.arcLength(largest_contour, True)
    approx = cv.approxPolyDP(largest_contour, 0.02 * arclen, True)

    if len(approx) < 4:
        sys.exit("Couldn't find enough corner points")

    def distance_cost(a_tuple, b_tuple):
        return (a_tuple[0] - b_tuple[0])**2 + (a_tuple[1] - b_tuple[1])**2

    # This isn't completely robust, extreme enough perspective transforms could result in the same point
    # getting chosen twice
    # Order: TL, TR, BR, BL

    image_width = image.shape[1]
    image_height = image.shape[0]

    distance_costs = [[], [], [], []]
    for corner in approx:
        new_coord = (corner[0][0], corner[0][1])
        distance_costs[0].append(distance_cost((0, 0), new_coord))
        distance_costs[1].append(distance_cost((image_width, 0), new_coord))
        distance_costs[2].append(distance_cost((image_width, image_height), new_coord))
        distance_costs[3].append(distance_cost((0, image_height), new_coord))

    return_array = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        #print (distance_costs[i])
        #print(approx[distance_costs[i].index(min(distance_costs[i]))])
        print(approx[distance_costs[i].index(min(distance_costs[i]))])
        return_array[i, ::] = approx[distance_costs[i].index(min(distance_costs[i]))]

    print (return_array)
    return return_array

if __name__ == '__main__':
    source_filename = "pcb.jpeg"
    max_rotation_deg = 15
    max_hue_shift = 50
    max_scale_factor = .1
    max_affine_warp = .04
    max_projective_warp = .00015

    raw_image = cv.imread(source_filename)

    resized_image = cv.resize(raw_image, random_scale((raw_image.shape[1], raw_image.shape[0]), max_scale_factor),
                              interpolation=cv.INTER_LINEAR)
    resized_center = (resized_image.shape[0] // 2, resized_image.shape[1] // 2)

    # Rotating and warping an image require plenty of padding to ensure we don't clip useful parts in the process.
    # Using BORDER_REPLICATE means there are no sudden transitions in the background. That will help with finding
    # the true PCB corners later.
    padded_image = cv.copyMakeBorder(resized_image, resized_image.shape[0], resized_image.shape[0], resized_image.shape[1], resized_image.shape[1],
                                     borderType=cv.BORDER_REPLICATE)

    rows, columns, channels = padded_image.shape
    padded_center = (columns // 2, rows // 2)


    # A bit unfortunate to have rotation and warping done separately, since rotations are a subset of affine warps.
    # Homogeneous coordinates everywhere would make the matrix math cleaner.
    modified_image = cv.warpAffine(padded_image, random_rotation_matrix(max_rotation_deg, padded_center), (columns, rows),
                                  borderMode=cv.BORDER_REPLICATE)

    modified_image = cv.warpPerspective(modified_image, random_distort(max_affine_warp, max_projective_warp), (columns, rows), borderMode=cv.BORDER_REPLICATE, flags=cv.WARP_INVERSE_MAP)

    modified_image = random_hue_shift(modified_image, max_hue_shift)

    corners = find_corners(modified_image)

    x_size = 600
    y_size = 450

    output_corners = np.array([
        [0, 0],
        [x_size - 1, 0],
        [x_size - 1, y_size - 1],
        [0, y_size - 1]], dtype="float32")

    print (corners.shape)
    print(output_corners.shape)

    #output_corners = np.array([(0,0), (0, 400), (400, 0), (400, 400)])

    correction_matrix = cv.getPerspectiveTransform(corners, output_corners)
    correct_image = cv.warpPerspective(modified_image, correction_matrix, (600, 450))

    while True:
        k = cv.waitKey(1)
        if k == 'q' or k == 27 or k == 'Q' or k == 1048603 or k == 1048689:
            break

        for corner in corners:
            cv.circle(modified_image, (corner[0], corner[1]), 10, (100, 0, 40), 3)
        cv.imshow("modified", modified_image)
        cv.imshow("corrected", correct_image)
