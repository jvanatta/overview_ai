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

def random_scale(image, max_amount):
    scale_amount = 1.0 + random.uniform(-1 * max_amount, max_amount)
    new_size = int(round(image.shape[1] * scale_amount)), int(round(image.shape[0] * scale_amount))
    return cv.resize(image, new_size, interpolation=cv.INTER_LINEAR)


def random_rotation(image, output_size, max_amount_deg, center_point):
    rotation_matrix = cv.getRotationMatrix2D(center_point, random.uniform(-1 * max_amount_deg, max_amount_deg), 1)

    return cv.warpAffine(image, rotation_matrix, output_size, borderMode=cv.BORDER_REPLICATE)


def random_distort(image, max_affine, max_projective):
    distortion_array = np.array([
        [1 + random.uniform(-1 * max_affine, max_affine), random.uniform(-1 * max_affine, max_affine), 0],
        [random.uniform(-1 * max_affine, max_affine), 1 + random.uniform(-1 * max_affine, max_affine), 0],
        [random.uniform(-1 * max_projective, max_projective), random.uniform(-1 * max_projective, max_projective), 1]
    ], dtype="float32")

    return cv.warpPerspective(image, distortion_array, (image.shape[0], image.shape[1]), borderMode=cv.BORDER_REPLICATE, flags=cv.WARP_INVERSE_MAP)


def find_corners(image):
    # Adapted from Kinght's answer here:
    # https://stackoverflow.com/questions/8667818/opencv-c-obj-c-detecting-a-sheet-of-paper-square-detection
    # Masking in saturation works well for the two base PCB images I tested on, but it's certainly not infallible.
    # Consistency of the lighting, background, and PCB color is important. Masking on value also works.
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
        print("ERROR: Couldn't find enough corner points")

    return approx


def sort_corners(unsorted_corners, image_height, image_width):
    # This isn't completely robust, extreme enough perspective transforms could result in the same point
    # getting chosen twice
    # Order: TL, TR, BR, BL
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

if __name__ == '__main__':
    # These random values normally generate good output, but there's a possibility a 'perfect storm' alignment
    # will result in the image being transformed beyond the edge of padding.
    # If so, the corner recovery will fail.
    source_filename = "pcb.jpeg"
    max_rotation_deg = 15
    max_hue_shift = 30
    max_scale_factor = .1
    max_affine_warp = .04
    max_projective_warp = .00015

    raw_image = cv.imread(source_filename)

    # Rotating and warping an image require plenty of padding to ensure we don't clip useful parts in the process.
    # Using BORDER_REPLICATE means there are no sudden transitions in the background. That will help with finding
    # the true PCB corners later.
    adjusted_image = cv.copyMakeBorder(raw_image, raw_image.shape[0], raw_image.shape[0], raw_image.shape[1], raw_image.shape[1],
                                     borderType=cv.BORDER_REPLICATE)
    adjusted_image = random_hue_shift(adjusted_image, max_hue_shift)
    adjusted_image = random_scale(adjusted_image, max_scale_factor)
    adjusted_image = random_rotation(adjusted_image, (adjusted_image.shape[0], adjusted_image.shape[1]), max_rotation_deg,
                                     (adjusted_image.shape[1] // 2, adjusted_image.shape[0] // 2))
    adjusted_image = random_distort(adjusted_image, max_affine_warp, max_projective_warp)

    corners = sort_corners(find_corners(adjusted_image), adjusted_image.shape[1], adjusted_image.shape[0])

    output_x_size = 600
    output_y_size = 450

    output_corners = np.array([
        [0, 0],
        [output_x_size - 1, 0],
        [output_x_size - 1, output_y_size - 1],
        [0, output_y_size - 1]], dtype="float32")

    correction_matrix = cv.getPerspectiveTransform(corners, output_corners)
    correct_image = cv.warpPerspective(adjusted_image, correction_matrix, (600, 450))

    while True:
        k = cv.waitKey(1)
        if k == 'q' or k == 27 or k == 'Q' or k == 1048603 or k == 1048689:
            break

        for corner in corners:
            cv.circle(adjusted_image, (corner[0], corner[1]), 10, (100, 0, 40), 3)
        cv.imshow("modified", adjusted_image)
        cv.imshow("corrected", correct_image)
