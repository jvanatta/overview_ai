#!/usr/bin/env python
# Python 3
import random

import cv2 as cv
import numpy as np

''' There's some weirdness here if this goes above 75 or so, wherein the hue shifts appear visually discontinuous
'''
def random_hue_shift(image, max_amount):
    hue_channel, sat_channel, val_channel = cv.split(cv.cvtColor(image, cv.COLOR_BGR2HSV))
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


''' max_projective values must be very small to avoid transforming the image far outside its boundaries. There's no
    compensating translation to adjust for that.
'''
def random_distort(image, max_affine, max_projective):
    distortion_array = np.array([
        [1 + random.uniform(-1 * max_affine, max_affine), random.uniform(-1 * max_affine, max_affine), 0],
        [random.uniform(-1 * max_affine, max_affine), 1 + random.uniform(-1 * max_affine, max_affine), 0],
        [random.uniform(-1 * max_projective, max_projective), random.uniform(-1 * max_projective, max_projective), 1]
    ], dtype="float32")

    return cv.warpPerspective(image, distortion_array, (image.shape[0], image.shape[1]), borderMode=cv.BORDER_REPLICATE, flags=cv.WARP_INVERSE_MAP)


if __name__ == '__main__':
    # These random values normally generate good output, but there's a possibility a 'perfect storm'
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

    cv.imwrite("variation.jpg", adjusted_image)
