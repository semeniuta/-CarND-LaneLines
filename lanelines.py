import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def define_lanes_region(image, x_from=450, x_to=518, y_lim=317, left_offset=50, right_offset=0):

    y_hi, x_hi = image.shape[:2]
    vertices = np.array([[
        [x_from, y_lim],
        [x_to, y_lim],
        [x_hi-right_offset, y_hi],
        [left_offset, y_hi],
    ]], dtype=np.int32)

    return vertices


def apply_region_mask(image, region_vertices):

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, region_vertices, 255)

    return cv2.bitwise_and(image, mask)


def find_hough_lines(im_masked, rho, theta, threshold, min_line_length, max_line_gap):

    lines = cv2.HoughLinesP(im_masked, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines.reshape(lines.shape[0], 4)


def compute_line_tangents(lines):

    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]

    tans = (y2 - y1) / (x2 - x1)

    return tans


def draw_lines_on_image(canvas_im, lines, color=[255, 0, 0], thickness=2):

    for i in range(lines.shape[0]):
        x1, y1, x2, y2 = lines[i, :]
        cv2.line(canvas_im, (x1, y1), (x2, y2), color, thickness)


def visualize_test_images(images, proc_func=lambda im : im):

    plt.figure(figsize=(15, 8))
    for i, im in enumerate(images):
        plt.subplot(2, 3, i+1)
        plt.imshow(proc_func(im))
