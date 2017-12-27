import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def e2h(x):
    return np.array([x[0], x[1], 1.0])


def h2e(x):
    x = np.array(x)
    return x[:2] / x[2]


def define_lanes_region(n_rows, n_cols, x_from=450, x_to=518, y_lim=317, left_offset=50, right_offset=0):

    vertices = np.array([[
        [x_from, y_lim],
        [x_to, y_lim],
        [n_cols-right_offset, n_rows],
        [left_offset, n_rows],
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


def line_vector_constant_y(val):
    return np.array([0, 1, -val])


def line_vector_from_opencv_points(line):

    x1, y1, x2, y2 = line
    return np.cross([x1, y1, 1], [x2, y2, 1])


def extend_lane_lines(lines, y_const_0, y_const_1):

    n = len(lines)

    res = np.zeros((n, 4), dtype=np.int32)

    line_y0 = line_vector_constant_y(y_const_0)
    line_y1 = line_vector_constant_y(y_const_1)

    for i in range(n):

        line = line_vector_from_opencv_points(lines[i, :])

        intersection_0 = h2e(np.cross(line, line_y0))
        intersection_1 = h2e(np.cross(line, line_y1))

        res[i, :2] = intersection_0
        res[i, 2:] = intersection_1

    return res


def extend_lane_lines_grouped_by_slopes(lines, slopes, y_const_0, y_const_1):

    lines_left = extend_lane_lines(lines[slopes < 0], y_const_0, y_const_1)
    lines_right = extend_lane_lines(lines[slopes > 0], y_const_0, y_const_1)

    return lines_left, lines_right


def draw_lines_on_image(canvas_im, lines, color=[255, 0, 0], thickness=2):

    for i in range(lines.shape[0]):
        x1, y1, x2, y2 = lines[i, :]
        cv2.line(canvas_im, (x1, y1), (x2, y2), color, thickness)


def plot_homogeneous_line_vector(vec, x_from, x_to, **kvargs):

    a, b, c = vec

    def line_func(x):
        return (-a * x - c) / b

    xs = np.arange(x_from, x_to)
    ys = line_func(xs)

    plt.plot(xs, ys, **kvargs)


def visualize_test_images(images, proc_func=lambda im : im):

    plt.figure(figsize=(15, 8))
    for i, im in enumerate(images):
        plt.subplot(2, 3, i+1)
        plt.imshow(proc_func(im))
