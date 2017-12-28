import udacityhelpers as udacity
from lanelines import *
from compgraph import CompGraph, CompGraphRunner


func_dict = {
    'grayscale': udacity.grayscale,
    'get_image_shape': lambda im : im.shape,
    'canny': udacity.canny,
    'define_lanes_region': define_lanes_region,
    'apply_region_mask': apply_region_mask,
    'gaussian_blur': udacity.gaussian_blur,
    'hough_lines': find_hough_lines,
    'compute_line_tangents': compute_line_tangents,
    'extend_lines': extend_lane_lines_grouped_by_slopes,
    'average_endpoints_left': weighted_average_lines_endpoints,
    'average_endpoints_right': weighted_average_lines_endpoints,
    'lines_distances_to_bottom': lines_distances_to_bottom,
    'split_distances_to_bottom': split_distances_to_bottom
}

func_io = {
    'grayscale': ('image', 'image_gray'),
    'get_image_shape': ('image_gray', ('n_rows', 'n_cols')),
    'define_lanes_region': (
        ('n_rows', 'n_cols', 'x_from', 'x_to', 'y_lim', 'left_offset', 'right_offset'),
        'region_vertices'
    ),
    'gaussian_blur': (('image_gray', 'blur_kernel'), 'blurred_image'),
    'canny': (('blurred_image', 'canny_lo', 'canny_hi'), 'image_canny'),
    'apply_region_mask': (('image_canny', 'region_vertices'), 'masked_image'),
    'hough_lines': (('masked_image', 'rho', 'theta', 'hough_threshold', 'min_line_length', 'max_line_gap'), 'lines'),
    'compute_line_tangents': ('lines', 'tangents'),
    'extend_lines': (('lines', 'tangents', 'y_lim', 'n_rows'), ('extended_lines_left', 'extended_lines_right')),
    'average_endpoints_left': (('extended_lines_left', 'dtb_left'), 'avg_line_left'),
    'average_endpoints_right': (('extended_lines_right', 'dtb_right'), 'avg_line_right'),
    'lines_distances_to_bottom': (('lines', 'n_rows'), 'dist_to_bottom'),
    'split_distances_to_bottom': (('dist_to_bottom', 'tangents'), ('dtb_left', 'dtb_right'))

}

cg = CompGraph(func_dict, func_io)

frozen = {
    'x_from': 450,
    'x_to': 518,
    'y_lim': 317,
    'left_offset': 50,
    'right_offset': 0,
    'blur_kernel': 11,
    'canny_lo': 70,
    'canny_hi': 200,
    'rho': 1,
    'theta': np.pi/180,
    'hough_threshold': 20,
    'min_line_length': 7,
    'max_line_gap': 1
}

runner = CompGraphRunner(cg, frozen_tokens=frozen)
