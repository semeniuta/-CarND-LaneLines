# **Finding Lane Lines on the Road**

## Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:

* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[Pipeline]: ./pipeline.png "Pipeline"

[solidWhiteCurve]: ./test_images_output/solidWhiteCurve.jpg "solidWhiteCurve"

[solidWhiteRight]: ./test_images_output/solidWhiteRight.jpg "solidWhiteCurve"

[solidYellowCurve]: ./test_images_output/solidYellowCurve.jpg "solidWhiteCurve"

[solidYellowCurve2]: ./test_images_output/solidYellowCurve2.jpg "solidWhiteCurve"

[solidYellowLeft]: ./test_images_output/solidYellowLeft.jpg "solidWhiteCurve"

[whiteCarLaneSwitch]: ./test_images_output/whiteCarLaneSwitch.jpg "solidWhiteCurve"

---

### System setup

The project uses the [carnd-term1](https://github.com/udacity/CarND-Term1-Starter-Kit) conda environment with additionally installed Graphviz and nxpd. These components can be installed via:

```
conda install graphviz
pip install graphviz pygraphviz nxpd
```

### Pipeline description

The pipeline for lane detection was built upon the computational graph implementation from the [EPypes project](https://github.com/semeniuta/EPypes). The idea is that you create a library of Python functions for certain application-specific computations, and later declaratively combine them into a directed acyclic graph. The latter can be run once its topological sort is performed. The graph can also be visualized with Graphviz. It is bipartite: the functions are shown as rectangles, and the data tokens -- as ellipses. The shaded ellipses correspond to *frozen* tokens, i.e. source tokens with fixed values. They represent tuned parameters of a vision algorithm.

The core functions used in the developed pipeline reside in the `lanelines.py` Python module, while the original Udacity helper functions are decomposed into `udacityhelpers.py`. The computational graph for the pipeline is defined in `lanespipeline.py`. Its visualization is shown below:

![alt text][Pipeline]

The original RGB image is grayscaled, and smoothed with a Gaussian kernel. A region mask is applied to the smoothed image, where the region of interest polygon is defined for a particular camera setup (`define_lanes_region`). Having a masked image, Hough transform is applied for find line segments, which are later extrapolated to the top and bottom of the region of interest (`extend_lines`). When extrapolating, line segments with too small slope (hence, close to horizontal) are eliminated. The extrapolated line segments are grouped by their slopes into those corresponding to the left lane and the right lane. Both groups are then averaged to return a single pair of line segments that can be drawn on the test images and video frames.

Dealing with the computational graph and its integration into media files generations is simplified with the `LaneFinder` class and the generated `find_and_draw_lanes` closure function (both in `lanefinder.py`).

The result of applying the developed lane detection pipeline to the test images is shown below:

![alt text][solidWhiteCurve]

![alt text][solidWhiteRight]

![alt text][solidYellowCurve]

![alt text][solidYellowCurve2]

![alt text][solidYellowLeft]

![alt text][whiteCarLaneSwitch]

The videos:

<video width="960" height="540" controls>
  <source src="./test_videos_output/solidYellowLeft.mp4">
</video>


### Pipeline shortcomings


### Further work
