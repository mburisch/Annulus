# Annulus

This library detects annuli (donuts) and their center in images which can be used for camera calibration.

The detection is performed in two steps:
- Detection of annuli in the image
- Recovery of the grid

The grid of annuli does not have to be fully visible, i.e. it is fine to capture only a part of the grid,
which helps in calibrating the border and corners of the camera. Optionally the grid can contain numbering
references.

## Install
```
pip install annulus
```

## Example

A basic example
```
image = cv2.imread("image.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binary = annulus.binarize(gray, block_size = 65)

detector = annulus.AnnulusDetection()
annuli = detector.detect(gray, binary, high_quality = True)
grid = annulus.Grid(outer_circle_diamater = 0.02, marker_spacing = 0.03)
H, idx, grid, pixel = grid.find_numbered_grid(annuli, binary)
```

A more complex example
```
# Create detector
import annulus
import cv2

# Grayscale input image
image = cv2.imread("image.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binary version of the image (can also provide own version)
binary = annulus.binarize(gray, block_size = 65)

# Create detector
detector = annulus.AnnulusDetection()

# Optionally add filters for rejecting false positives
detector.add_filter(annulus.annuli_shape_filter())
detector.add_filter(annulus.cross_ratio_filter(inner_circle_diameter = 0.01,
                                               outer_circle_diameter = 0.02,
                                               tolerance = 0.2))
detector.add_filter(annulus.neighbor_filter(outer_circle_diameter = 0.02,
                                            marker_spacing = 0.03))

# Detect the annuli. Uses gray scale and binary image.#
# high_quality = True improves accuracy at the cost of speed
annuli = detector.detect(gray, binary, high_quality = True)

# Find "unnumbered" grid, the parameters define the dimensions of the grid
grid = annulus.Grid(outer_circle_diamater = 0.02, marker_spacing = 0.03)
H, idx, grid, pixel = grid.find_grid(annuli)

# Try to establish a consistent numbering of the grid using reference points
M = annulus.find_numbering(binary, H, grid)
if M is not None:
  # If numbering is found update homography and grid coordinates to reflect new numbering
  H, grid = annulus.transformed_homography(M, pixel, grid)

# Instead of the above can also use:
H, idx, grid, pixel = grid.find_numbered_grid(annuli, binary)
```

### The resulting data is as follows
```
- annuli is a list of tuples:
  (center, outer_ellipse, inner_ellipse, outer_ellipse_homogeneous, inner_ellipse_homogeneous)
             
- outer_ellipse and inner_ellipse are rotated rectangles from OpenCV fitEllipse:
  ((center_x, center_y), (width, height), angle)

- outer_ellipse_homogeneous and inner_ellipse_homogeneous are the 6 parameter of an ellipse equation:
  A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0

- H:     3x3 homography from pixel to grid coordinates
- idx:   Indices of annuli used for homography
- grid:  Grid coordinate of each used point
- pixel: Pixel coordinate of each used point
```
## Result

![Result of detection](https://github.com/mgb4/Annulus/blob/master/doc/result.png)

