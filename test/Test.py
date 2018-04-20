
import os
import numpy as np
import cv2
import pytest

import annulus

@pytest.fixture()
def image():
    image_file = os.path.dirname(__file__) + "/../data/image.png"
    image = cv2.imread(image_file)
    return image

def test_annulus(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5, 5))
    binary = annulus.binarize(gray, 65)

    detector = annulus.AnnulusDetection()
    detector.add_filter(annulus.annuli_shape_filter())
    detector.add_filter(annulus.cross_ratio_filter(inner_circle_diameter = 0.01, outer_circle_diameter = 0.02, tolerance = 0.2))
    detector.add_filter(annulus.neighbor_filter(outer_circle_diameter = 0.02, marker_spacing = 0.03))
    annuli = detector.detect(gray,  binary, high_quality = True)
    points = np.array([m[0] for m in annuli])

    assert len(points) > 0

    grid = annulus.Grid(outer_circle_diamater = 0.02, marker_spacing = 0.03)
    H, idx, grid, pixel = grid.find_grid(annuli)
    assert H
    if H is not None:
        M = annulus.find_numbering(binary, H, grid)
        assert M
        if M is not None:
            H, grid = annulus.transformed_homography(M, pixel, grid)


