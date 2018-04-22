import pytest

import annulus
import cv2
import os.path

@pytest.fixture
def image():
    path = os.path.abspath(os.path.dirname(__file__)) + "/../examples/image.png"
    image = cv2.imread(path)
    return image


def test_annulus(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5, 5))
    binary = annulus.binarize(gray, block_size = 65)

    detector = annulus.AnnulusDetection()
    annuli = detector.detect(gray, binary, high_quality = True)
    assert annuli is not None, len(annuli) > 0
        
    grid = annulus.Grid(outer_circle_diamater = 0.02, marker_spacing = 0.03)
    H, idx, grid, pixel = grid.find_numbered_grid(annuli, binary)
    assert H is not None
    

