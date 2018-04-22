
import os
import numpy as np
import cv2

import annulus

def imshow(image):
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", image)
    cv2.waitKey()


def draw_annuli(image, annuli):
    if annuli is None:
        return

    for c, e1, e2, _, _ in annuli:
        u = c.round().astype(np.int)
        cv2.rectangle(image, tuple(u - [1,1]), tuple(u + [1, 1]), (0, 255, 0))
        cv2.ellipse(image, e1, (0, 255, 0))
        cv2.ellipse(image, e2, (0, 0, 255))


def draw_numbering(image, H, grid, color):
    if H is None:
        return

    Hinv = np.linalg.inv(H)
    for g in grid:
        x = map_point(Hinv, g).astype(np.int)
        cv2.putText(image, str(g.astype(np.int)), tuple(x), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)


def draw_grid(image, H, grid):
    if H is None:
        return

    Hinv = np.linalg.inv(H)
    for x in grid:
        x1 = map_point(Hinv, x + [-0.5, -0.5]).astype(np.int)
        x2 = map_point(Hinv, x + [ 0.5, -0.5]).astype(np.int)
        x3 = map_point(Hinv, x + [ 0.5,  0.5]).astype(np.int)
        x4 = map_point(Hinv, x + [-0.5,  0.5]).astype(np.int)

        cv2.line(image, tuple(x1), tuple(x2), (255, 0, 0))
        cv2.line(image, tuple(x2), tuple(x3), (255, 0, 0))
        cv2.line(image, tuple(x3), tuple(x4), (255, 0, 0))
        cv2.line(image, tuple(x4), tuple(x1), (255, 0, 0))



def map_point(H, x):
    y = np.dot(H, np.hstack((x, 1)))
    return y[0:2] / y[2]


def process(image):
    image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5, 5))
    #binary = annulus.threshold_image(gray, (64, 64))
    binary = annulus.binarize(gray, 65)

    detector = annulus.AnnulusDetection()
    detector.add_filter(annulus.annuli_shape_filter())
    detector.add_filter(annulus.cross_ratio_filter(inner_circle_diameter = 0.01, outer_circle_diameter = 0.02, tolerance = 0.2))
    detector.add_filter(annulus.neighbor_filter(outer_circle_diameter = 0.02, marker_spacing = 0.03))
    annuli = detector.detect(gray,  binary, high_quality = True)
    points = np.array([m[0] for m in annuli])

    draw_annuli(image, annuli)

    grid = annulus.Grid(outer_circle_diamater = 0.02, marker_spacing = 0.03)
    H, idx, grid, pixel = grid.find_grid(annuli)
    if H is not None:
        draw_grid(image, H, grid)
        

        M = annulus.find_numbering(binary, H, grid)
        if M is not None:
            H, grid = annulus.transformed_homography(M, pixel, grid)
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
            
        draw_numbering(image, H, grid, color)
    
    return image




def single(image_file):
    image = cv2.imread(image_file)
    image = process(image)
    imshow(image)


def video():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1270)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    while True:
        key = cv2.waitKey(10)
        _, image = cam.read()
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite("image.png", image)
    
        try:
            image = process(image)
        except:
            print("error")
            cv2.imwrite("error.png", image)

        cv2.imshow("Image", image);


image_file = os.path.dirname(__file__) + "/../data/image.png"

single(image_file)
#video()


