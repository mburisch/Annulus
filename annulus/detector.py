
import numpy as np
import cv2


def _cross_ratio(z1, z2, z3, z4):
    """Calculate cross ratio between four values on a line"""
    nom = (z3 - z1) * (z4 - z2)
    den = (z3 - z2) * (z4 - z1)
    return nom / den

def _cross_ratio_annulus(center, circle1, circle2):
    """Return a cross ratio for a single annulus.

    z1 z3   z4  z2
    (  (  x  )  )
    """
    p1, p2 = _line_ellipse_intersection(center, [1, 0], circle1)
    p3, p4 = _line_ellipse_intersection(center, [1, 0], circle2)

    # Reorder points if necessary
    if np.linalg.norm(p1 - p3) > np.linalg.norm(p1 - p4):
        p3, p4 = p4, p3

    z1 = -np.linalg.norm(p1 - center)
    z2 = -np.linalg.norm(p3 - center)
    z3 =  np.linalg.norm(p4 - center)
    z4 =  np.linalg.norm(p2 - center)

    return _cross_ratio(z1, z2, z3, z4)


def _cross_ratio_neighbors(center1, center2, circle1, circle2):
    """Cross ratio between two ellipse: 

    The following points are used (center is not used):

    z1    z2    z4     z3
    (  x  )     (  x   )

    The lines are the intersection of the ellipse with a line
    connecting both ellipse centers.
    """
    center1 = np.asarray(center1)
    center2 = np.asarray(center2)
    p1, p2 = _line_ellipse_intersection(center1, center2 - center1, circle1)
    p3, p4 = _line_ellipse_intersection(center1, center2 - center1, circle2)

    # Pick points closer to each other
    if np.linalg.norm(center1 - p3) > np.linalg.norm(center1 - p4):
        p3, p4 = p4, p3
            
    if np.linalg.norm(center2 - p2) > np.linalg.norm(center2 - p1):
        p1, p2 = p2, p1
        
    z1 = 0
    z2 = np.linalg.norm(p2 - p1)
    z3 = np.linalg.norm(p4 - p1)
    z4 = np.linalg.norm(p3 - p1)

    return _cross_ratio(z1, z2, z3, z4)


def _ellipse_to_homogeneous(parameter):
    """Convert from OpenCV ellipse form into A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0"""
    xc    = parameter[0][0]
    yc    = parameter[0][1]
    a     = parameter[1][0] / 2
    b     = parameter[1][1] / 2
    theta = parameter[2] * np.pi / 180
    A = a**2 * np.sin(theta)**2 + b**2 * np.cos(theta)**2
    B = 2 * (b**2 - a**2) * np.sin(theta) * np.cos(theta)
    C = a**2 * np.cos(theta)**2 + b**2 * np.sin(theta)**2
    D = -2 * A * xc - B * yc
    E = -B * xc - 2 * C * yc
    F = A * xc**2 + B * xc * yc + C * yc**2 - a**2 * b**2
    den = F if np.abs(F) > 1e-3 else 1
    return np.array([A, B, C, D, E, F]) / den

       
def _line_ellipse_intersection(x0, xd, ellipse):
    """Calculate intersection of line with ellipse in homogeneous form."""
    x0 = np.hstack((x0, 1.0))
    xd = np.hstack((xd, 0.0))
    xd /= np.linalg.norm(xd)

    C = np.array([[ellipse[0], ellipse[1] / 2, ellipse[3] / 2],
                  [ellipse[1] / 2, ellipse[2], ellipse[4] / 2],
                  [ellipse[3] / 2, ellipse[4] / 2, ellipse[5]]])
    a = np.dot(xd, np.dot(C, xd))
    b = np.dot(x0, np.dot(C, xd)) + np.dot(xd, np.dot(C, x0))
    c = np.dot(x0, np.dot(C, x0))

    v = b**2 - 4 * a *c
    if v < 0 or np.isclose(a, 0):
        return None, None
    

    r1 = (-b + np.sqrt(v)) / (2 * a)
    r2 = (-b - np.sqrt(v)) / (2 * a)
    x1 = x0 + r1 * xd
    x2 = x0 + r2 * xd
    return x1[0:2], x2[0:2]


def map_point(H, point):
    """Performs a homogeneous mapping of a 2D point"""

    x = np.dot(H, np.hstack((point, 1)))
    x = x[0:2] / x[2]
    return x


def map_points(H, pixel):
    """Performs a homogeneous mapping of a list of 2D point"""
    pixel = np.column_stack((pixel, np.ones(len(pixel))))
    x = np.dot(H, pixel.T).T
    x = x[:,0:2] / x[:,2][:,np.newaxis]
    return x

        
def map_ellipse(H, ellipse):
    """Performs a homogeneous mapping of an ellipse center point"""
    points = np.array([e[0] for e in ellipse])
    return map_points(H, points)


def annuli_shape_filter(axis_ratio = 0.2, max_angle = 10 * np.pi / 180, angle_ratio = 1.2):
    """Filter based on the similiraity of the two ellipses of the annulus.
        
    Calculates the ratios betwen large and small ellipse axis and checks if they are similar.
    Compare the angle angle (in rad) between their main axes.

    Args:
        axis_ratio:  Range of maximum allowed difference between axis ratios.
        max_angle:   Angle (in rand) between the main axes.
        angle_ratio: Minimum ratio of large to small ellipse axis to compare axis angles.
    """
    deg = np.pi / 180

    def run(annulus):
        e1 = annulus[0]
        e2 = annulus[1]
        axr1 = e1[1][1] / e1[1][0]
        axr2 = e2[1][1] / e2[1][0]
        ratio = axr1 / axr2

        if ratio < 1 - axis_ratio or ratio > 1 + axis_ratio:
            return False
                
        # Only compare axis if it is a strong ellipse, because otherwise the angle is dominated
        # by noise; especially for a circle the directions of the "main" axis is somewhat random.
        if axr1 > angle_ratio:
            # Angle between two ellipse directions
            angle = np.arccos(np.cos(e1[2] * deg) * np.cos(e2[2] * deg) + np.sin(e1[2] * deg) * np.sin(e2[2] * deg))
            if angle > max_angle:
                return False

        return True

    return lambda annuli: list(filter(run, annuli))


def cross_ratio_filter(inner_circle_diameter, outer_circle_diameter, tolerance = 0.1):
    """Filter annuli based on the cross ratio of the two circles.
        
    Args:
        inner_circle_diameter: Diameter of inner circle
        outer_circle_diameter: Diameter of outer circle
        tolerance:             Tolerance for cross ratio
    """
    annulus_cr = _cross_ratio(-0.5 * outer_circle_diameter, -0.5 * inner_circle_diameter, 0.5 * inner_circle_diameter, 0.5 * outer_circle_diameter)

    def run(annulus):
        center = (np.array(annulus[0][0]) + np.array(annulus[1][0])) / 2
        cr = _cross_ratio_annulus(center, annulus[2], annulus[3])
        return np.isclose(cr, annulus_cr, rtol = tolerance)

    return lambda annuli: list(filter(run, annuli))


def neighbor_filter(outer_circle_diameter, marker_spacing):
    """Filter annuli based on the cross ratio between two annuli
   
    Only allows annuli which have a direct neighbor.

    Args:
        outer_circle_diameter: Diameter of outer circle
        marker_spacing:        Distance between two neighboring annuli
    """
    cr_grid = _cross_ratio(0, outer_circle_diameter, marker_spacing + outer_circle_diameter, marker_spacing)

    def run(annuli):
        result = []
        for i in range(len(annuli)):
            m1 = annuli[i]
            for j in range(len(annuli)):
                if i == j:
                    continue
                m2 = annuli[j]
                cr = _cross_ratio_neighbors(m1[0][0], m2[0][0], m1[2], m2[2])
                if np.isclose(cr, cr_grid, rtol = 0.2):
                    result.append(m1)
                    break
        return result

    return run


class AnnulusDetection(object):
    """Detect annuli in images."""
    
    def __init__(self, **kwargs):
        """Detect ring shaped object bounded by two concentric circles transformed by a homography (camera image)

        All the parameters are optional

        Args:
            minimum_inner_circle_size: Minimum size in pixel of inner circle
            minimum_outer_circle_size: Minimum size in pixel of outer circle
            relative_outer_inner_size: Maximum difference in size between outer and inner circle
            border_distance:           Minimum distance in pixel to image border
            minimum_circle_points:     Minimum number of points for fitting circle
        """

        self.minimum_inner_circle_size = kwargs.pop("minimum_inner_circle_size", 8)  # Minimum size in pixel of inner circle
        self.minimum_outer_circle_size = kwargs.pop("minimum_outer_circle_size", 16) # Minimum size in pixel of outer circle
        self.relative_outer_inner_size = kwargs.pop("relative_outer_inner_size", 4)  # Maximum difference in size between outer and inner circle
        self.border_distance           = kwargs.pop("border_distance", 5)            # Minimum distance in pixel to image border 

        self.minimum_circle_points     = kwargs.pop("minimum_circle_points", 20)     # Minimum number of points for fitting circle

        self.filter = []

        if len(kwargs) > 0:
            raise ValueError("Unknown arguments: {0}".format(list(kwargs.keys())))


    def add_filter(self, f):
        """Add a filter to list."""
        self.filter.append(f)

    def _filter_annuli(self, annuli):
        """Apply filters."""
        for f in self.filter:
            annuli = f(annuli)
        return annuli


    def detect(self, image, binary_image, high_quality = True):
        """Detect annuli in image."

        Args:
            image:        Gray image used for detection
            binary_image: Binary image used for detection
            high_quality: True to detect the annuli using the gray image. Improves quality but more time consuming.

        Returns:
            List of detected annuli
        """
        assert image.shape == binary_image.shape, "Binary image size does not correspond to gray image size"
        
        inv_binary_image = 255 - binary_image

        stats_annulus, stats_background = self._label_image(binary_image, inv_binary_image)
        candidates = self._find_candidates(stats_annulus, stats_background, image.shape)

        annuli, rect = self._approx_annuli(inv_binary_image, candidates)
        annuli = self._filter_annuli(annuli)

        if high_quality and len(annuli) > 0:
            annuli = self._fit_annuli(image, annuli, rect)
                    
        result = self._calculate_center(annuli)

        return result
    

    def _label_image(self, binary_image, inv_binary_image):
        """Detects connected components for foreground and background in binary image."""
        _, label_background, stats_background, _ = cv2.connectedComponentsWithStats(binary_image)
        _, label_annulus,    stats_annulus,    _ = cv2.connectedComponentsWithStats(inv_binary_image)

        return stats_annulus, stats_background


    def _find_candidates(self, annulus_areas, background_areas, image_shape):
        """Find potential candidates. Afterwards its only elimination."""
        def get_background(annulus):
            bg_cand = None
            for background in background_areas[1:, (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA)]:
                # Minimum size of ellipse
                if background[2] < self.minimum_inner_circle_size or background[3] < self.minimum_inner_circle_size:
                    continue

                # Cotained in ellipse
                if background[0] < annulus[0] or background[1] < annulus[1] or background[0] + background[2] > annulus[0] + annulus[2] or background[1] + background[3] > annulus[1] + annulus[3]:
                    continue

                if annulus[2] > self.relative_outer_inner_size * background[2] or annulus[3] > self.relative_outer_inner_size * background[3]:
                    return None

                # Only one candiate allowed, so note the current one and check the rest
                if bg_cand is None:
                    bg_cand = background
                else:
                    # If we found another candidate, discard current annulus
                    return None

            return bg_cand

        candidates = []
        for annulus in annulus_areas[1:, (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA)]:
            # Minimum size of annulus
            if annulus[2] < self.minimum_outer_circle_size or annulus[3] < self.minimum_outer_circle_size:
                continue
            
            # Annulus should not be at border of image
            if annulus[0] <= self.border_distance or annulus[0] + annulus[2] >= image_shape[1] - self.border_distance or annulus[1] <= self.border_distance or annulus[1] + annulus[3] >= image_shape[0] - self.border_distance:
                continue

            background = get_background(annulus)
            if background is not None:
                candidates.append((annulus, background))
        
        return candidates


    def _approx_annuli(self, inv_binary_image, candidates):
        """Fits annuli from binary image."""
        annuli = []
        rect = []
        for annulus, background in candidates:
            x1 = annulus[0] - 1
            x2 = annulus[0] + annulus[2] + 1
            y1 = annulus[1] - 1
            y2 = annulus[1] + annulus[3] + 1
            view = inv_binary_image[y1:y2, x1:x2]
            _, cont, hierachy = cv2.findContours(view, cv2.RETR_CCOMP , cv2.CHAIN_APPROX_NONE, offset = (x1, y1))
            if len(cont) == 2:
                e1 = cv2.fitEllipse(cont[0])
                e2 = cv2.fitEllipse(cont[1])
                h1 = _ellipse_to_homogeneous(e1)
                h2 = _ellipse_to_homogeneous(e2)
                annuli.append((e1, e2, h1, h2))
                rect.append((annulus, background))

        return annuli, rect


    def _fit_annuli(self, image, annuli_list, rect_list):
        """Fits annuli in grayscale image."""
        def find_hq_circle(circle, rect, inner):
            points = self._sample_ellipse(circle, rect)
            dx = 2 * circle[0] * points[:,0] + circle[1] * points[:,1] + circle[3]
            dy = 2 * circle[2] * points[:,1] + circle[1] * points[:,0] + circle[4]
            dir = np.abs(dx) > np.abs(dy)
            xs = np.sign(circle[0] * dx) * inner
            ys = np.sign(circle[2] * dy) * inner
            pt = points.round().astype(np.int)

            x1 = rect[0] - 4
            y1 = rect[1] - 4
            x2 = rect[0] + rect[2] + 5
            y2 = rect[1] + rect[3] + 5

            view = image[y1:y2,x1:x2]
            du = cv2.filter2D(view, cv2.CV_16S, np.array([[-1, 1]]))
            dv = cv2.filter2D(view, cv2.CV_16S, np.array([[-1], [1]]))

            circle_points = []
            for i in range(len(points)):
                if pt[i, 0] < rect[0] - 1 or pt[i, 0] > rect[0] + rect[2] + 1 or pt[i, 1] < rect[1] - 1 or pt[i, 1] > rect[1] + rect[3] + 1:
                    continue

                if dir[i]:
                    g = xs[i] * du[pt[i, 1] - y1, (pt[i,0] - 3 - x1):(pt[i,0] + 4 - x1)]
                else:
                    g = ys[i] * dv[(pt[i,1] - 3 - y1):(pt[i,1] + 4 - y1), pt[i,0] - x1]
                if len(g) == 0:
                    continue
                max_idx = np.argmax(g)
                if max_idx > 0 and max_idx < 5:
                    den = 2 * (g[max_idx - 1] - 2 * g[max_idx] + g[max_idx + 1])
                    p = (g[max_idx - 1] - g[max_idx + 1]) / den
                    p += max_idx - 3
                    if dir[i]:
                        circle_points.append([pt[i,0] + p, pt[i,1]])
                    else:
                        circle_points.append([pt[i,0], pt[i,1] + p])
            
            if len(circle_points) < self.minimum_circle_points:
                return None

            e = cv2.fitEllipse(np.array(circle_points, dtype = np.float32))
            return e

        result = []
        for annulus, rect in zip(annuli_list, rect_list):
            e1 = find_hq_circle(annulus[2], rect[0],  1)
            e2 = find_hq_circle(annulus[3], rect[1], -1)
            if e1 is not None and e2 is not None:
                result.append((e1, e2, _ellipse_to_homogeneous(e1), _ellipse_to_homogeneous(e2)))

        return result


    def _sample_ellipse(self, ellipse, rect):
        """Returns a list of 2D points corresponding to an ellipse equation."""
        points = []

        x = np.arange(rect[0], rect[0] + rect[2])
        y = np.arange(rect[1], rect[1] + rect[3])

        # x coordinates
        a = ellipse[2]
        b = ellipse[1] * x + ellipse[4]
        c = ellipse[0] * x**2 + ellipse[3] * x + ellipse[5]

        s = b**2 - 4 * a * c
        idx = np.nonzero(s >= 0)
        y1 = (-b[idx] + np.sqrt(s[idx])) / (2 * a)
        y2 = (-b[idx] - np.sqrt(s[idx])) / (2 * a)
        x = x[idx]

        # y coordinates
        a = ellipse[0]
        b = ellipse[1] * y + ellipse[3]
        c = ellipse[2] * y**2 + ellipse[4] * y + ellipse[5]

        s = b**2 - 4 * a * c
        idx = np.nonzero(s >= 0)
        x1 = (-b[idx] + np.sqrt(s[idx])) / (2 * a)
        x2 = (-b[idx] - np.sqrt(s[idx])) / (2 * a)
        y = y[idx]

        return np.row_stack((np.column_stack((x, y1)), np.column_stack((x, y2)), np.column_stack((x1, y)), np.column_stack((x1, y))))
    

    def _calculate_center(self, annuli_list):
        """Calculate the centers for the annuli."""
        def calc_center(annulus):
            """Find cross ratio based on 4 points (line-ellipse intersection) and center: p1--p2--c--p3--p4.
               (p1 - p3) * (c - p4) / ((c - p3) * (p1 - p4)) == (p4 - p2) * (c - p1) / ((c - p2) * (p4 - p1))
               Solving this for c yields the center.
            """

            u1 = annulus[0][0]
            u2 = annulus[1][0]
            line_dir = np.subtract(u2, u1)

            if np.isclose(np.linalg.norm(line_dir), 0):
                return u1

            p1, p4 = _line_ellipse_intersection(u1, line_dir, annulus[2])
            p2, p3 = _line_ellipse_intersection(u1, line_dir, annulus[3])
            
            if p1 is None or p2 is None:
                return None

            r1 = 0
            r2 = np.linalg.norm(p2 - p1)
            r3 = np.linalg.norm(p3 - p1)
            r4 = np.linalg.norm(p4 - p1)
            
            line_dir = p4 - p1
            line_dir /= np.linalg.norm(line_dir)

            k = (r1 - r3) / (r2 - r4)

            # Solve for quadratic equation
            a = k - 1
            b = k * (-r2 - r4) - (-r1 - r3)
            c = k * r2 * r4 - r1 * r3

            v = b**2 - 4 * a * c
            if v < 0 or np.isclose(a, 0):
                return None

            c1 = (-b + np.sqrt(v)) / (2 * a)
            c2 = (-b - np.sqrt(v)) / (2 * a)

            if c1 < r2 or c1 > r3:
                r = c2
            else:
                r = c1
            
            return p1 + r * line_dir

        circles = []
        for annulus in annuli_list:
            c = calc_center(annulus)
            if c is not None:
                circles.append((c, annulus[0], annulus[1], annulus[2], annulus[3]))
                
        return circles


class Grid(object):
    """"Grid numbering."""
    def __init__(self, marker_spacing, outer_circle_diamater, **kwargs):
        """Find the numbering of the ellipse on a grid of annuli.

        Args:
            marker_spacing:        Distance between two neighboring markers
            outer_circle_diamater: Diameter of outer circle
            cr_margin:             Relative tolerance of cross ratio between neighboring annuli
            grid_margin:           Relative tolerance for grid projection
            
        """
        self.marker_spacing        = marker_spacing                 # Distance between two neighboring markers
        self.outer_circle_diameter = outer_circle_diamater          # Diameter of outer circle
        self.cr_margin             = kwargs.pop("cr_margin", 0.2)   # Relative tolerance for cross ratio
        self.grid_margin           = kwargs.pop("grid_margin", 0.1) # Relative tolerance for grid projection

        if len(kwargs) > 0:
            raise ValueError("Unknown arguments: {0}".format(list(kwargs.keys())))

    def find_grid(self, ellipse):
        """Find mapping from ellipse to grid positions.
        
        Args:
            ellipse: List of annuli (only the outer ellipse and the annulus center is ever used)

        Returns:
            H:     3x3 homography from pixel to grid coordinates
            idx:   Indices of annuli used for homography
            grid:  Grid coordinate of each used point
            pixel: Pixel coordinate of each used point
        """
        if len(ellipse) < 4:
            return None, None, None, None

        points = np.array([e[0] for e in ellipse])

        H = self._get_initial_homography(ellipse)
        if H is None:
            return None, None, None, None

        H = self._refine_homography(points, H)
        pixel, grid, idx = self._get_grid_points(H, points)

        return H, idx, grid, pixel


    def find_numbered_grid(self, ellipse, binary_image, pattern = None):
        """Find mapping from ellipse to grid positions.

        First uses find_grid() to detect the numbering than find_numbering() to detect
        the pattern and finnaly transforms the homography renumber the grid.
        
        Args:
            ellipse:      List of annuli (only the outer ellipse and the annulus center is ever used)
            binary_image: Binary image to find numbering
            pattern:      Pattern to detect (see find_numbering())

        Returns:
            H:     3x3 homography from pixel to grid coordinates
            idx:   Indices of annuli used for homography
            grid:  Grid coordinate of each used point
            pixel: Pixel coordinate of each used point
        """
        H, idx, grid, pixel = self.find_grid(ellipse)
        if H is None:
            return None, None, None, None
        
        M = find_numbering(binary_image, H, grid, pattern)
        if M is None:
            return None, None, None, None
        
        H, grid = transformed_homography(M, pixel, grid)
        
        return H, idx, grid, pixel


    def _get_initial_homography(self, ellipse):
        """Returns an initial estimate of the homography"""
        candidates = self._get_homography_candidates(ellipse)

        for c in candidates:
            H = self._get_homography_from_candidate(ellipse, c)
            if H is not None:
                return H

        return None


    def _get_homography_candidates(self, ellipse):
        """"Returns a list of potential homography candidates."""
        # The points for the cross ratio are choosen so that
        #cr_grid = 1 - (d/m)**2
        cr_grid = _cross_ratio(0, self.outer_circle_diameter, self.marker_spacing + self.outer_circle_diameter, self.marker_spacing)

        candidates = []
        for i in range(len(ellipse)):
            neighbors = [i]
            for j in range(len(ellipse)):
                if i == j:
                    continue
                cr = _cross_ratio_neighbors(ellipse[i][0], ellipse[j][0], ellipse[i][3], ellipse[j][3])
                if np.isclose(cr, cr_grid, rtol = self.cr_margin):
                    neighbors.append(j)

                if len(neighbors) == 5:
                    candidates.append(neighbors)
                    break

        return candidates


    def _get_homography_from_candidate(self, ellipse, candidate):
        """Find a homography from a point that has four neighbors, i.e. left, right, top and bottom."""

        cr_grid = _cross_ratio(0, self.outer_circle_diameter, 2 * self.marker_spacing + self.outer_circle_diameter, 2 * self.marker_spacing)

        center = candidate[0]
        ax1_1 = candidate[1]
        for i in range(3):
            ax1_2 = candidate[i + 2]
            cr = _cross_ratio_neighbors(ellipse[ax1_1][0], ellipse[ax1_2][0], ellipse[ax1_1][3], ellipse[ax1_2][3])
            if np.isclose(cr, cr_grid, rtol = 0.05):
                s = set((1, 2, 3))
                s.remove(i + 1)
                ax2_1 = candidate[1 + s.pop()]
                ax2_2 = candidate[1 + s.pop()]

                d1 = (ellipse[ax1_2][0] - ellipse[ax1_1][0])
                d2 = (ellipse[ax2_2][0] - ellipse[ax2_1][0])

                # Larger x change is x axis
                if np.abs(d2[0]) > np.abs(d1[0]):
                    ax1_1, ax1_2, ax2_1, ax2_2 = ax2_1, ax2_2, ax1_1, ax1_2
                    d1, d2 = d2, d1

                # x axis is left to right
                if d1[0] < 0:
                    ax1_1, ax1_2 = ax1_2, ax1_1

                # y axis is top to bottom
                if d2[1] < 0:
                    ax2_1, ax2_2 = ax2_2, ax2_1
                        
                # Use this ellipse and its neighbors to calculate homography
                src = np.row_stack([ellipse[center][0], ellipse[ax1_1][0], ellipse[ax1_2][0], ellipse[ax2_1][0], ellipse[ax2_2][0]])
                dst = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])
                H, _ = cv2.findHomography(src, dst)
                return H
                
        return None

    def _refine_homography(self, points, Hest, iter = 2):
        #Refine up to iter times to account for as many markers as possible.
        # This is due to the  initial homography only being estimated from 5 points

        H = Hest

        for i in range(iter):
            # Given a homography, try to recalculate it using more markers
            # Therefore, map marker pixels to world coordinates and if they are close
            # to a potential grid position use it for caclulating the homography.
            pixel, grid, _ = self._get_grid_points(H, points)
            H, _ = cv2.findHomography(pixel, grid)
            if len(pixel) == len(points):
                # Stop if all points have been used
                return H

        return H


    def _get_grid_points(self, H, points):
        """Return the points and grid positions of points which map "nicely" according to H."""
        x = map_points(H, points)
        y = np.round(x)

        idx = np.linalg.norm(x - y, axis = 1) < self.grid_margin
        pixel = points[idx]
        grid = y[idx]

        return pixel, grid, idx
        


def find_numbering(binary_image, H, grid, pattern = None):
    """Finds the given pattern around an annulus in the binary_image.
    
    The default pattern list is:

    *   _   _       0   1   2

    _   x   _       3   4   5

    *   *   *       6   7   8

    x = center of ellipse
    _ = blank
    * = filled
    
    Args:
        H:       Homography from pixel to points
        grid:    Grid numbering
        pattern: Pattern to detect

    Returns:
       Returns a transformation for the points in grid.
    """
    
    Hinv = np.linalg.inv(H)
    coord = np.array([[-0.5, -0.5], [0, -0.5], [0.5, -0.5], [-0.5, 0], [0, 0], [0.5, 0], [-0.5, 0.5], [0, 0.5], [0.5, 0.5]])

    if pattern is None:
        pattern = np.array([True, False, False, False, False, False, True, True, True]).reshape(3, 3)
    
    def check_number(g):
        match = []
        for c in coord:
            p = map_point(Hinv, g + c).astype(np.int)
            if np.all(p >= [0, 0]) and np.all(p[[1, 0]] < binary_image.shape):
                v = binary_image[p[1], p[0]]
            else:
                v = 0
            match.append(v == 0)
            
        match = np.array(match).reshape(3, 3)
        if np.count_nonzero(match) != np.count_nonzero(pattern):
            return None

        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        T = np.array([[1, 0, -g[0]], [0, 1, -g[1]], [0, 0, 1]])

        # Potential match, however, need to consider rotation and mirroring
        for i in range(4):
            m = np.rot90(match, -i)
            if np.all(m == pattern):
                return np.dot(np.diag([1, -1, 1]), np.dot(np.linalg.matrix_power(R, i), T))

        match = np.fliplr(match)
        for i in range(4):
            m = np.rot90(match, -i)
            if np.all(m == pattern):
                return np.dot(np.diag([-1, -1, 1]), np.dot(np.linalg.matrix_power(R, i), T))
            
        return None
    
    candidates = []
    for g in grid:
        M = check_number(g)
        if M is not None:
            candidates.append(M)

    if len(candidates) == 1:
        return candidates[0]
    else:
        return None

def transformed_homography(M, pixel, grid):
    """Returns a homography where each grid point is previously transformed by M.

    Args:
        M:     Homography to apply
        pixel: Pixel coordinates of points
        grid:  Grid coordinates of points to transform by M

    Returns:
        H:    Homography transformed by M
        grid: New numbering of grid
    """
    grid = map_points(M, grid)
    H, _ = cv2.findHomography(pixel, grid)
    return H, grid