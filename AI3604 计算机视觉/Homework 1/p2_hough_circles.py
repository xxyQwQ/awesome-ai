import sys
import cv2
import numpy as np


def convolve(image, kernel):
    rows, cols = image.shape
    size = kernel.shape[0]
    result = np.zeros_like(image)
    image_full = np.pad(image, size // 2)
    for i in range(rows):
        for j in range(cols):
            result[i, j] = np.sum(image_full[i:i+size, j:j+size] * kernel)
    return result


def detect_edges(image):
    """Find edge points in a grayscale image.

    Args:
    - image (2D uint8 array): A grayscale image.

    Return:
    - edge_image (2D float array): A heat map where the intensity at each point
        is proportional to the edge magnitude.
    """
    raw_image = image.astype(float)
    nabla_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    nabla_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
    grad_x = convolve(raw_image, nabla_x)
    grad_y = convolve(raw_image, nabla_y)
    edge_image = np.sqrt(np.square(grad_x) + np.square(grad_y))
    edge_image /= np.max(edge_image)
    return edge_image


def hough_circles(edge_image, edge_thresh, radius_values):
    """Threshold edge image and calculate the Hough transform accumulator array.

    Args:
    - edge_image (2D float array): An H x W heat map where the intensity at each
        point is proportional to the edge magnitude.
    - edge_thresh (float): A threshold on the edge magnitude values.
    - radius_values (1D int array): An array of R possible radius values.

    Return:
    - thresh_edge_image (2D bool array): Thresholded edge image indicating
        whether each pixel is an edge point or not.
    - accum_array (3D int array): Hough transform accumulator array. Should have
        shape R x H x W.
    """
    rows, cols = edge_image.shape
    thresh_edge_image = np.where(edge_image >= edge_thresh, 1, 0).astype(bool)
    accum_array = np.zeros((len(radius_values), rows, cols))
    for i in range(rows):
        for j in range(cols):
            if thresh_edge_image[i, j]:
                for k in range(len(radius_values)):
                    r = radius_values[k]
                    t = np.linspace(0, 2 * np.pi, 360)
                    x = np.round(i + r * np.cos(t)).astype(int)
                    y = np.round(j + r * np.sin(t)).astype(int)
                    p = np.argwhere((0 <= x) & (x < rows) & (0 <= y) & (y < cols))
                    accum_array[k, x[p], y[p]] += 1
    return thresh_edge_image, accum_array


def find_circles(image, accum_array, radius_values, hough_thresh):
    """Find circles in an image using output from Hough transform.

    Args:
    - image (3D uint8 array): An H x W x 3 BGR color image. Here we use the
        original color image instead of its grayscale version so the circles
        can be drawn in color.
    - accum_array (3D int array): Hough transform accumulator array having shape
        R x H x W.
    - radius_values (1D int array): An array of R radius values.
    - hough_thresh (int): A threshold of votes in the accumulator array.

    Return:
    - circles (list of 3-tuples): A list of circle parameters. Each element
        (r, y, x) represents the radius and the center coordinates of a circle
        found by the program.
    - circle_image (3D uint8 array): A copy of the original image with detected
        circles drawn in color.
    """
    rows, cols, _ = image.shape
    circles = []
    circle_image = image.copy()
    for i in range(len(radius_values)):
        for j in range(rows):
            for k in range(cols):
                if accum_array[i, j, k] >= hough_thresh:
                    rmin, rmax = max(0, i - 2), min(len(radius_values), i + 3)
                    xmin, xmax = max(0, j - 2), min(rows, j + 3)
                    ymin, ymax = max(0, k - 2), min(cols, k + 3)
                    if np.max(accum_array[rmin:rmax, xmin:xmax, ymin:ymax]) == accum_array[i, j, k]:
                        circles.append((radius_values[i], j, k))
                        cv2.circle(circle_image, (k, j), radius_values[i], (0, 255, 0), 2)
    return circles, circle_image


def main(argv):
    img_name = argv[0]
    edge_thresh, hough_thresh = float(argv[1]), int(argv[2])
    img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    radius_values = np.arange(20, 41)
    edge_image = detect_edges(gray_image)
    thresh_edge_image, accum_array = hough_circles(edge_image, edge_thresh, radius_values)
    circles, circle_image = find_circles(img, accum_array, radius_values, hough_thresh)
    print(circles)
    
    cv2.imwrite('output/' + img_name + '_gray.png', gray_image)
    cv2.imwrite('output/' + img_name + '_sobel.png', 255 * edge_image)
    cv2.imwrite('output/' + img_name + '_edges.png', 255 * thresh_edge_image)
    cv2.imwrite('output/' + img_name + '_circles.png', circle_image)


if __name__ == '__main__':
    main(sys.argv[1:])
