import cv2
import glob
import numpy as np


def read_images(image_directory):
    # Read all jpg images from the specified directory
    return [cv2.imread(image_path) for image_path in glob.glob("{}/*.jpg".format(image_directory))]


def find_image_points(images, pattern_size):
    world_points = []
    image_points = []
    
    # Initialize the chessboard world coordinate points
    def init_world_points(pattern_size):
        width, height = pattern_size
        return np.mgrid[0:width, 0:height].T.reshape(-1, 2).astype(np.float32)
    
    # Detect chessboard corners in each image
    def detect_corners(image, pattern_size):
        found, corners = cv2.findChessboardCorners(image, pattern_size, None, cv2.CALIB_CB_ADAPTIVE_THRESH)
        return corners.reshape(-1, 2) if found else None

    # Complete the loop below to obtain the corners of each image and the corresponding world coordinate points
    for image in images:
        corners = detect_corners(image, pattern_size)
        if corners is not None:
            image_points.append(corners)
            world_points.append(init_world_points(pattern_size))
    
    return world_points, image_points


def calibrate_camera(world_points, image_points, image_size):
    assert len(world_points) == len(image_points), "The number of world coordinates and image coordinates must match"

    # Initialize the matrix G
    m = len(world_points)
    G = np.zeros((2 * m, 6), dtype=np.float32)
    P = []

    # Analyze each view separately
    for i in range(m):
        # Choose the central points
        world_point, image_point = world_points[i], image_points[i]
        center_point = np.array(image_size) / 2
        index = np.argsort(np.linalg.norm(image_point - center_point, axis=1))[:300]
        world_point, image_point = world_point[index], image_point[index]

        # Initialize the matrix A
        n = world_point.shape[0]
        A = np.zeros((2 * n, 9), dtype=np.float32)

        # Construct the matrix A
        for j in range(n):
            x, y = world_point[j]
            u, v = image_point[j]
            A[2 * j] = np.array([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            A[2 * j + 1] = np.array([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
        
        # Solve the matrix H
        eigenvalues, eigenvectors = np.linalg.eig(A.T @ A)
        h = eigenvectors[:, np.argmin(eigenvalues)]
        H = h.reshape(3, 3)
        P.append(H)

        # Update the matrix G
        h11, h12, _, h21, h22, _, h31, h32, _ = H.reshape(-1)
        G[2 * i] = np.array([h11 * h11 - h12 * h12, 2 * (h11 * h21 - h12 * h22), 2 * (h11 * h31 - h12 * h32), h21 * h21 - h22 * h22, 2 * (h21 * h31 - h22 * h32), h31 * h31 - h32 * h32])
        G[2 * i + 1] = np.array([h11 * h12, h11 * h22 + h12 * h21, h11 * h32 + h12 * h31, h21 * h22, h21 * h32 + h22 * h31, h31 * h32])
    
    # Solve the matrix B
    eigenvalues, eigenvectors = np.linalg.eig(G.T @ G)
    b = eigenvectors[:, np.argmin(eigenvalues)]
    B = np.array([[b[0], b[1], b[2]], [b[1], b[3], b[4]], [b[2], b[4], b[5]]])

    # Solve the matrix K
    K = np.linalg.cholesky(B)
    K = np.linalg.inv(K.T)
    K /= K[2, 2]
    
    return K, P


def reprojection_error(world_points, image_points, projection_matrix):
    error_list = []

    for i in range(len(world_points)):
        # Expand the world points
        expand_vector = np.ones((world_points[i].shape[0], 1), dtype=np.float32)
        object_points = np.append(world_points[i], expand_vector, axis=1)

        # Calculate the projected points
        projected_points = projection_matrix[i] @ object_points.T
        projected_points /= projected_points[2]

        # Calculate the reprojection error
        norm_error = np.linalg.norm(image_points[i].T - projected_points[:2], axis=0)
        error_list.append(np.mean(norm_error))
    
    return error_list


def standard_reference(world_points, image_points, image_size):
    # Expand the world points
    expand_vector = np.zeros((world_points[0].shape[0], 1), dtype=np.float32)
    object_points = [np.append(view, expand_vector, axis=1) for view in world_points]

    # Apply camera calibration function
    _, camera_matrix, _, _, _ = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    print("standard reference:\n{}".format(camera_matrix))


def main():
    # Read the required images
    image_path = './data'
    images = read_images(image_path)
    image_size = images[0].shape[:2][::-1]

    # Find the image points
    pattern_size = (31, 23)
    world_points, image_points = find_image_points(images, pattern_size)

    # Calculate the standard reference
    standard_reference(world_points, image_points, image_size)

    # Calculate the calibration matrix
    camera_matrix, projection_matrix = calibrate_camera(world_points, image_points, image_size)
    print("calibration matrix:\n{}".format(camera_matrix))

    # Calculate the reprojection error
    error_list = reprojection_error(world_points, image_points, projection_matrix)
    print("reprojection error:\n{}".format(error_list))


if __name__ == "__main__":
    main()
