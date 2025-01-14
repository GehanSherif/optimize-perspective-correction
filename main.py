import os
import cv2
from pylsd.lsd import lsd
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

# Functions for the rectification process
def line_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lines = lsd(gray)
    p_lines = []
    image_one = np.ones(np.shape(image)) * 255

    for i in range(lines.shape[0]):
        pt1 = (int(lines[i, 0]), int(lines[i, 1]))
        pt2 = (int(lines[i, 2]), int(lines[i, 3]))
        width = np.sqrt((lines[i, 0] - lines[i, 2]) ** 2 + (lines[i, 1] - lines[i, 3]) ** 2)
        angle = abs(lines[i, 0] - lines[i, 2]) / abs(lines[i, 1] - lines[i, 3])

        if width > 10 and (angle < 1):
            p_lines.append([lines[i, 0], lines[i, 1], lines[i, 2], lines[i, 3]])
            cv2.line(image_one, pt1, pt2, (0, 0, 255), int(2))

    cv2.imwrite('test_lines.jpg', image_one)
    return p_lines


def crop_black_border(image, points):
    x1 = int(max(points[0][1] / points[0][2], points[1][1] / points[1][2]))
    x2 = int(min(points[2][1] / points[2][2], points[3][1] / points[3][2]))
    y1 = int(max(points[0][0] / points[0][2], points[2][0] / points[2][2]))
    y2 = int(min(points[1][0] / points[1][2], points[3][0] / points[3][2]))
    image_out = image[x1:x2, y1:y2, :]
    cv2.imwrite('output.jpg', image_out)
    return image_out


def cost_function(h, w, f, lines, H1):
    E = 0
    for line in lines:
        u = [line[0], line[1], 1]
        u1 = np.dot(H1, u)
        v = [line[2], line[3], 1]
        v1 = np.dot(H1, v)

        w = (line[0] + line[2]) ** 2 + (line[1] + line[3]) ** 2
        d = min(abs(u1[0] / u1[2] - v1[0] / v1[2]), abs(u1[1] / u1[2] - v1[1] / v1[2])) ** 2
        E += w * d

    a = max(h, w)
    F = (max(a, f) / min(a, f) - 1) ** 2
    cost = E + 0.1 * F

    return cost


def gradient(image, lines, f_, theta_, phi_, gamma_):
    h_, w_, _ = np.shape(image)

    K = np.array([[f_, 0, w_ / 2],
                  [0, f_, h_ / 2],
                  [0, 0, 1]])

    K1 = np.array([[1 / f_, 0, - w_ / (2 * f_)],
                   [0, 1 / f_, - h_ / (2 * f_)],
                   [0, 0, 1]])

    R = np.array([[0, -gamma_, phi_],
                  [gamma_, 0, -theta_],
                  [-phi_, theta_, 0]])
    R = expm(R)
    H1 = np.dot(K, np.dot(R, K1))

    return H1


def gradient_descent_optimizer(image, max_iters, initial_lr, decay_rate):
    init_theta, init_phi, init_gamma = 0, 0, 0
    h, w, _ = np.shape(image)
    lines = line_detection(image)
    d = np.sqrt(h ** 2 + w ** 2)
    init_f = d / 2

    lr = initial_lr
    prev_cost = float('inf')

    for i in range(max_iters):
        H1 = gradient(image, lines, init_f, init_theta, init_phi, init_gamma)
        cost = cost_function(h, w, init_f, lines, H1)

        print(f"Iteration {i + 1}: Cost = {cost}, Learning Rate = {lr}")
        lr = initial_lr / (1 + decay_rate * i)

        if abs(prev_cost - cost) < 1e-5:
            break
        prev_cost = cost
        # Transform and crop the image
    top_left = np.dot(H1, [0, 0, 1])
    top_right = np.dot(H1, [w, 0, 1])
    bottom_left = np.dot(H1, [0, h, 1])
    bottom_right = np.dot(H1, [w, h, 1])

    points = [top_left, top_right, bottom_left, bottom_right]
    image_crop = crop_black_border(cv2.warpPerspective(image.copy(), np.float32(H1), (2 * w, 2 * h)), points)

    return image_crop, H1

def similarity_score(edited_image, output_image= "output.jpg"):
      import cv2

      image1 = cv2.imread(edited_image)
      image2 = cv2.imread(output_image)

      sift = cv2.SIFT_create(nfeatures=15000)

      keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
      keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

      index_params = dict(algorithm=1, trees=5)  
      search_params = dict(checks=500)           # Higher = more accurate but slower

      flann = cv2.FlannBasedMatcher(index_params, search_params)

      matches = flann.knnMatch(descriptors1, descriptors2, k=2)

      good_matches = []
      for m, n in matches:
          if m.distance < 0.7 * n.distance:  
              good_matches.append(m)

      similarity = len(good_matches) / max(len(keypoints1), len(keypoints2))

      return -1* similarity
      
def total_error(params, raw_images, edited_images):
    initial_lr, decay_rate, max_iters = params
    total_mse = 0

    for raw_path, edited_path in zip(raw_images, edited_images):
        raw_image = cv2.imread(raw_path)
        edited_image = cv2.imread(edited_path)

        #raw_image = cv2.resize(raw_image, (600, 400))
        #edited_image = cv2.resize(edited_image, (600, 400))

        gradient_descent_optimizer(raw_image, int(max_iters), initial_lr, decay_rate)
        cost = similarity_score(edited_path)
        total_mse += cost

    return total_mse


# Optimize hyperparameters
if __name__ == "__main__":
    raw_images = ["/content/interior_data/train_A/"+i for i in sorted(os.listdir("/content/interior_data/train_A"))]  # Replace with your file paths
    edited_images = ["/content/interior_data/train_B/"+i for i in sorted(os.listdir("/content/interior_data/train_B"))]

    initial_params = [0.01, 0.001, 5]  # Initial guess: [learning rate, decay rate, iterations]
    bounds = [(0.01, 1.0), (0.001,0.1), (5, 100)]  # Parameter bounds

    result = minimize(
        total_error,
        initial_params,
        args=(raw_images, edited_images),
        bounds=bounds,
        method='SLSQP',
        options={'disp': True}
    )

    optimal_params = result.x
    print("Optimal Learning Rate:", optimal_params[0])
    print("Optimal Decay Rate:", optimal_params[1])
    print("Optimal Number of Iterations:", int(optimal_params[2]))
