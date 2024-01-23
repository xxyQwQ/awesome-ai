import sys
import cv2
import numpy as np


class UnionFindSet:
	def __init__(self, size):
		self.parent = np.arange(size)
	
	def find(self, x):
		if(x != self.parent[x]):
			self.parent[x] = self.find(self.parent[x])
		return self.parent[x]
	
	def union(self, x, y):
		u, v = self.find(x), self.find(y)
		if u != v:
			self.parent[u] = v


def binarize(gray_image, thresh_val):
	binary_image = np.where(gray_image >= thresh_val, 255, 0)
	return binary_image


def label(binary_image):
	rows, cols = binary_image.shape
	table = UnionFindSet(rows * cols)
	label = 0
	labeled_image = np.zeros_like(binary_image)

	# first pass
	for i in range(rows):
		for j in range(cols):
			if binary_image[i, j] > 0:
				if i > 0 and j > 0:
					if labeled_image[i-1, j-1] > 0:
						labeled_image[i, j] = labeled_image[i-1, j-1]
					elif labeled_image[i-1, j] > 0 and labeled_image[i, j-1] > 0:
						labeled_image[i, j] = labeled_image[i-1, j]
						table.union(labeled_image[i, j-1], labeled_image[i-1, j])
					elif labeled_image[i-1, j] > 0:
						labeled_image[i, j] = labeled_image[i-1, j]
					elif labeled_image[i, j-1] > 0:
						labeled_image[i, j] = labeled_image[i, j-1]
					else:
						label += 1
						labeled_image[i, j] = label
				elif i > 0:
					if labeled_image[i-1, j] > 0:
						labeled_image[i, j] = labeled_image[i-1, j]
					else:
						label += 1
						labeled_image[i, j] = label
				elif j > 0:
					if labeled_image[i, j-1] > 0:
						labeled_image[i, j] = labeled_image[i, j-1]
					else:
						label += 1
						labeled_image[i, j] = label
				else:
					label += 1
					labeled_image[i, j] = label
	
	# second pass
	for i in range(rows):
		for j in range(cols):
			if labeled_image[i, j] > 0:
				labeled_image[i, j] = table.find(labeled_image[i, j])
	label_list = np.unique(labeled_image)
	for i in range(1, len(label_list)):
		labeled_image[labeled_image == label_list[i]] = i
	labeled_image = labeled_image * 255 / len(label_list)

	return labeled_image


def get_attribute(labeled_image):
	rows, _ = labeled_image.shape
	attribute_list = []
	label_list = np.unique(labeled_image)
	for i in range(1, len(label_list)):
		index = np.argwhere(labeled_image == label_list[i])
		xpos, ypos = index[:, 1], rows - 1 - index[:, 0]
		xbar, ybar = np.mean(xpos), np.mean(ypos)
		xhat, yhat = xpos - xbar, ypos - ybar
		a = np.sum(np.square(xhat))
		b = 2 * np.sum(xhat * yhat)
		c = np.sum(np.square(yhat))
		tmin = np.arctan(b / (a - c)) / 2
		tmax = tmin + np.pi / 2
		emin = a * np.square(np.sin(tmin)) - b * np.sin(tmin) * np.cos(tmin) + c * np.square(np.cos(tmin))
		emax = a * np.square(np.sin(tmax)) - b * np.sin(tmax) * np.cos(tmax) + c * np.square(np.cos(tmax))
		if emin > emax:
			tmin, tmax = tmax, tmin
			emin, emax = emax, emin
		attribute = {'position': {'x': xbar, 'y': ybar}, 'orientation': tmin, 'roundness': emin / emax}
		attribute_list.append(attribute)
	return attribute_list


def main(argv):
	img_name = argv[0]
	thresh_val = int(argv[1])
	img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	binary_image = binarize(gray_image, thresh_val)
	labeled_image = label(binary_image)
	attribute_list = get_attribute(labeled_image)

	cv2.imwrite('output/' + img_name + "_gray.png", gray_image)
	cv2.imwrite('output/' + img_name + "_binary.png", binary_image)
	cv2.imwrite('output/' + img_name + "_labeled.png", labeled_image)
	print(attribute_list)


if __name__ == '__main__':
	main(sys.argv[1:])
