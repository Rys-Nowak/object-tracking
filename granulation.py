from typing import Tuple
import cv2
import numpy as np


def region_growing(diff_frame, threshold):
	height, width = diff_frame.shape
	segmented_image = np.zeros_like(diff_frame, dtype=np.int32)
	region_label = 1

	def get_neighbors(y_, x_):
		neighbors = []
		for ny in range(max(0, y_ - 1), min(height, y_ + 2)):
			for nx in range(max(0, x_ - 1), min(width, x_ + 2)):
				if ny != y_ or nx != x_:
					neighbors.append((ny, nx))
		return neighbors

	for y in range(height):
		for x in range(width):
			if segmented_image[y, x] == 0 and diff_frame[y, x] > threshold:
				stack = [(y, x)]
				while stack:
					cy, cx = stack.pop()
					if segmented_image[cy, cx] == 0 and diff_frame[cy, cx] > threshold:
						segmented_image[cy, cx] = region_label
						stack.extend(get_neighbors(cy, cx))
				region_label += 1

	return segmented_image


def region_growing_color(image, x, y, Thr):
	height, width, _ = image.shape
	segmented_image = np.zeros((height, width), dtype=np.int32)
	region_label = 1
	seed_color = image[y, x]

	def get_neighbors(y, x):
		neighbors = []
		for ny in range(max(0, y - 1), min(height, y + 2)):
			for nx in range(max(0, x - 1), min(width, x + 2)):
				if ny != y or nx != x:
					neighbors.append((ny, nx))
		return neighbors

	stack = [(y, x)]
	while stack:
		cy, cx = stack.pop()
		if segmented_image[cy, cx] == 0:
			current_color = image[cy, cx]
			color_diff = np.linalg.norm(current_color - seed_color)
			if color_diff < Thr:
				segmented_image[cy, cx] = region_label
				stack.extend(get_neighbors(cy, cx))

	return segmented_image


def spatio_color_granules(image: np.ndarray, Thr: int) -> Tuple[list, np.ndarray]:
	height, width, _ = image.shape
	granulated_image = np.zeros((height, width), dtype=np.int32)
	region_label = 1
	granules_list = []

	print("Calculating spatio-color granules, image height:", height)
	for y in range(height):
		print("\tAnalysing row no:", y)
		for x in range(width):
			if granulated_image[y, x] == 0:
				new_region = region_growing_color(image, x, y, Thr)
				if np.count_nonzero(new_region) > 0:
					granulated_image[new_region == 1] = region_label
					region_mask = (new_region == 1)
					mean_value = np.mean(image[region_mask])
					variance_value = np.var(image[region_mask])
					granules_list.append((region_label, mean_value, variance_value))
					region_label += 1

	granulated_image_normalized = cv2.normalize(granulated_image, None, 0, 255, cv2.NORM_MINMAX)
	granulated_image_uint8 = granulated_image_normalized.astype(np.uint8)

	return granules_list, granulated_image_uint8


def spatio_temporal_granules(current_frame: np.ndarray,
                             previous_frames: list[np.ndarray], threshold: int) -> Tuple[list, np.ndarray]:
	p = len(previous_frames)
	height, width, _ = current_frame.shape
	granulated_image = np.zeros((height, width), dtype=np.int32)

	diffs_first2all = np.array([cv2.absdiff(current_frame, frame) for frame in previous_frames])

	median_diff = np.median(diffs_first2all, axis=0)
	threshold = 0.2 * np.max(median_diff)

	diffs_frame2frame = np.array(
		[cv2.absdiff(frame1, frame2) for frame1, frame2 in zip(previous_frames, previous_frames[1:])])

	first2all_diff_matrix = np.zeros((height, width, p), dtype=np.int32)
	frame2frame_diff_matrix = np.zeros((height, width, p - 1), dtype=np.int32)

	for i in range(p):
		first2all_diff_matrix[:, :, i] = cv2.cvtColor(diffs_first2all[i], cv2.COLOR_BGR2GRAY)
		if i < p - 1:
			frame2frame_diff_matrix[:, :, i] = cv2.cvtColor(diffs_frame2frame[i], cv2.COLOR_BGR2GRAY)

	spatial_granules = [region_growing(first2all_diff_matrix[:, :, i], threshold) for i in range(p)]

	temporal_granules = np.zeros_like(first2all_diff_matrix, dtype=np.int32)
	for i in range(p):
		for y in range(height):
			for x in range(width):
				if spatial_granules[i][y, x] != 0:
					temporal_granules[y, x, i] = spatial_granules[i][y, x]

	granules_list = []
	for i in range(p):
		unique_granules = np.unique(temporal_granules[:, :, i])
		for granule in unique_granules:
			if granule != 0:
				granule_mask = (temporal_granules[:, :, i] == granule)
				mean_value = np.mean(first2all_diff_matrix[granule_mask])
				variance_value = np.var(first2all_diff_matrix[granule_mask])
				granules_list.append((granule, mean_value, variance_value))
				granulated_image[granule_mask] = granule

	granulated_image_normalized = cv2.normalize(granulated_image, None, 0, 255, cv2.NORM_MINMAX)
	granulated_image_uint8 = granulated_image_normalized.astype(np.uint8)

	return granules_list, granulated_image_uint8


def color_granules(granules: list, image: np.ndarray, image_depth: np.ndarray, threshold: int) -> np.ndarray:
	region_label = 1
	granulated_image = np.zeros(image.shape[:2])
	image = image[:, :, np.newaxis]
	image_rgb_d = np.dstack([image, image_depth])
	print("Calculating color granules, total length:", len(granules))
	for i in range(len(granules)):
		print("\tAnalysing granule no:", i)
		ys, xs = np.where(np.all(image == i, axis=2))
		if ys.any() and xs.any():
			first_point = image_rgb_d[ys[0], xs[0]]
			if first_point[0] == 0:
				continue
		for y_ in ys:
			for x_ in xs:
				if np.linalg.norm(image_rgb_d[y_, x_] - first_point) < threshold:
					granulated_image[y_, x_] = region_label
		region_label += 1

	granulated_image_normalized = cv2.normalize(granulated_image, None, 0, 255, cv2.NORM_MINMAX)
	granulated_image_uint8 = granulated_image_normalized.astype(np.uint8)

	return granulated_image_uint8
