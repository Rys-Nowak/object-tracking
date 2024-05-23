from typing import Tuple

import cv2
import numpy as np


def spatio_color_granules(img: np.ndarray, threshold: int) -> Tuple[list, np.ndarray]:
	height, width, _ = img.shape
	granulated_image = np.zeros_like(img)

	visited = np.zeros((height, width), dtype=bool)

	granules_out = []

	for i in range(height):
		for j in range(width):
			if not visited[i, j]:
				granules_out.append(_grow_region(i, j, img, img, visited, threshold, granulated_image))

	return granules_out, granulated_image


def spatio_temporal_granules(current_frame: np.ndarray,
                             previous_frames: list[np.ndarray], threshold: int) -> Tuple[list, np.ndarray]:
	p = len(previous_frames)
	height, width, _ = current_frame.shape
	granulated_image = np.zeros_like(current_frame)

	diffs = np.array([cv2.absdiff(current_frame, frame) for frame in previous_frames])
	diffs_shifted = diffs[np.newaxis, :, :, :] - diffs[:, np.newaxis, :, :]
	distances = np.linalg.norm(diffs_shifted, axis=-1)

	neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

	granules_out = []

	visited = np.zeros((height, width), dtype=bool)
	for i_p in range(1, p):
		for i in range(height):
			for j in range(width):
				if not visited[i, j]:
					region = []
					queue = [(i, j)]
					region_color = current_frame[i, j]

					while queue:
						cx, cy = queue.pop(0)
						if visited[cx, cy]:
							continue

						visited[cx, cy] = True
						region.append((cx, cy))
						granulated_image[cx, cy] = region_color

						for dx, dy in neighbors:
							nx, ny = cx + dx, cy + dy
							if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny]:
								if distances[0, i_p, nx, ny] < threshold:
									queue.append((nx, ny))
					granules_out.append(region)

				# granules_out.append(_grow_region(i, j, current_frame, median_diff, visited, threshold, granulated_image))

	del granules_out[np.argmax([len(x) for x in granules_out])]

	return granules_out, granulated_image


def color_neighborhood_granules(granules: list[list[tuple[int, int]]],
                                image: np.ndarray, threshold: int) -> Tuple[list, np.ndarray]:
	granules_out = []
	granulated_image = np.zeros_like(image)

	for granule in granules:
		values = np.array([image[x, y] for x, y in granule])
		granule = np.array(granule)

		diffs = values[:, np.newaxis, :] - values[np.newaxis, :, :]
		distances = np.linalg.norm(diffs, axis=2)

		for row in distances:
			granules_out.append(granule[np.where(row < threshold)])

	# TODO: find a faster way
	for granule in granules_out:
		a, b = granule[0]
		color = image[a, b, :]
		for x, y in granule:
			granulated_image[x, y] = color

	return [(xy[0], xy[1]) for xy in granule for granule in granules_out], granulated_image


def _grow_region(x: int, y: int, image1: np.ndarray, image2: np.ndarray,
                 visited: np.ndarray, thr: int, granulated_image: np.ndarray) -> list[tuple[int, int]]:
	neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
	height, width, _ = image1.shape

	region = []
	queue = [(x, y)]
	region_color = image1[x, y]

	while queue:
		cx, cy = queue.pop(0)
		if visited[cx, cy]:
			continue

		visited[cx, cy] = True
		region.append((cx, cy))
		granulated_image[cx, cy] = region_color

		for dx, dy in neighbors:
			nx, ny = cx + dx, cy + dy
			if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny]:
				neighbor_color = image2[nx, ny]
				if np.linalg.norm(region_color - neighbor_color) < thr:
					queue.append((nx, ny))

	return region
