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
				granules_out.append(grow_region(i, j, img, img, visited, threshold, granulated_image))

	return granules_out, granulated_image


def spatio_temporal_granules(current_frame: np.ndarray,
                             previous_frames: list[np.ndarray], threshold: int) -> Tuple[list, np.ndarray]:
	height, width, _ = current_frame.shape
	granulated_image = np.zeros_like(current_frame)

	diffs = [cv2.absdiff(current_frame, frame) for frame in previous_frames]
	median_diff = np.median(diffs, axis=0).astype(np.uint8)

	granules_out = []

	visited = np.zeros((height, width), dtype=bool)
	for i in range(height):
		for j in range(width):
			if not visited[i, j]:
				granules_out.append(grow_region(i, j, current_frame, median_diff, visited, threshold, granulated_image))

	return granules_out, granulated_image


def color_neighborhood_granules():
	pass


def grow_region(x: int, y: int, image1: np.ndarray, image2: np.ndarray,
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
