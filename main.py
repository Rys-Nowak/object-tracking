import matplotlib.pyplot as plt
import numpy as np
import os
import granulation
import cv2
import time

if __name__ == '__main__':

    path = 'data/GOT-10k_Test_000063'
    frames = []

    for i in range(1, 10):
        im = cv2.imread(os.path.join(path, f'{i:08d}.jpg'))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_resized = cv2.resize(im, (0, 0), None, 0.5, 0.5)
        frames.append(im_resized)

    im1 = cv2.imread('data/lena.png')
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

    time1 = time.time()

    gran1, granulated_image1 = granulation.spatio_color_granules(frames[0], 50)
    print("Spatio color granules: done")
    gran2, granulated_image2 = granulation.spatio_temporal_granules(frames[0], frames[1:], 100)
    print("Spatio temporal granules: done")
    gran3, granulated_image3 = granulation.color_neighborhood_granules(gran2, frames[0], 50)
    print("Color neighborhood granules: done")

    time2 = time.time()

    print("Elapsed time:", time2-time1)

    plt.imsave('spatio_color_granules.jpg', granulated_image1)
    plt.imsave('spatio_temporal_granules.jpg', granulated_image2)
    plt.imsave('color_neighborhood_granules.jpg', granulated_image3)



