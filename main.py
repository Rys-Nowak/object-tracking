import matplotlib.pyplot as plt
import numpy as np
import os
import granulation
import cv2
import time
import object_model

if __name__ == '__main__':

    path = 'data/pedestrian/input'
    frames = []

    for i in range(970, 975):
        im = cv2.imread(os.path.join(path, f'in{i:06d}.jpg'))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_resized = cv2.resize(im, (0, 0), None, 1, 1)
        frames.append(im_resized)

    frames = frames[::-1]

    # im1 = cv2.imread('data/lena.png')
    # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    
    time1 = time.time()
    
    gran1, granulated_image1 = granulation.spatio_color_granules(frames[0], 50)
    print("Spatio color granules: done")
    gran2, granulated_image2 = granulation.spatio_temporal_granules(frames[0], frames[0:], 30)
    print("Spatio temporal granules: done")
    gran3, granulated_image3 = granulation.color_neighborhood_granules(gran2, frames[0], 50)
    print("Color neighborhood granules: done")
    
    time2 = time.time()
    # print(gran2)
    
    print("Elapsed time:", time2-time1)
    
    plt.imsave('spatio_color_granules.jpg', granulated_image1)
    plt.imsave('spatio_temporal_granules.jpg', granulated_image2)
    plt.imsave('color_neighborhood_granules.jpg', granulated_image3)

    # current_frame = frames[0]
    # previous_frames = frames[0:]
    #
    #
    # time1 = time.time()
    # distances = np.linalg.norm(diffs_shifted, axis=-1)
    # time2 = time.time()
    # print(time2-time1)
    # print(diffs[np.newaxis, :, :, :].shape)
    # print(diffs[:, np.newaxis, :, :].shape)
    # print(diffs_shifted.shape)
    # print(diffs_shifted[2, 3])
    # plt.imshow(diffs[4])
    # plt.show()

    time1 = time.time()
    model_bottom, model_top = object_model.create_initial_object_model(frames[0], frames[1:])
    print("Elapsed time object model:", time.time()-time1)

    plt.imsave("top_object_model_estimation.jpg", model_top)
    plt.imsave("bottom_object_model_estimation.jpg", model_bottom)
    
