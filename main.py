import matplotlib.pyplot as plt
import numpy as np
import os
import granulation
import cv2
import time
import object_model

if __name__ == '__main__':

    path = 'data/mensa_seq0_1.1'
    sequence_num = 0
    frames_rgb = []
    frames_depth = []

    for i in range(500):
        im_rgb = cv2.imread(os.path.join(path, f'rgb/seq0_{i:04d}_{sequence_num}.ppm'))
        im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2RGB)
        im_rgb = cv2.rotate(im_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        im_rgb = cv2.resize(im_rgb, (0, 0), None, 0.5, 0.5)
        frames_rgb.append(im_rgb)

        im_depth = cv2.imread(os.path.join(path, f'depth/seq0_{i:04d}_{sequence_num}.pgm'), cv2.IMREAD_GRAYSCALE)
        im_depth = cv2.rotate(im_depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
        im_depth = cv2.resize(im_depth, (0, 0), None, 0.5, 0.5)
        frames_depth.append(im_depth)

    # frames_rgb = frames_rgb[::-1]

    p = 5
    for i in range(500 - p):
        # gran1, granulated_image1 = granulation.spatio_color_granules(frames_rgb[i], 50)
        # print(gran1)
        gran2, granulated_image2 = granulation.spatio_temporal_granules(frames_rgb[i],
                                                                        frames_rgb[i:i+p], 50)
        # gran3, granulated_image3 = granulation.color_neighborhood_granules(gran2, frames_depth[i], 100)
        # cv2.imshow("granulated image1", granulated_image1)
        cv2.imshow("granulated image2", granulated_image2)
        # cv2.imshow("granulated image3", granulated_image3)
        cv2.waitKey(1)


    # time2 = time.time()
    # print("Elapsed time:", time2-time1)
    #
    # plt.imsave('spatio_color_granules.jpg', granulated_image1)
    # plt.imsave('spatio_temporal_granules.jpg', granulated_image2)
    # plt.imsave('color_neighborhood_granules.jpg', granulated_image3)

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

    # time1 = time.time()
    # model_bottom, model_top = object_model.create_initial_object_model(frames[0], frames[1:])
    # print("Elapsed time object model:", time.time()-time1)
    #
    # plt.imsave("top_object_model_estimation.jpg", model_top)
    # plt.imsave("bottom_object_model_estimation.jpg", model_bottom)
    
