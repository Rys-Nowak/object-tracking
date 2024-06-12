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

    for i in range(100):
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
    for i in range(100 - p):
        gran1, granulated_image1 = granulation.spatio_color_granules(frames_rgb[i], 50)
        np.savetxt(f"saved_data/granule{i}_1.csv", gran1, delimiter=",")
        plt.imsave(f'saved_data/spatio_color_granules{i}.jpg', granulated_image1)

        # print(gran1)
        gran2, granulated_image2 = granulation.spatio_temporal_granules(frames_rgb[i],
                                                                        frames_rgb[i:i+p], 50)
        np.savetxt(f"saved_data/granule{i}_2.csv", gran2, delimiter=",")
        plt.imsave(f'saved_data/spatio_temporal_granules{i}.jpg', granulated_image2)
        # print(granulated_image2)
        # print(len(gran2))

        granulated_image3 = granulation.color_neighborhood_granules(gran2, granulated_image2,
                                                                    frames_depth[i], 70)
        plt.imsave(f'saved_data/color_neighborhood_granules{i}.jpg', granulated_image3)
        # cv2.imshow("granulated image1", granulated_image1)
        # cv2.imshow("granulated image2", granulated_image2)
        # cv2.imshow("granulated image3", granulated_image3)
        # cv2.waitKey(1)

    # plt.imsave('spatio_color_granules.jpg', granulated_image1)
    # plt.imsave('spatio_temporal_granules.jpg', granulated_image2)
    # plt.imsave('color_neighborhood_granules.jpg', granulated_image3)

    # current_frame = frames[0]
    # previous_frames = frames[0:]

    # model_bottom, model_top = object_model.create_initial_object_model(frames[0], frames[1:])
    # print("Elapsed time object model:", time.time()-time1)
    #
    # plt.imsave("top_object_model_estimation.jpg", model_top)
    # plt.imsave("bottom_object_model_estimation.jpg", model_bottom)
    
