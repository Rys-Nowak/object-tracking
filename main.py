import matplotlib.pyplot as plt
import numpy as np
import os
import granulation
import cv2
import time
import object_model
import rule_base

if __name__ == '__main__':
    path = 'data/mensa_seq0_1.1'
    sequence_num = 0
    frames_rgb = []
    frames_depth = []
    n_first_frames = 50 # P
    scale = 0.5
    for i in range(n_first_frames):
        im_rgb = cv2.imread(os.path.join(path, f'rgb/seq0_{i:04d}_{sequence_num}.ppm'))
        im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2RGB)
        im_rgb = cv2.rotate(im_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        im_rgb = cv2.resize(im_rgb, (0, 0), fx=scale, fy=scale)
        frames_rgb.append(im_rgb)

        im_depth = cv2.imread(os.path.join(path, f'depth/seq0_{i:04d}_{sequence_num}.pgm'), cv2.IMREAD_GRAYSCALE)
        im_depth = cv2.rotate(im_depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
        im_depth = cv2.resize(im_depth, (0, 0), fx=scale, fy=scale)
        frames_depth.append(im_depth)

        # cv2.imshow("rgb", im_rgb)
        # cv2.imshow("depth", im_depth)
        # cv2.waitKey()

    cv2.destroyAllWindows()
    frames_rgb = frames_rgb[::-1]
    frames_depth = frames_depth[::-1]
    frames = np.concatenate([frames_rgb, np.array(frames_depth)[..., np.newaxis]], axis=3) # rgb-d values
    
    start = time.time()
    # spatio_color_gran, granulated_image1 = granulation.spatio_color_granules(frames_rgb[0], 50)
    # plt.imsave('spatio_color_granules.jpg', granulated_image1)
    # print("Spatio color granules: done")

    spatio_temp_gran, granulated_image2 = granulation.spatio_temporal_granules(frames_rgb[0], frames_rgb[1:], 50)
    plt.imsave('spatio_temporal_granules.jpg', granulated_image2)
    print("Spatio temporal granules: done")

    rgb_gran, granulated_image3 = granulation.color_neighborhood_granules(spatio_temp_gran, granulated_image2, frames_rgb[0], 70)
    plt.imsave('RGB_granules.jpg', granulated_image3)
    print("RGB granules: done")

    depth_gran, granulated_image4 = granulation.color_neighborhood_granules(spatio_temp_gran, granulated_image2, frames_depth[0], 70)
    plt.imsave('depth_granules.jpg', granulated_image4)
    print("Depth granules: done")
    print("Elapsed time granules:", time.time()-start)

    start = time.time()
    model_bottom, model_top = object_model.create_initial_object_model(frames[0], frames[1:])
    print("Elapsed time object model:", time.time()-start)
    plt.imsave("RGB_top_object_model_estimation.jpg", model_top[:, :, :3])
    plt.imsave("RGB_bottom_object_model_estimation.jpg", model_bottom[:, :, :3])
    plt.imsave("D_top_object_model_estimation.jpg", model_top[:, :, 3], cmap='gray')
    plt.imsave("D_bottom_object_model_estimation.jpg", model_bottom[:, :, 3], cmap='gray')

    # start = time.time()
    # rule_base = rule_base.generate_rule_base(model_bottom, spatio_temp_gran, rgb_gran, depth_gran, verbose=False)
    # print("Elapsed time rule_base:", time.time()-start)
    # plt.imsave("rule_base.jpg", rule_base, cmap='hot')
