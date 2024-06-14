import matplotlib.pyplot as plt
import numpy as np
import os
import granulation
import cv2
import time
import object_model
import rule_base
import pickle


if __name__ == '__main__':
    path = 'data/mensa_seq0_1.1'
    sequence_num = 0
    frames_rgb = []
    frames_depth = []
    P = 5
    n_first_frames = P
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
    model_bottom, model_top = object_model.create_initial_object_model(frames[0], frames[1:])
    print("Elapsed time object model:", time.time()-start)
    plt.imsave("saved_data/RGB_top_object_model_estimation.jpg", model_top[:, :, :3])
    plt.imsave("saved_data/RGB_bottom_object_model_estimation.jpg", model_bottom[:, :, :3])
    plt.imsave("saved_data/D_top_object_model_estimation.jpg", model_top[:, :, 3], cmap='gray')
    plt.imsave("saved_data/D_bottom_object_model_estimation.jpg", model_bottom[:, :, 3], cmap='gray')
    
    i = 0 # init
    start = time.time()
    sp_clr_info = None
    sp_clr_granules = None
    try:
        with open(f"saved_data/sp_clr_info{i}.pkl", "rb") as f_info_r_1:
            sp_clr_info = pickle.load(f_info_r_1)
        with open(f'saved_data/sp_clr_granules{i}.pkl', "rb") as f_im_r_1:
            sp_clr_granules = pickle.load(f_im_r_1)
    except:
        sp_clr_info, sp_clr_granules = granulation.spatio_color_granules(frames_rgb[i], 20)
        plt.imsave(f'saved_data/spatio_color_granules{i}.jpg', sp_clr_granules, cmap='hot')
        with open(f"saved_data/sp_clr_info{i}.pkl", "wb") as f_info_w_1:
            pickle.dump(sp_clr_info, f_info_w_1)
        with open(f'saved_data/sp_clr_granules{i}.pkl', "wb") as f_im_w_1:
            pickle.dump(sp_clr_granules, f_im_w_1)

    print("Spatio color granules: done")

    sp_tmp_info = None
    sp_tmp_granules = None
    try:
        with open(f"saved_data/sp_tmp_info{i}.pkl", "rb") as f_info_r_2:
            sp_tmp_info = pickle.load(f_info_r_2)
        with open(f'saved_data/sp_tmp_granules{i}.pkl', "rb") as f_im_r_2:
            sp_tmp_granules = pickle.load(f_im_r_2)
    except:
        sp_tmp_info, sp_tmp_granules = granulation.spatio_temporal_granules(frames_rgb[i], frames_rgb[i+1: i+P], 50)
        plt.imsave(f'saved_data/spatio_temporal_granules{i}.jpg', sp_tmp_granules, cmap="hot")
        with open(f"saved_data/sp_tmp_info{i}.pkl", "wb") as f_info_w_2:
            pickle.dump(sp_tmp_info, f_info_w_2)
        with open(f'saved_data/sp_tmp_granules{i}.pkl', "wb") as f_im_w_2:
            pickle.dump(sp_tmp_granules, f_im_w_2)

    print("Spatio temporal granules: done")

    depth_granules = None
    try:
        with open(f"saved_data/depth_granules{i}.pkl", "rb") as f_im_r_3:
            depth_granules = pickle.load(f_im_r_3)
    except:
        depth_granules = granulation.color_granules(sp_tmp_info, sp_tmp_granules, frames_depth[i], 70)
        plt.imsave(f'saved_data/depth_granules{i}.jpg', depth_granules, cmap="hot")
        with open(f"saved_data/depth_granules{i}.pkl", "wb") as f_im_w_3:
            pickle.dump(depth_granules, f_im_w_3)

    print("Depth granules: done")
    print("Elapsed time granules:", time.time()-start)

    start = time.time()
    rule_base = rule_base.generate_rule_base(sp_clr_granules, sp_tmp_granules, sp_tmp_granules, depth_granules, verbose=True)
    print("Elapsed time rule_base:", time.time()-start)
    plt.imsave(f"saved_data/rule_base{i}.jpg", rule_base, cmap='hot')
