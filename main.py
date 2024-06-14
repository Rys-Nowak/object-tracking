import matplotlib.pyplot as plt
import numpy as np
import os
import granulation
import cv2
import copy
import time
import object_model
import rule_base
import pickle


if __name__ == '__main__':

    img0 = cv2.imread("out/frame5.jpg")
    width, height, layers = img0.shape
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('video.avi', fourcc, 15, (width, height), isColor=True)

    for i in range(5, 500):
        img = cv2.imread(f"out/frame{i}.jpg")
        # cv2.imshow("out", img)
        # cv2.waitKey(10)
        out.write(img)

    # cv2.destroyAllWindows()
    out.release()


    # path = 'data/mensa_seq0_1.1'
    # sequence_num = 0
    # frames_rgb = []
    # frames_depth = []
    # P = 5
    # n_first_frames = 500
    # scale = 0.5
    # for i in range(n_first_frames):
    #     im_rgb = cv2.imread(os.path.join(path, f'rgb/seq0_{i:04d}_{sequence_num}.ppm'))
    #     im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2RGB)
    #     im_rgb = cv2.rotate(im_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #     im_rgb = cv2.resize(im_rgb, (0, 0), fx=scale, fy=scale)
    #     frames_rgb.append(im_rgb)
    #
    #     im_depth = cv2.imread(os.path.join(path, f'depth/seq0_{i:04d}_{sequence_num}.pgm'), cv2.IMREAD_GRAYSCALE)
    #     im_depth = cv2.rotate(im_depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #     im_depth = cv2.resize(im_depth, (0, 0), fx=scale, fy=scale)
    #     frames_depth.append(im_depth)
    #
    #     # cv2.imshow("rgb", im_rgb)
    #     # cv2.imshow("depth", im_depth)
    #     # cv2.waitKey()
    #
    # frames_rgb = frames_rgb[::-1]
    # frames_depth = frames_depth[::-1]
    # frames = np.concatenate([frames_rgb, np.array(frames_depth)[..., np.newaxis]], axis=3)  # rgb-d values
    #
    #
    # for i in range(n_first_frames- P+1):
    #     # _, spatio_temporal_gran_image = granulation.spatio_temporal_granules(frames_rgb[i], frames_rgb[i-P:i], 50)
    #     model_bottom, model_top = object_model.create_initial_object_model(frames[i], frames[i+1:i+P+1])
    #     common_part = cv2.bitwise_and(model_bottom, model_top)
    #     common2first_ch = np.logical_and(common_part[:, :, 0], common_part[:, :, 1])
    #     common2last_ch = np.logical_and(common_part[:, :, 2], common_part[:, :, 3])
    #
    #     common2first_ch = common2first_ch.astype(np.uint8)*255
    #     common2last_ch = common2last_ch.astype(np.uint8)*255
    #     common_part = cv2.bitwise_and(common2first_ch, common2last_ch)
    #
    #     # depth_granules = granulation.color_granules(sp_tmp_info, spatio_temporal_gran_image, frames_depth[i], 70)
    #     # B = cv2.medianBlur(common_part, 3)
    #     # B_eroded = cv2.erode(B, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=4)
    #     # B_opened = cv2.dilate(B_eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=4)
    #     # _, thresh1 = cv2.threshold(B_opened, 1, 255, cv2.THRESH_BINARY)
    #     #
    #     # # cv2.imshow("bounding boxes", thresh1)
    #     # # cv2.imshow("B", B_opened)
    #     # # cv2.waitKey(1)
    #     #
    #     retval, labels, stats, centroids = cv2.connectedComponentsWithStats(common_part)
    #     labels_scaled = np.uint8(labels / retval * 255)
    #
    #     I_VIS = copy.deepcopy(frames_rgb[i])
    #     if stats.shape[0] > 1:  # are there any objects
    #         tab = stats[1:, 4]  # 4 columns without first element
    #         # print("Tab:", stats)
    #         for pi in range(1, len(stats)):
    #             if stats[pi, 4] > 200:
    #                 # object = object[:4]
    #                 # print("Object:", stats[pi])
    #                 # pi = np.argmax(tab) # finding the index of the largest item
    #                 # pi = pi + 1 # increment because we want the index in stats, not in tab
    #                 # drawing a bbox
    #                 cv2.rectangle(I_VIS, (stats[pi, 0], stats[pi, 1]),
    #                               (stats[pi, 0] + stats[pi, 2], stats[pi, 1] + stats[pi, 3]), (255, 0, 0), 2)
    #                 # print information about the field and the number of the largest element
    #                 # cv2.putText(I_VIS, "%f" % stats[pi, 4], (stats[pi, 0], stats[pi, 1]),
    #                 #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    #                 # cv2.putText(I_VIS, "%d" % pi, (int(centroids[pi, 0]), int(centroids[pi, 1])),
    #                 #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    #     # plt.imsave(f"out/frame{i}.jpg", I_VIS)
    #     cv2.imshow("granules", I_VIS)
    #     # cv2.imshow("bin", thresh1)
    #     # cv2.imshow("image", I_VIS)
    #     cv2.waitKey(100)



    # cv2.destroyAllWindows()
    # frames = np.concatenate([frames_rgb, np.array(frames_depth)[..., np.newaxis]], axis=3) # rgb-d values
    #
    # start = time.time()
    # model_bottom, model_top = object_model.create_initial_object_model(frames[0], frames[1:])
    # print("Elapsed time object model:", time.time() - start)
    # plt.imsave("saved_data/RGB_top_object_model_estimation.jpg", model_top[:, :, :3])
    # plt.imsave("saved_data/RGB_bottom_object_model_estimation.jpg", model_bottom[:, :, :3])
    # plt.imsave("saved_data/D_top_object_model_estimation.jpg", model_top[:, :, 3], cmap='gray')
    # plt.imsave("saved_data/D_bottom_object_model_estimation.jpg", model_bottom[:, :, 3], cmap='gray')
    #
    # i = 0 # init
    # start = time.time()
    # sp_clr_info = None
    # sp_clr_granules = None
    # try:
    #     with open(f"saved_data/sp_clr_info{i}.pkl", "rb") as f_info_r_1:
    #         sp_clr_info = pickle.load(f_info_r_1)
    #     with open(f'saved_data/sp_clr_granules{i}.pkl', "rb") as f_im_r_1:
    #         sp_clr_granules = pickle.load(f_im_r_1)
    # except:
    #     sp_clr_info, sp_clr_granules = granulation.spatio_color_granules(frames_rgb[i], 50)
    #     plt.imsave(f'saved_data/spatio_color_granules{i}.jpg', sp_clr_granules, cmap='hot')
    #     with open(f"saved_data/sp_clr_info{i}.pkl", "wb") as f_info_w_1:
    #         pickle.dump(sp_clr_info, f_info_w_1)
    #     with open(f'saved_data/sp_clr_granules{i}.pkl', "wb") as f_im_w_1:
    #         pickle.dump(sp_clr_granules, f_im_w_1)
    #
    # print("Spatio color granules: done")
    #
    # sp_tmp_info = None
    # sp_tmp_granules = None
    # try:
    #     with open(f"saved_data/sp_tmp_info{i}.pkl", "rb") as f_info_r_2:
    #         sp_tmp_info = pickle.load(f_info_r_2)
    #     with open(f'saved_data/sp_tmp_granules{i}.pkl', "rb") as f_im_r_2:
    #         sp_tmp_granules = pickle.load(f_im_r_2)
    # except:
    #     sp_tmp_info, sp_tmp_granules = granulation.spatio_temporal_granules(frames_rgb[i], frames_rgb[i+1: i+P], 50)
    #     plt.imsave(f'saved_data/spatio_temporal_granules{i}.jpg', sp_tmp_granules, cmap="hot")
    #     with open(f"saved_data/sp_tmp_info{i}.pkl", "wb") as f_info_w_2:
    #         pickle.dump(sp_tmp_info, f_info_w_2)
    #     with open(f'saved_data/sp_tmp_granules{i}.pkl', "wb") as f_im_w_2:
    #         pickle.dump(sp_tmp_granules, f_im_w_2)
    #
    # print("Spatio temporal granules: done")
    #
    # depth_granules = None
    # try:
    #     with open(f"saved_data/depth_granules{i}.pkl", "rb") as f_im_r_3:
    #         depth_granules = pickle.load(f_im_r_3)
    # except:
    #     depth_granules = granulation.color_granules(sp_tmp_info, sp_tmp_granules, frames_depth[i], 70)
    #     plt.imsave(f'saved_data/depth_granules{i}.jpg', depth_granules, cmap="hot")
    #     with open(f"saved_data/depth_granules{i}.pkl", "wb") as f_im_w_3:
    #         pickle.dump(depth_granules, f_im_w_3)
    #
    # print("Depth granules: done")
    # print("Elapsed time granules:", time.time()-start)
    #
    # start = time.time()
    # rule_base = rule_base.generate_rule_base(sp_clr_granules, sp_tmp_granules, sp_tmp_granules, depth_granules, verbose=True)
    # print("Elapsed time rule_base:", time.time()-start)
    # plt.imsave(f"saved_data/rule_base{i}.jpg", rule_base, cmap='hot')
