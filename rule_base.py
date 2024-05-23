import matplotlib.pyplot as plt
import numpy as np


def classify_granules(sp_clr_granules, sp_tmp_granules, rgb_granules, d_granules=None):
    labels = []
    print("No granules:", len(sp_clr_granules))
    for granule in sp_clr_granules:
        print(".")
        granule = set(granule)
        granule_len = len(granule)
        for tmp_granule in sp_tmp_granules:
            sp_tmp_len = len(tmp_granule)
            tmp_intersect_len = len(granule.intersection(set(tmp_granule)))
            for rgb_granule in rgb_granules:
                rgb_intersect_len = len(granule.intersection(set(rgb_granule)))
                rgb_len = len(rgb_granule)
                if tmp_intersect_len == 0 and rgb_intersect_len == 0 or\
                        tmp_intersect_len == 0 and rgb_intersect_len == granule_len or\
                        \
                        \
                        tmp_intersect_len == granule_len and rgb_intersect_len == 0:
                    labels.append(1)  # background
                elif min(granule_len, sp_tmp_len) > tmp_intersect_len > 0 and rgb_intersect_len == granule_len or\
                        tmp_intersect_len == granule_len and rgb_intersect_len == granule_len or\
                        tmp_intersect_len == sp_tmp_len and rgb_intersect_len == granule_len or\
                        tmp_intersect_len == granule_len and rgb_intersect_len == 0 or\
                        min(granule_len, sp_tmp_len) and min(granule_len, rgb_len) > rgb_intersect_len > 0 or\
                        tmp_intersect_len == granule_len and rgb_intersect_len == granule_len:
                    labels.append(2)  # object

    return labels


def show_labels(granules, labels, shape):
    img = np.zeros(shape, np.uint8)
    for label, granule in zip(labels, granules):
        for x, y in granule:
            img[x, y] = label

    plt.figure()
    plt.axis('off')
    plt.imshow(img, 'gray')
    plt.imsave("objects.jpg", img)
