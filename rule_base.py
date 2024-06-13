import numpy as np
from enum import IntEnum


class Decision(IntEnum):
    UNDEFINED = 0
    BACKGROUND = 1
    OBJECT = 2


class Attribute(IntEnum):
    UNDEFINED = 0
    NB = 1  # does not belong
    PB = 2  # partially belongs
    BE = 3  # belongs
    CC = 4  # contained in


def belongs(feature_granule, intersection):
    return np.sum(intersection) == np.sum(feature_granule)


def partially_belongs(feature_granule, sp_clr_granule, intersection):
    return min(np.sum(sp_clr_granule), np.sum(feature_granule)) > np.sum(intersection) > 0


def does_not_belong(intersection):
    return np.sum(intersection) == 0


def contained_in(sp_clr_granule, intersection):
    return np.sum(intersection) == np.sum(sp_clr_granule)


def get_attribute(feature_granule, sp_clr_granule, intersection):
    if belongs(feature_granule, intersection):
        return Attribute.BE
    elif partially_belongs(feature_granule, sp_clr_granule, intersection):
        return Attribute.PB
    elif does_not_belong(intersection):
        return Attribute.NB
    elif contained_in(sp_clr_granule, intersection):
        return Attribute.CC


def calc_attribute(sp_tmp_granules, sp_clr_granule, i, j):
    label = sp_tmp_granules[i, j]
    current_granule = sp_tmp_granules == label
    intersection = np.logical_and(current_granule, sp_clr_granule)
    return get_attribute(current_granule, sp_clr_granule, intersection)


# Iterating over all pixels, for each checking corresponding sp-tmp, RGB and D granules
# According to their relation (Arttribute) with the sp-clr granules, the decision is taken
def generate_rule_base(sp_clr_granules, sp_tmp_granules, rgb_granules, d_granules, verbose=True):
    if verbose:
        print("Generating rule base...")

    img_height, img_width = sp_clr_granules.shape[:2]
    result_sp_tmp = np.zeros(sp_clr_granules.shape[:2], Attribute)
    result_rgb = np.zeros(sp_clr_granules.shape[:2], Attribute)
    result_d = np.zeros(sp_clr_granules.shape[:2], Attribute)
    for i in range(img_height):
        if verbose:
            print("\t\tAnalysing row no:", i)

        for j in range(img_width):
            sp_clr_label = sp_clr_granules[i, j]
            sp_clr_granule = sp_clr_granules == sp_clr_label
            result_sp_tmp[i, j] = calc_attribute(sp_tmp_granules, sp_clr_granule, i, j)
            result_rgb[i, j] = calc_attribute(rgb_granules, sp_clr_granule, i, j)
            result_d[i, j] = calc_attribute(d_granules, sp_clr_granule, i, j)

    if verbose:
        print("\tMaking decision...")

    result_object = np.logical_or.reduce((
        np.logical_and.reduce((result_sp_tmp == Attribute.PB, result_rgb == Attribute.BE, result_d == Attribute.BE)),
        np.logical_and.reduce((result_sp_tmp == Attribute.BE, result_rgb == Attribute.BE, result_d == Attribute.PB)),
        np.logical_and.reduce((result_sp_tmp == Attribute.CC, result_rgb == Attribute.BE, result_d == Attribute.BE)),
        np.logical_and.reduce((result_sp_tmp == Attribute.BE, result_rgb == Attribute.NB, result_d == Attribute.NB)),
        np.logical_and.reduce((result_sp_tmp == Attribute.PB, result_rgb == Attribute.PB, result_d == Attribute.CC)),
        np.logical_and.reduce((result_sp_tmp == Attribute.BE, result_rgb == Attribute.BE, result_d == Attribute.BE))
    ))
    result_background = np.logical_or.reduce((
        np.logical_and.reduce((result_sp_tmp == Attribute.NB, result_rgb == Attribute.NB, result_d == Attribute.NB)),
        np.logical_and.reduce((result_sp_tmp == Attribute.NB, result_rgb == Attribute.BE, result_d == Attribute.BE)),
        np.logical_and.reduce((result_sp_tmp == Attribute.PB, result_rgb == Attribute.BE, result_d == Attribute.PB)),
        np.logical_and.reduce((result_sp_tmp == Attribute.NB, result_rgb == Attribute.NB, result_d == Attribute.BE)),
        np.logical_and.reduce((result_sp_tmp == Attribute.BE, result_rgb == Attribute.NB, result_d == Attribute.NB)),
    ))
    result = np.zeros(sp_clr_granules.shape[:2], Decision)
    result[result_object] = Decision.OBJECT
    result[result_background] = Decision.BACKGROUND
    if verbose:
        print("\tRule base generated")

    return result.astype(np.uint8)
