# To przerobić. Jak nie działa, to znaczy że inna koncepcja, wtedy:
#   1. Lecimy po pikselach, ale patrzymy ich granule
#   2. Jak dalej nie działa, to liczymy granule na dolnym oszacowaniu obiektu, a nie na obrazie
#      - tu trochę problem ze spatio temporal i spatio color (O_ rgb lub rgb-d, muszę sam sprawdzić)
#        i jeszcze sp-tmp to więcej klatek biorą przecież i powinny być 3D - muszę kurde sam poczytać

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


def belongs(granule, intersection):
    return np.sum(intersection) == np.sum(granule)


def partially_belongs(granule, object_model, intersection):
    return min(np.sum(object_model), np.sum(granule)) > np.sum(intersection) > 0


def does_not_belong(intersection):
    return np.sum(intersection) == 0


def contained_in(object_model, intersection):
    return np.sum(intersection) == np.sum(object_model)


def get_attribute(granule, object_model, intersection):
    if belongs(granule, intersection):
        return Attribute.BE
    elif partially_belongs(granule, object_model, intersection):
        return Attribute.PB
    elif does_not_belong(intersection):
        return Attribute.NB
    elif contained_in(object_model, intersection):
        return Attribute.CC


def calc_attribute(sp_tmp_granules, object_model, i, j):
    label = sp_tmp_granules[i, j]
    current_granule = sp_tmp_granules == label
    intersection = np.logical_and(current_granule, object_model)
    return get_attribute(current_granule, object_model, intersection)


# Iterating over all pixels, for each checking corresponding sp-tmp, RGB and D granules
# According to their relation (Arttribute) with the object model (RGB-D bottom object estimation), the decision is taken
def generate_rule_base(object_model, sp_clr_granules, sp_tmp_granules, rgb_granules, d_granules, verbose=True):
    if verbose:
        print("Generating rule base...")

    # Idk if i should treat d and rgb values separate, nor should i use any or all
    # object_model_rgb = np.all(object_model[..., :3], axis=2)
    # object_model_d = object_model[..., 3] > 0

    if verbose:
        print("\tCalculating attributes...")

    img_height, img_width = object_model.shape[:2]
    result_sp_tmp = np.zeros(object_model.shape[:2], Attribute)
    result_rgb = np.zeros(object_model.shape[:2], Attribute)
    result_d = np.zeros(object_model.shape[:2], Attribute)
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
    result = np.zeros(object_model.shape[:-1], Decision)
    result[result_object] = Decision.OBJECT
    result[result_background] = Decision.BACKGROUND
    if verbose:
        print("\tRule base generated")

    return result.astype(np.uint8)
