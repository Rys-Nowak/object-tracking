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


# Idk if i should treat d and rgb values separate, nor should i use any or all
def prepare_object_model(object_model):
    result_RGB_D = []
    result_RGB = []
    result_D = []
    for i in range(object_model.shape[0]):
        for j in range(object_model.shape[1]):
            rgb = object_model[i, j, :3]
            d = object_model[i, j, 3]
            if all(rgb):
                result_RGB.append((i, j))
            if d:
                result_D.append((i, j))
            if all(object_model[i, j]):
                result_RGB_D.append((i, j))

    return result_RGB_D, result_RGB, result_D


def belongs(granule, intersection):
    return len(intersection) == len(granule)


def partially_belongs(granule, object_model, intersection):
    return min(len(object_model), len(granule)) > len(intersection) > 0


def does_not_belong(intersection):
    return len(intersection) == 0


def contained_in(object_model, intersection):
    return len(intersection) == len(object_model)


def calc_attributes(granules, object_model_pixels, object_model, verbose=True):
    result = np.zeros(object_model.shape[:-1], Attribute)
    percentages = []
    for i, granule in enumerate(granules):
        if verbose:
            percentage = int(i / len(granules) * 100)
            if percentage not in percentages:
                percentages.append(percentage)
                print(f"\t\t{percentage}%")

        granule_set = set(granule)
        object_model_set = set(object_model_pixels)
        intersection = granule_set.intersection(object_model_set)
        if belongs(granule, intersection):
            result[granule] = Attribute.BE
        elif partially_belongs(granule, object_model_set, intersection):
            result[granule] = Attribute.PB
        elif does_not_belong(intersection):
            result[granule] = Attribute.NB
        elif contained_in(object_model_set, intersection):
            result[granule] = Attribute.CC

    return result


# Iterating over all pixels, for each checking corresponding sp-tmp, RGB and D granules
# According to their relation (Arttribute) with the object model (RGB-D bottom object estimation), the decision is taken
def generate_rule_base(object_model, sp_tmp_granules, rgb_granules, d_granules, verbose=True):
    if verbose:
        print("Generating rule base...")
        print("\tPreparing object model...")

    object_model_pixels_RGB_D, object_model_pixels_RGB, object_model_pixels_D =\
        prepare_object_model(object_model)

    if verbose:
        print("\tCalculating attributes for sp-tmp granules...")

    result_sp_tmp = calc_attributes(
        sp_tmp_granules, object_model_pixels_RGB_D, object_model, verbose)

    if verbose:
        print("\tCalculating attributes for rgb granules...")

    result_rgb = calc_attributes(
        rgb_granules, object_model_pixels_RGB, object_model, verbose)

    if verbose:
        print("\tCalculating attributes for d granules...")

    result_d = calc_attributes(
        d_granules, object_model_pixels_D, object_model, verbose)

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
