import permutation_utils
import color_stripped_amps
from sympy import symbols

def create_abstract_reflection_dict(n_gluons: int):
    refl_dict = {}
    for perm in permutation_utils.non_cyclic_perms(range(1, n_gluons + 1)):
        refl_perm = tuple([perm[0],] + list(perm[1:][::-1]))
        if refl_perm in refl_dict.keys():
            refl_dict[perm] = ((-1)**n_gluons, refl_perm)

        else:
            refl_dict[perm] = (1, perm)

    return refl_dict

def create_specific_reflection_dict(letter, lvl: int, abs_refl_dict: dict):
    n_gluons = len(list(abs_refl_dict.keys())[0])
    base_list = list(range(1, n_gluons + 1))
    sym_list = [symbols(f"{el}") for el in base_list]
    refl_dict = {}
    for key, value in abs_refl_dict.items():
        new_key = color_stripped_amps.abstract_cs_amp(letter, lvl, [sym_list[el-1] for el in key])
        new_val = value[0] * color_stripped_amps.abstract_cs_amp(letter, lvl, [sym_list[el-1] for el in value[1]])

        refl_dict[new_key] = new_val

    return refl_dict

def reflect_pdr(pdr_list: list, refl_dict: dict):
    refl_pdr_list = []
    for pdr in pdr_list:
        refl_pdr = []
        for el in pdr:
            refl_pdr.append(refl_dict[el])

        refl_pdr_list.append(refl_pdr)

    return (refl_pdr_list)