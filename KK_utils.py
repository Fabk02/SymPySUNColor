import permutation_utils
from sympy import symbols,Add
import color_stripped_amps
import amplitude

def kk_rel(gluons_list: list, fixed_part: int):
    fixed_part_pos = gluons_list.index(fixed_part)
    alpha_set = gluons_list[1:fixed_part_pos]
    beta_set = gluons_list[fixed_part_pos+1:]

    n_beta = len(beta_set)

    kk_list = []

    for shuffle in permutation_utils.shuffle_product(alpha_set, beta_set[::-1]):

        kk_list.append([1,] + shuffle + [fixed_part,])

    return (-1)**n_beta, kk_list

def gen_kk_rel(n_gluons: int):
    kk_dict = {}
    for perm in permutation_utils.non_cyclic_perms(range(1,n_gluons + 1)):
        if perm[n_gluons-1] == n_gluons:
            kk_dict[perm] = (1, [perm,])

        else:
            kk_dict[perm] = kk_rel(perm, n_gluons)

    return kk_dict

def gen_dressed_kk_rel(opt: amplitude.settings):
    
    kk_dict = gen_kk_rel(opt.n_gluons)
    kk_dressed_dict = {}

    for key, value in kk_dict.items():
        expr_dressed_value = 0
        for el in value[1]:
            sym_list = [symbols(f'{x}') for x in el]
            expr_dressed_value += color_stripped_amps.abstract_cs_amp(opt.cs_amp_letter, 0, sym_list)

        expr_dressed_value *= value[0]

        sym_list = [symbols(f'{x}') for x in key]
        dressed_key = color_stripped_amps.abstract_cs_amp(opt.cs_amp_letter, 0, sym_list)

        kk_dressed_dict[dressed_key] = expr_dressed_value

    return kk_dressed_dict

def gen_kk_pdr(kk_dict: dict, pdr_list: list):

    new_pdr_list = []

    for pdr in pdr_list:

        new_pdr = []

        for el in pdr:
            list_expr = Add.make_args(kk_dict[el])
            new_pdr += list(list_expr)

        new_pdr_list.append(new_pdr)

    return new_pdr_list

            