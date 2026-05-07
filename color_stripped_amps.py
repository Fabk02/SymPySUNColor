from sympy import symbols, IndexedBase

def abstract_cs_amp(letter, lvl, idx_lst, quark_loop = False, return_head = False):
    """
    Generate the color stripped factor

    It will generate an idx list with the letter, lvl and len(idx_lst)

    For example:
    - letter = "A"
    - lvl = 1
    - len(idx_lst) = 5

    -> IndexedBase = A_5;1

    lvl = 0 means tree lvl
    """

    n = len(idx_lst)

    if quark_loop:
        idx_base = IndexedBase(f'{letter}^q_{n};{lvl}')
    else:
        idx_base = IndexedBase(f'{letter}_{n};{lvl}')

    if return_head:
        return idx_base[*idx_lst], idx_base
    else:
        return idx_base[*idx_lst]
