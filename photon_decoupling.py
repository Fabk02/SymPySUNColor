import permutation_utils
from sympy import symbols
from color_stripped_amps import *

def gen_tree_lvl_pdr(letter, idx_lst):
    n_gluons = len(idx_lst)
    pdr_lst = []
    for head_gluon in range(1, n_gluons + 1):
        remaining_gluons = [gluon for gluon in range(1, n_gluons + 1) if gluon != head_gluon]
        
        for perm in permutation_utils.non_cyclic_perms(remaining_gluons):
            addend_lst = []
            for cyc_perm in permutation_utils.cyclic_perms(perm):
                non_sym_pdr_addend = permutation_utils.cycle_order(cyc_perm + (head_gluon,))
                sym_pdr_addend = []

                for el in non_sym_pdr_addend:
                    sym_pdr_addend.append(idx_lst[el-1])
                
                addend_lst.append(abstract_cs_amp(letter, 0, sym_pdr_addend))
            pdr_lst.append(addend_lst)

    return pdr_lst

def apply_tree_lvl_pdr(pdr_lst, expr):

    current_expr = expr
    for rel in pdr_lst:

        if all(current_expr.has(addend) for addend in rel):
            neg_sum = 0
            for addend in rel[1:]:
                neg_sum -= addend
            
            current_expr = current_expr.subs(rel[0], neg_sum)
    
    return current_expr