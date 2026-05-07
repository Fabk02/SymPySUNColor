import permutation_utils
import sympy as sp
from sympy import Add
from color_stripped_amps import *

def apply_pdr_relations(SUNN, delta, expr_collected, *pdr_appliers):
    # Scroll through the collected expression to check PDR identities
    final_expr = 0

    # 1. Iterate over each Nc power "token"
    for nc_term in sp.Add.make_args(expr_collected):
        # Separate the Nc power from its coefficient
        factors = sp.Mul.make_args(nc_term)
        nc_power = sp.Mul(*[f for f in factors if f.has(SUNN)])
        coeff_sum = sp.Mul(*[f for f in factors if not f.has(SUNN)])
        
        # 2. Iterate over each specific color structure "token" inside this SUNN power
        reduced_coeff = 0
        for color_term in sp.Add.make_args(coeff_sum):
            # Separate the Delta chain from the sum of Amplitudes
            c_factors = sp.Mul.make_args(color_term)
            delta_chain = sp.Mul(*[f for f in c_factors if f.func == delta])
            amp_sum = sp.Mul(*[f for f in c_factors if f.func != delta])
            
            # 3. Apply the PDR "Check"
            # Since you use all(), this will ONLY trigger if the full identity is found
            # in this specific color structure.
            reduced_amp_sum = amp_sum
            for pdr_applier in pdr_appliers:
                reduced_amp_sum = pdr_applier(reduced_amp_sum)
            
            # If the PDR is correct and the identity is complete, 
            # this term will naturally algebraically simplify to 0.
            reduced_coeff += delta_chain * reduced_amp_sum
            
        final_expr += nc_power * reduced_coeff
    
    return final_expr

def subtract_relation(current_expr, rel, rel_expr):
    current_expr = sp.expand(current_expr)

    coeffs_e = [current_expr.coeff(addend) for addend in rel]
    coeffs_r = [rel_expr.coeff(addend) for addend in rel]

    if any(c == 0 for c in coeffs_e):
        return current_expr
    
    m_candidates = [c_e / c_r for c_e, c_r in zip(coeffs_e, coeffs_r)]

    if all(m > 0 for m in m_candidates):
        m = min(m_candidates)
        current_expr = sp.expand(current_expr - m * rel_expr)

    elif all(m < 0 for m in m_candidates):
        m = max(m_candidates)
        current_expr = sp.expand(current_expr - m * rel_expr)

    return current_expr

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

    current_expr = sp.expand(expr)
    for rel in pdr_lst:

        rel_expr = sum(rel)
        current_expr = subtract_relation(current_expr, rel, rel_expr)

        """ if all(current_expr.has(addend) for addend in rel):
            neg_sum = 0
            for addend in rel[1:]:
                neg_sum -= addend
            
            current_expr = current_expr.subs(rel[0], neg_sum) """
    
    return current_expr

def gen_1loop_4g_pdr(letter, idx_lst):
    n_gluons = 4
    pdr_lst = []
    for lhs_perm in permutation_utils.double_trace_inv_perms(range(1, n_gluons + 1),2):
                term_lst = []
                sym_pdr_term = [idx_lst[el - 1] for el in lhs_perm]
                term_lst.append(abstract_cs_amp(letter, 3 ,sym_pdr_term))

                for rhs_perm in permutation_utils.non_cyclic_perms(lhs_perm):
                    sym_pdr_term = [idx_lst[el - 1] for el in rhs_perm]
                    term_lst.append(abstract_cs_amp(letter, 1, sym_pdr_term))
                
                pdr_lst.append(term_lst)

    return pdr_lst


def gen_1loop_5g_onlyA3_pdr(letter, idx_lst):
    n_gluons = 5
    pdr_lst = []
    for decoupled_gluon in range(1, n_gluons + 1):

        if decoupled_gluon == 1:
            head_lst = [[2,el] for el in range(3,n_gluons + 1)]
        else:
            head_lst = [[1,el] for el in [x for x in range(2, n_gluons + 1) if x != decoupled_gluon]]

        for head in head_lst:
            term_lst = []
            uncomplete_tail = [el for el in range(2,n_gluons + 1) if el not in head and el != decoupled_gluon]
            
            first_term = head + permutation_utils.cycle_order(uncomplete_tail + [decoupled_gluon,])
            sym_pdr_term = [idx_lst[el - 1] for el in first_term]
            term_lst.append(abstract_cs_amp(letter, 3 ,sym_pdr_term))

            second_term = head + permutation_utils.cycle_order(uncomplete_tail[::-1] + [decoupled_gluon,])
            sym_pdr_term = [idx_lst[el - 1] for el in second_term]
            term_lst.append(abstract_cs_amp(letter, 3 ,sym_pdr_term))

            third_term = uncomplete_tail + permutation_utils.cycle_order(head + [decoupled_gluon,])
            sym_pdr_term = [idx_lst[el - 1] for el in third_term]
            term_lst.append(abstract_cs_amp(letter, 3 ,sym_pdr_term))
            
            fourth_term = uncomplete_tail + permutation_utils.cycle_order(head[::-1] + [decoupled_gluon,])
            sym_pdr_term = [idx_lst[el - 1] for el in fourth_term]
            term_lst.append(abstract_cs_amp(letter, 3 ,sym_pdr_term))

            pdr_lst.append(term_lst)
    
    return pdr_lst

def apply_4g_1loop_pdr(pdr_lst,expr):
    current_expr = sp.expand(expr)

    for rel in pdr_lst:

        rel_expr = rel[0] - sum(rel[1:])
        current_expr = subtract_relation(current_expr, rel, rel_expr)

        """ if all(current_expr.has(addend) for addend in rel):
            sum = 0
            for addend in rel[1:]:
                sum += addend
            
            current_expr = current_expr.subs(rel[0], sum)   """
    
    return current_expr

def gen_1loop_5g_A3A1_pdr(letter, idx_lst):
    n_gluons = len(idx_lst)
    pdr_lst = []

    for decoupled_gluon in range(1, n_gluons + 1):
        for other_gluon in [x for x in range(1, n_gluons + 1) if x != decoupled_gluon]:
            for remaining_perms in permutation_utils.non_cyclic_perms([x for x in range(1, n_gluons + 1) if x != decoupled_gluon and x != other_gluon]):
                
                term_lst = []
                term_lst.append(abstract_cs_amp(letter, 3, [idx_lst[el - 1] for el in (permutation_utils.cycle_order([decoupled_gluon,]+[other_gluon,]) + list(remaining_perms))]))
                
                for perm in permutation_utils.cop(remaining_perms, [other_gluon,]):
                    lst = permutation_utils.cycle_order(perm + tuple([decoupled_gluon,]))
                    term_lst.append(abstract_cs_amp(letter, 1, [idx_lst[el - 1] for el in lst]))
                
                pdr_lst.append(term_lst)
    
    return pdr_lst

def gen_1loop_5g_onlyA1_pdr(letter, idx_lst):
    n_gluon = 5
    pdr_lst = []
    term_lst = []
    for perm in permutation_utils.non_cyclic_perms(range(1, n_gluon + 1)):
        term_lst.append(abstract_cs_amp(letter, 1, [idx_lst[el - 1] for el in perm]))
    
    pdr_lst.append(term_lst)

    return pdr_lst


def apply_5g_1loop_pdr(only_A3_pdr_lst, A3A1_pdr_lst, only_A1_pdr_lst, expr):
    current_expr = expr


    for rel in only_A3_pdr_lst:

        rel_expr = sum(rel)
        current_expr = subtract_relation(current_expr, rel, rel_expr)

        """ if all(current_expr.has(addend) for addend in rel):
            neg_sum = 0
            for addend in rel[1:]:
                neg_sum -= addend
            
            current_expr = current_expr.subs(rel[0], neg_sum) """

    for rel in A3A1_pdr_lst:

        rel_expr = rel[0] - sum(rel[1:])
        current_expr = subtract_relation(current_expr, rel, rel_expr)

        """ if all(current_expr.has(addend) for addend in rel):
            sum = 0
            for addend in rel [1:]:
                sum += addend
            
            current_expr = current_expr.subs(rel[0],sum) """
        
    for rel in A3A1_pdr_lst:  
        if current_expr.is_Add: 
            current_expr = current_expr.subs(rel[0], sum(rel[1:]))
    
    for rel in only_A1_pdr_lst:
        rel_expr = sum(rel)
        current_expr = subtract_relation(current_expr, rel, rel_expr)

    return current_expr   

def apply_brutally_pdr(SUNN, n_gluons, expr):

    result = expr
    for exp in range(1,n_gluons+1):
        result = sp.Add(*[arg for arg in sp.Add.make_args(result) if not arg.has(1/SUNN**exp)])

    return result

""" N_GLUON = 5
num_idx_list = symbols(f'1:{N_GLUON+1}')
#print(gen_1loop_pdr('A', num_idx_list))
lst = gen_1loop_pdr('A', num_idx_list)
for el in lst:
    print(el)
print(len(lst)) """