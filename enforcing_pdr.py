#%%
from IPython.display import display
import permutation_utils
from color_stripped_amps import *
import sun_utils
from SUNDelta import SUNDelta
from format_utils import VisualDelta, collect_color_structure
import fierz_identity
import photon_decoupling
from sympy import IndexedBase, symbols,expand, latex, pprint, init_printing
from functools import partial
from time import time
import sympy as sp
import math

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

start = int(time()*1000)

T = IndexedBase('T')
d = SUNDelta
Nc = symbols('Nc', positive = True, integer = True)

fierz = partial(fierz_identity.abstract_fierz, T, d, Nc)
gellman_chain = partial(sun_utils.abstract_gellmann_chain, T)
trace = partial(sun_utils.abstract_trace,T)
contract_deltas = partial(sun_utils.abstract_contract_deltas,Nc)

LEVEL = "1LOOP"
N_GLUON = 5
adj_idx_list = symbols(f'a1:{N_GLUON+1}')
base_num_idx_list = symbols(f'1:{N_GLUON+1}')

gchain = gellman_chain(adj_idx_list, 'i', 'j')

if LEVEL == "TREE":

    #INIT RELATIONS AND APPLIER
    tree_lvl_pdr_rel = photon_decoupling.gen_tree_lvl_pdr('A', base_num_idx_list)
    tree_pdr_applier = partial(photon_decoupling.apply_tree_lvl_pdr, tree_lvl_pdr_rel)

    expr = 0
    for perm in permutation_utils.non_cyclic_perms(range(1,N_GLUON+1)):
        num_idx_list = symbols([str(n) for n in perm])
        adj_idx_list = symbols([f'a{n}' for n in perm])
        cs_amp = abstract_cs_amp('A', 0, num_idx_list)

        tr, internal_tr_idx = trace(adj_idx_list, 'k', True)
        expr_fierz = fierz(tr*gchain*cs_amp)
        expr_contracted = contract_deltas(internal_tr_idx, expand(expr_fierz))

        expr += expr_contracted

    expr_collected = collect_color_structure(expr, Nc, SUNDelta)

    final_expr = apply_pdr_relations(Nc, SUNDelta, expr_collected, tree_pdr_applier)

elif LEVEL == "1LOOP":
    expr = 0
    for perm in permutation_utils.non_cyclic_perms(range(1,N_GLUON+1)):
        num_idx_list = symbols([str(n) for n in perm])
        adj_idx_list = symbols([f'a{n}' for n in perm])
        cs_amp = abstract_cs_amp('A', 1, num_idx_list)

        tr, internal_tr_idx = trace(adj_idx_list, 'k', True)
        expr_fierz = fierz(Nc*tr*gchain*cs_amp)
        expr_contracted = contract_deltas(internal_tr_idx, expand(expr_fierz))
        expr += expr_contracted

    max_lvl = math.floor(N_GLUON/2) + 1
    for lvl in range(2,max_lvl + 1):
        for perm in permutation_utils.double_trace_inv_perms(range(1,N_GLUON+1),lvl-1):
            
            num_idx_list = symbols([str(n) for n in perm])
            adj_idx_list = symbols([f'a{n}' for n in perm])
            cs_amp = abstract_cs_amp('A', lvl, num_idx_list)

            tr1, internal_tr1_idx = trace(adj_idx_list[:lvl-1], 'k', True)
            tr2, internal_tr2_idx = trace(adj_idx_list[lvl-1:], 'q', True)
            expr_fierz = fierz(tr1*tr2*gchain*cs_amp)
            expr_contracted_first = contract_deltas(internal_tr1_idx, expand(expr_fierz))
            expr_contracted_second = contract_deltas(internal_tr2_idx, expand(expr_contracted_first))
            expr += expr_contracted_second
           
    expr_collected = collect_color_structure(expr, Nc, SUNDelta) 

    if N_GLUON == 4:
        #INIT RELATIONS AND APPLIER
        g4_1loop_pdr_rel = photon_decoupling.gen_1loop_4g_pdr('A', base_num_idx_list)
        g4_1loop_pdr_applier = partial(photon_decoupling.apply_4g_1loop_pdr, g4_1loop_pdr_rel)
        
        #APPLY RELATIONS
        final_expr = apply_pdr_relations(Nc, SUNDelta, expr_collected, g4_1loop_pdr_applier)
    
    elif N_GLUON == 5:
        #INIT RELATIONS AND APPLIER
        g5_1loop_onlyA3_pdr_rel = photon_decoupling.gen_1loop_5g_onlyA3_pdr('A', base_num_idx_list)
        g5_1loop_A3A1_pdr_rel = photon_decoupling.gen_1loop_5g_A3A1_pdr('A',base_num_idx_list)
        g5_1loop_onlyA1_pdr_rel = photon_decoupling.gen_1loop_5g_onlyA1_pdr('A', base_num_idx_list)
        
        g5_1loop_pdr_applier = partial(photon_decoupling.apply_5g_1loop_pdr, g5_1loop_onlyA3_pdr_rel, g5_1loop_A3A1_pdr_rel, g5_1loop_onlyA1_pdr_rel)
        
        #APPLY RELATIONS
        final_expr = apply_pdr_relations(Nc, SUNDelta, expr_collected, g5_1loop_pdr_applier)
    
    else:
        final_expr = expr_collected

    #final_expr = expr_collected

end_computation = int(time()*1000)

init_printing(use_latex = 'mathjax')
display_expr = final_expr.subs(SUNDelta, VisualDelta)

display(display_expr)

end = int(time()*1000)
print("---------------------------------------------------------------------------------------")
print(f"computation time: {end_computation-start}")
print(f"drawing time: {end-end_computation}")