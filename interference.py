#%%
from IPython.display import display
import amplitude
import photon_decoupling
import sun_utils
from SUNDelta import SUNDelta
from format_utils import VisualDelta
from sympy import init_printing, expand, symbols
from functools import partial
from collections import defaultdict
from sympy import Mul, Add, default_sort_key, Pow
import sympy as sp
import permutation_utils
import color_stripped_amps
from time import time

from sympy import symbols, Add, Mul, Integer, Abs

def count_addends_with_multiplicity(expr):
    """Count addends with their integer coefficients as multiplicity."""
    if not isinstance(expr, Add):
        return 1  # single term
    
    total = 0
    for term in expr.args:
        if isinstance(term, Mul):
            # Extract the numeric coefficient from the Mul
            coeff = term.args[0]  # SymPy always puts the number first
            if coeff.is_Number:
                total += int(Abs(coeff))
            else:
                total += 1  # no numeric coefficient, multiplicity = 1
        else:
            total += 1  # plain symbol, multiplicity = 1
    
    return total

def apply_pdr_relations_interference(SUNN, letter, expr_collected, *pdr_appliers):
    """
    Like apply_pdr_relations, but works on the output of collect_by_Nc_then_B.
    Iterates over Nc-power tokens, then B-factor tokens, and applies PDR
    appliers directly on the remaining scalar coefficient.
    """
    final_expr = 0

    for nc_term in sp.Add.make_args(expr_collected):
        factors = sp.Mul.make_args(nc_term)
        nc_power = sp.Mul(*[f for f in factors if f.has(SUNN)])
        coeff_no_nc = sp.Mul(*[f for f in factors if not f.has(SUNN)])

        reduced_coeff = 0
        for b_term in sp.Add.make_args(coeff_no_nc):
            b_factors = []
            rest_factors = []
            for f in sp.Mul.make_args(b_term):
                if hasattr(f, 'base') and str(f.base).startswith(letter):
                    b_factors.append(f)
                else:
                    rest_factors.append(f)

            b_product = sp.Mul(*b_factors)
            scalar = sp.Mul(*rest_factors)

            reduced_scalar = scalar
            for pdr_applier in pdr_appliers:
                reduced_scalar = pdr_applier(reduced_scalar)

            reduced_coeff += b_product * reduced_scalar

        final_expr += nc_power * reduced_coeff

    return final_expr

def collect_by_Nc_then_B(expr, SUNN, letter):
    """
    Returns a SymPy expression collected first by SUNN power, then by B factor.
    """
    # Group terms: {nc_power: {b_key: coeff}}
    grouped = defaultdict(lambda: defaultdict(int))
    
    for term in Add.make_args(expr):
        nc_power = term.as_powers_dict().get(SUNN, 0)
        coeff_no_nc = term / (SUNN**nc_power)
        
        b_factors = []
        rest = []
        for factor in Mul.make_args(coeff_no_nc):
            if hasattr(factor, 'base') and str(factor.base).startswith(letter):
                b_factors.append(factor)
            else:
                rest.append(factor)
        
        b_key = tuple(sorted(b_factors, key=default_sort_key))
        grouped[nc_power][b_key] += Mul(*rest)
    
    # Reconstruct as a single SymPy expression
    total = 0
    for nc_power, b_dict in grouped.items():
        for b_key, coeff in b_dict.items():
            total += Pow(SUNN, nc_power) * Mul(*b_key) * coeff
    
    return total

def expand_subleading(opt: amplitude.settings, n_gluons, sub_lvl, expr):
    num_idx_list = symbols(f'1:{n_gluons+1}')
    for perm in permutation_utils.double_trace_inv_perms(range(1, n_gluons + 1), sub_lvl - 1):
        sym_lst = [num_idx_list[p-1] for p in list(perm)]
        cs_amp_to_sub = color_stripped_amps.abstract_cs_amp(opt.cs_amp_letter, sub_lvl, sym_lst)
        
        new_expr = 0
        for cop in permutation_utils.cop(list(perm[:sub_lvl-1])[::-1], perm[sub_lvl-1:n_gluons-1]):
            right_idx = permutation_utils.cycle_order(list(cop) + [perm[n_gluons-1],])
            right_sym_lst = [num_idx_list[p-1] for p in list(right_idx)]
            new_expr += ((-1)**(sub_lvl - 1)) * color_stripped_amps.abstract_cs_amp(opt.cs_amp_letter, 1, right_sym_lst)

        expr = expr.subs(cs_amp_to_sub, new_expr)

    return expr

def apply_reflection(opt: amplitude.settings, n_gluons, expr):
    num_idx_list = symbols(f'1:{n_gluons+1}')
    base = []
    for perm in permutation_utils.non_cyclic_perms(range(1, n_gluons + 1)):
        if tuple([perm[0],] + list(perm[1:][::-1])) in base:
            straight_idx = [num_idx_list[p-1] for p in [perm[0],] + list(perm[1:][::-1])]
            reversed_idx = [num_idx_list[p-1] for p in list(perm)]
            
            for quark in [0,1]:
                target_amp = color_stripped_amps.abstract_cs_amp(opt.cs_amp_letter, 1, reversed_idx, quark)
                to_sub_amp = color_stripped_amps.abstract_cs_amp(opt.cs_amp_letter, 1, straight_idx, quark)
                expr = expr.subs(target_amp, ((-1)**n_gluons)*to_sub_amp)
 
        else:
            base.append(perm)

    return expr
            

start = int(time())

opt_tree = amplitude.settings()

opt_loop = amplitude.settings()
opt_loop.cs_amp_letter = 'B'

N_GLUONS = 4
base_num_idx_list = symbols(f'1:{N_GLUONS+1}')
tree_lvl_pdr_rel = photon_decoupling.gen_tree_lvl_pdr(opt_tree.cs_amp_letter, base_num_idx_list)
tree_pdr_applier = partial(photon_decoupling.apply_tree_lvl_pdr, tree_lvl_pdr_rel)

if N_GLUONS == 4:
        #INIT RELATIONS AND APPLIER
        g4_1loop_pdr_rel = photon_decoupling.gen_1loop_4g_pdr(opt_loop.cs_amp_letter, base_num_idx_list)
        loop_pdr_applier = partial(photon_decoupling.apply_4g_1loop_pdr, g4_1loop_pdr_rel)

if N_GLUONS == 5:
        #INIT RELATIONS AND APPLIER
    g5_1loop_onlyA3_pdr_rel = photon_decoupling.gen_1loop_5g_onlyA3_pdr(opt_loop.cs_amp_letter, base_num_idx_list)
    g5_1loop_A3A1_pdr_rel = photon_decoupling.gen_1loop_5g_A3A1_pdr(opt_loop.cs_amp_letter,base_num_idx_list)
    g5_1loop_onlyA1_pdr_rel = photon_decoupling.gen_1loop_5g_onlyA1_pdr(opt_loop.cs_amp_letter, base_num_idx_list)
    
    loop_pdr_applier = partial(photon_decoupling.apply_5g_1loop_pdr, g5_1loop_onlyA3_pdr_rel, g5_1loop_A3A1_pdr_rel, g5_1loop_onlyA1_pdr_rel)

up_idx_lst = symbols(f'{opt_tree.gellman_chain_up_idx}1:{N_GLUONS+1}')
down_idx_lst = symbols(f'{opt_tree.gellman_chain_down_idx}1:{N_GLUONS+1}')

contract_deltas = partial(sun_utils.abstract_contract_deltas,opt_tree.sun_n)

expr_tree = amplitude.generate_amplitude(opt_tree, N_GLUONS, 0)
decoupled_tree = photon_decoupling.apply_brutally_pdr(opt_tree.sun_n, N_GLUONS, expr_tree)
#decoupled_tree = amplitude.generate_leading_amplitude(opt_tree, N_GLUONS, 0)
conj_tree = decoupled_tree.replace(SUNDelta, lambda a,b: SUNDelta(b, a))

expr_loop = amplitude.generate_amplitude(opt_loop, N_GLUONS, 1)
decoupled_loop = photon_decoupling.apply_brutally_pdr(opt_loop.sun_n, N_GLUONS, expr_loop)
#decoupled_loop = amplitude.generate_leading_amplitude(opt_loop, N_GLUONS, 1)
decoupled_loop = expand_subleading(opt_loop, N_GLUONS, 3, decoupled_loop)
decoupled_loop = apply_reflection(opt_loop, N_GLUONS, decoupled_loop)
#decoupled_loop += amplitude.generate_loop_quark_amplitude(opt_loop, N_GLUONS)

first_contr = contract_deltas(up_idx_lst, expand(conj_tree*decoupled_loop))
second_contr = contract_deltas(down_idx_lst, expand(first_contr))

tot_collected_by_loop = collect_by_Nc_then_B(second_contr, opt_loop.sun_n, opt_loop.cs_amp_letter)
tot_reduced_by_tree = apply_pdr_relations_interference(opt_loop.sun_n, opt_loop.cs_amp_letter, tot_collected_by_loop, tree_pdr_applier)
tot_reduced_by_tree = apply_reflection(opt_loop, N_GLUONS, tot_reduced_by_tree)
tot_reduced_by_tree = apply_pdr_relations_interference(opt_loop.sun_n, opt_loop.cs_amp_letter, tot_reduced_by_tree, tree_pdr_applier)
tot_reduced_by_tree = collect_by_Nc_then_B(tot_reduced_by_tree, opt_loop.sun_n, opt_loop.cs_amp_letter)


init_printing(use_latex = 'mathjax')
display_expr = tot_reduced_by_tree
print("tree expr:")
display(conj_tree.subs(SUNDelta, VisualDelta))
print(count_addends_with_multiplicity(expand(conj_tree)))
print('----------------------------------------------------------------------------------')
print("loop expr:")
display(decoupled_loop.subs(SUNDelta, VisualDelta))
print(count_addends_with_multiplicity(expand(decoupled_loop)))
print('----------------------------------------------------------------------------------')
print("non contr product:")
display(tot_collected_by_loop)
print("non contr product: " + str(count_addends_with_multiplicity(expand(tot_collected_by_loop))))
print('----------------------------------------------------------------------------------')
print("product:")
display(display_expr)
print(count_addends_with_multiplicity(expand(tot_reduced_by_tree)))

end = int(time())

print(f"computation time: {end - start}")