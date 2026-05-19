#%%
from IPython.display import display
from color_stripped_amps import *
from SUNDelta import SUNDelta
from format_utils import VisualDelta
import photon_decoupling
from sympy import symbols, init_printing, expand
from functools import partial
from time import time
import amplitude
import sun_utils
import sympy as sp
import math
import color_stripped_amps

def apply_pdr_relations_interference(opt:amplitude.settings, expr_collected, *pdr_appliers):
    """
    Like apply_pdr_relations, but works on the output of collect_by_Nc_then_B.
    Iterates over Nc-power tokens, then B-factor tokens, and applies PDR
    appliers directly on the remaining scalar coefficient.
    """
    final_expr = 0

    for nc_term in sp.Add.make_args(expr_collected):
        factors = sp.Mul.make_args(nc_term)
        nc_power = sp.Mul(*[f for f in factors if f.has(opt.sun_n)])
        coeff_no_nc = sp.Mul(*[f for f in factors if not f.has(opt.sun_n)])

        reduced_coeff = 0
        for b_term in sp.Add.make_args(coeff_no_nc):
            b_factors = []
            rest_factors = []
            for f in sp.Mul.make_args(b_term):
                if hasattr(f, 'base') and str(f.base).startswith(opt.cs_amp_letter):
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

print("started")
start = int(time())

N_GLUONS = 5
INTERFERENCE = False #False = tree lvl squared
DEBUG_PRINTING = False

APPLY_PDR = True
WITH_PARTIAL_PDR = False
EXPAND_SUBLEADING = True
APPLY_REFLECTION = True
APPLY_TREE_REFLECTION = True

APPLY_TREE_REFLECTION_AFTER = False
APPLY_REFLECTION_AFTER = False
APPLY_PDR_AFTER = False
WITH_PARTIAL_PDR_AFTER = False

ONLY_SUBLEADING = False

opt_loop = amplitude.settings()
opt_loop.n_gluons = N_GLUONS
opt_loop.cs_amp_letter = 'B'

opt_tree = amplitude.settings()
opt_tree.n_gluons = N_GLUONS

base_num_idx_list = symbols(f'1:{N_GLUONS+1}')
tree_lvl_pdr_rel = photon_decoupling.gen_tree_lvl_pdr(opt_tree.cs_amp_letter, base_num_idx_list)
tree_pdr_applier = partial(photon_decoupling.apply_tree_lvl_pdr, tree_lvl_pdr_rel)
partial_pdr_applier = [partial(photon_decoupling.apply_tree_lvl_pdr_partial, tree_lvl_pdr_rel, el) for el in range(math.floor((N_GLUONS - 1)/2) + 1, N_GLUONS)]

expr_tree = amplitude.generate_leading_amplitude(opt_tree, 0)
expr_tree = expr_tree.replace(SUNDelta, lambda a,b: SUNDelta(b, a))

if APPLY_TREE_REFLECTION:
    expr_tree = amplitude.apply_reflection(opt_tree, expr_tree, 0)

if DEBUG_PRINTING:
    debug_tree_expr = expr_tree

checkpoint = [start,]

print("tree generated")
checkpoint.append(int(time()))
print(f"time elapsed from previous step: {checkpoint[-1] - checkpoint[-2]}")
print()

if INTERFERENCE:

    if not ONLY_SUBLEADING:
        expr_loop = amplitude.generate_leading_amplitude(opt_loop, 1)

    else:
        expr_loop = amplitude.generate_only_leading_subleading(opt_loop)

    if DEBUG_PRINTING:
        debug_loop_expr = expr_loop

    if EXPAND_SUBLEADING:
        expr_loop = amplitude.expand_subleading(opt_loop, 3, expr_loop)
    
    if APPLY_REFLECTION:
        expr_loop = amplitude.apply_reflection(opt_loop, expr_loop)

else:
    expr_loop = amplitude.generate_leading_amplitude(opt_loop, 0)
    
    if DEBUG_PRINTING:
        debug_loop_expr = expr_loop
    
    if APPLY_REFLECTION:
        expr_loop = amplitude.apply_reflection(opt_loop, expr_loop, 0)


print("loop generated")
checkpoint.append(int(time()))
print(f"time elapsed from previous step: {checkpoint[-1] - checkpoint[-2]}")
print()

product = expr_tree*expr_loop
#print(product)

print("computed product")
checkpoint.append(int(time()))
print(f"time elapsed from previous step: {checkpoint[-1] - checkpoint[-2]}")
print()

#up_contraction = sun_utils.abstract_contract_deltas(opt_tree.sun_n, opt_tree.gellman_chain_up_idx_list, product)
#
#print("contracted first product")
#checkpoint.append(int(time()))
#print(f"time elapsed from previous step: {checkpoint[-1] - checkpoint[-2]}")
#print()
#
#down_contraction = sun_utils.abstract_contract_deltas(opt_tree.sun_n, opt_tree.gellman_chain_down_idx_list, up_contraction)
#
#print("contracted second product")
#checkpoint.append(int(time()))
#print(f"time elapsed from previous step: {checkpoint[-1] - checkpoint[-2]}")


down_contraction = sun_utils.compute_closed_cycles(opt_tree.sun_n, opt_tree.sun_delta, product)
print("contracted second product")
checkpoint.append(int(time()))
print(f"time elapsed from previous step: {checkpoint[-1] - checkpoint[-2]}")
print()



result = amplitude.collect_by_Nc_then_amp(opt_loop, down_contraction)

print("collected")
checkpoint.append(int(time()))
print(f"time elapsed from previous step: {checkpoint[-1] - checkpoint[-2]}")
print()


if APPLY_PDR:
    if WITH_PARTIAL_PDR:
        result = apply_pdr_relations_interference(opt_loop, result, tree_pdr_applier, *partial_pdr_applier, tree_pdr_applier)
    else:
        result = apply_pdr_relations_interference(opt_loop, result, tree_pdr_applier)

if APPLY_REFLECTION_AFTER:
    if INTERFERENCE:
        result = amplitude.apply_reflection(opt_loop, result)
    else:
        result = amplitude.apply_reflection(opt_loop, result, 0)

    result = amplitude.collect_by_Nc_then_amp(opt_loop, result)

if APPLY_TREE_REFLECTION_AFTER:
    result = amplitude. apply_reflection(opt_tree, expand(result))
    result = amplitude.collect_by_Nc_then_amp(opt_loop, result)
    
if APPLY_PDR_AFTER:
    if WITH_PARTIAL_PDR_AFTER:
        result = apply_pdr_relations_interference(opt_loop, result, tree_pdr_applier, *partial_pdr_applier, tree_pdr_applier)
    else:
        result = apply_pdr_relations_interference(opt_loop, result, tree_pdr_applier)


#if INTERFERENCE:
#    result = amplitude.apply_reflection(opt_loop, result)
#
#else:
#    result = amplitude.apply_reflection(opt_loop, result, 0)
#    result = amplitude.collect_by_Nc_then_amp(opt_loop, result)


print("decoupled")
checkpoint.append(int(time()))
print(f"time elapsed from previous step: {checkpoint[-1] - checkpoint[-2]}")
print()

init_printing(use_latex='mathjax')

if DEBUG_PRINTING:
    print("tree")
    print()
    display(debug_tree_expr.subs(SUNDelta, VisualDelta))
    print()
    print("loop")
    print()
    display(debug_loop_expr.subs(SUNDelta, VisualDelta))
    print()
    #print("non contracted product")
    #display(product.subs(SUNDelta, VisualDelta))
    #print()
    print("partial amp")
    expr = color_stripped_amps.abstract_cs_amp(opt_loop.cs_amp_letter, 3, base_num_idx_list)
    display(amplitude.expand_subleading(opt_loop, 3, expr))


print("contracted product")
display_expr = result.subs(SUNDelta, VisualDelta)
display(display_expr)