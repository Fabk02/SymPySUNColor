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


opt_tree_1 = amplitude.settings()
opt_tree_2 = amplitude.settings()
opt_tree_2.cs_amp_letter = 'B'

N_GLUONS = 4
base_num_idx_list = symbols(f'1:{N_GLUONS+1}')

up_idx_lst = symbols(f'{opt_tree_1.gellman_chain_up_idx}1:{N_GLUONS+1}')
down_idx_lst = symbols(f'{opt_tree_1.gellman_chain_down_idx}1:{N_GLUONS+1}')

contract_deltas = partial(sun_utils.abstract_contract_deltas,opt_tree_1.sun_n)

expr_tree_1 = amplitude.generate_amplitude(opt_tree_1, N_GLUONS, 0)
decoupled_tree_1 = photon_decoupling.apply_brutally_pdr(opt_tree_1.sun_n, N_GLUONS, expr_tree_1)
conj_tree_1 = decoupled_tree_1.replace(SUNDelta, lambda a,b: SUNDelta(b, a))

expr_tree_2 = amplitude.generate_amplitude(opt_tree_2, N_GLUONS, 0)
decoupled_tree_2 = photon_decoupling.apply_brutally_pdr(opt_tree_2.sun_n, N_GLUONS, expr_tree_2)

first_contr = contract_deltas(up_idx_lst, expand(conj_tree_1*decoupled_tree_2))
second_contr = contract_deltas(down_idx_lst, expand(first_contr))

tot_collected_by_tree_2 = collect_by_Nc_then_B(second_contr, opt_tree_2.sun_n, opt_tree_2.cs_amp_letter)

init_printing(use_latex = 'mathjax')
display_expr = second_contr
print("tree expr:")
display(conj_tree_1.subs(SUNDelta, VisualDelta))
print(len(sp.Add.make_args(expand(conj_tree_1))))
print('----------------------------------------------------------------------------------')
print("loop expr:")
display(decoupled_tree_2.subs(SUNDelta, VisualDelta))
print(len(sp.Add.make_args(expand(decoupled_tree_2))))
print('----------------------------------------------------------------------------------')
print("non contr product: " + str(len(sp.Add.make_args(expand(second_contr)))))
print('----------------------------------------------------------------------------------')
print("product:")
display(tot_collected_by_tree_2)
#print(len(sp.Add.make_args(expand(tot_collected_by_tree_2))))