#%%
from IPython.display import display
from SUNDelta import SUNDelta
from format_utils import VisualDelta
from sympy import symbols, init_printing, expand, Matrix
import sympy as sp
import amplitude
import color_stripped_amps
import photon_decoupling
import sun_utils
import permutation_utils
import numpy as np
from scipy.linalg import orth
import simplification_utils

def analyze_subspace(vectors):
    """
    Returns the dimension and a basis consisting of a 
    subset of the original vectors.
    """
    # 1. Convert to a Matrix where original vectors are columns
    # SymPy handles exact arithmetic, avoiding floating-point issues
    M = Matrix(vectors).T
    
    # 2. Get Reduced Row Echelon Form
    # pivot_indices identifies which original columns are linearly independent
    rref_matrix, pivot_indices = M.rref()
    
    # 3. Extract the original vectors that correspond to the pivots
    basis = [vectors[i] for i in pivot_indices]
    dimension = len(basis)
    
    return dimension, basis

def gen_single_line_expr(opt: amplitude.settings, sequence: list):
    if len(sequence) != opt.n_gluons:
        raise Exception("sequence not matching with number of gluons")

    delta_chain = 1
    for idx in sequence[: opt.n_gluons - 1]:

        up_idx = symbols(f"{opt.gellman_chain_up_idx}{idx}")
        down_idx = symbols(f"{opt.gellman_chain_down_idx}{idx + 1}")
        delta_chain *= SUNDelta(up_idx, down_idx)
    
    up_idx = symbols(f"{opt.gellman_chain_up_idx}{opt.n_gluons}")
    down_idx = symbols(f"{opt.gellman_chain_down_idx}{1}")
    delta_chain *= SUNDelta(up_idx, down_idx)

    idx_list = [symbols(f"{el}") for el in sequence]

    return delta_chain * color_stripped_amps.abstract_cs_amp(opt.cs_amp_letter, 0, idx_list)

def gen_factored_amplitude_list(expr_collected, opt: amplitude.settings, ref_dict: dict, power: int):

    tot_list = []
    for nc_term in sp.Add.make_args(expr_collected):
        factors = sp.Mul.make_args(nc_term)
        nc_power = sp.Mul(*[f for f in factors if f.has(opt.sun_n)])

        if nc_power != (opt.sun_n**power):
            continue
        
        clean_dict = dict.fromkeys(ref_dict, 0)
        coeff_no_nc = sp.Mul(*[f for f in factors if not f.has(opt.sun_n)])
        for b_term in sp.Add.make_args(coeff_no_nc):
            rest_factors = []
            for f in sp.Mul.make_args(b_term):
                if not (hasattr(f, 'base') and str(f.base).startswith(opt.cs_amp_letter)):
                    for atom in sp.Add.make_args(f):
                        #rest_factors.append(atom)
                        coeff, bare = atom.as_coeff_Mul()
                            
                        clean_dict[bare] += coeff
        tot_list.append(list(clean_dict.values()))
        #tot_list.append(rest_factors)

    return tot_list

def gen_pdr_binary_list(pdr_list: list, ref_dict:dict):
    full_lst = []
    for pdr in pdr_list:
        clean_dict = dict.fromkeys(ref_dict, 0)
        for el in pdr:
            coeff, bare = el.as_coeff_Mul()
                
            clean_dict[bare] += coeff
        
        full_lst.append(list(clean_dict.values()))

    return full_lst

def target_list(sequence: list, opt: amplitude.settings, ref_dict:dict):
    sym_lst = [symbols(f"{el}") for el in sequence]
    amp = color_stripped_amps.abstract_cs_amp(opt.cs_amp_letter, 0, sym_lst)
    clean_dict = dict.fromkeys(ref_dict, 0)
    clean_dict[amp] -= 1

    return list(clean_dict.values())

def reflected_target(amp, ref_dict: dict):
    coeff, bare = amp.as_coeff_Mul()
    clean_dict = dict.fromkeys(ref_dict, 0)
    clean_dict[bare] -= coeff

    return list(clean_dict.values())

def can_reach_target(start, others, target):
    """
    Find integer coefficients c_i (unrestricted) such that:
        start + sum(c_i * others[i]) == target
    
    Returns (True, coefficients) or (False, None)
    """
    diff = np.array(target, dtype=float) - np.array(start, dtype=float)
    A = np.array(others, dtype=float).T  # shape: (length, num_lists)

    # Check if diff is in the column span of A
    # i.e. solve A @ c = diff in the least-squares sense, then verify
    if A.size == 0:
        return (True, []) if np.all(diff == 0) else (False, None)

    coeffs, residuals, rank, sv = np.linalg.lstsq(A, diff, rcond=None)

    # Check if the solution actually reconstructs diff (within float tolerance)
    reconstruction = A @ coeffs
    if not np.allclose(reconstruction, diff, atol=1e-6):
        return False, None  # diff is not in the column span at all

    # Check if we can round to integers and still hit target exactly
    int_coeffs = np.round(coeffs).astype(int)
    if np.array_equal(A @ int_coeffs, diff):
        return True, int_coeffs.tolist()

    # If rounding failed, try a small search around the rounded solution
    # (handles cases where the least-squares solution isn't the integer one)
    from itertools import product
    n = len(others)
    offsets = list(product([-1, 0, 1], repeat=n))
    for offset in offsets:
        candidate = int_coeffs + np.array(offset)
        if np.array_equal((A @ candidate).astype(int), diff.astype(int)):
            return True, candidate.tolist()

    return False, None

def is_in_int_span(start, basis, target):

    if (len(start) != len(basis[0]) != len(target)):
        raise ValueError

    subtraction = [ el[1] - el[0] for el in zip(target, start)]

    print(subtraction)

    A = Matrix(basis).T
    b = Matrix(subtraction)

    #augmented = A.row_join(b)
    #H, _ = augmented.hermite_normal_form(with_transform = True)

    try:
        solution = A.solve(b)
        print(solution)
        if all(val.is_Integer for val in solution):
            return True, [int(val) for val in solution]
        
        else:
            return False, None
    except ValueError:
        return False, None


LOOP_SEQUENCE = [1,2,3,4,5]
N_GLUONS = len(LOOP_SEQUENCE)
SUBLEADING_ORDER = N_GLUONS - 2

APPLY_REFLECTION = False

opt_tree = amplitude.settings()
opt_tree.n_gluons = N_GLUONS

opt_loop = amplitude.settings()
opt_loop.n_gluons = N_GLUONS
opt_loop.cs_amp_letter = 'B'

base_num_idx_list = symbols(f'1:{N_GLUONS+1}')

tree_lvl_pdr_rel = photon_decoupling.gen_tree_lvl_pdr(opt_tree.cs_amp_letter, base_num_idx_list)

expr_tree = amplitude.generate_leading_amplitude(opt_tree, 0)
expr_tree = expr_tree.replace(SUNDelta, lambda a,b: SUNDelta(b, a))

if APPLY_REFLECTION:
    reflection_dict = {}
    full_dict = {}
    seen_list = []
    for perm in permutation_utils.non_cyclic_perms(range(1, opt_tree.n_gluons + 1)):
        sym_perm = [symbols(f"{el}") for el in perm]
        reflected_idx = [perm[0],] + list(perm[1:][::-1])
        if reflected_idx in seen_list:
            sym_refl = [symbols(f"{el}") for el in reflected_idx]
            reflection_dict[color_stripped_amps.abstract_cs_amp(opt_tree.cs_amp_letter, 0, sym_perm)] = ((-1)**N_GLUONS) * color_stripped_amps.abstract_cs_amp(opt_tree.cs_amp_letter, 0, sym_refl)
        
        else:
            seen_list.append(list(perm))
            reflection_dict[color_stripped_amps.abstract_cs_amp(opt_tree.cs_amp_letter, 0, sym_perm)] = color_stripped_amps.abstract_cs_amp(opt_tree.cs_amp_letter, 0, sym_perm)
            full_dict[color_stripped_amps.abstract_cs_amp(opt_tree.cs_amp_letter, 0, sym_perm)] = 0

    tree_lvl_pdr_rel = [
        [reflection_dict[el] for el in pdr]
            for pdr in tree_lvl_pdr_rel
    ]       

    #for (key, value) in reflection_dict.items():
    #    expr_tree = expr_tree.replace(key, value)

    expr_tree = expr_tree.subs(reflection_dict)

else:
    full_dict = {}
    for perm in permutation_utils.non_cyclic_perms(range(1, opt_tree.n_gluons + 1)):
        sym_perm = [symbols(f"{el}") for el in perm]
        full_dict[color_stripped_amps.abstract_cs_amp(opt_tree.cs_amp_letter, 0, sym_perm)] = 0

expr_loop = gen_single_line_expr(opt_loop, LOOP_SEQUENCE)

product = expr_tree * expr_loop

expr = sun_utils.compute_closed_cycles(opt_tree.sun_n, opt_tree.sun_delta, product)
expr = amplitude.collect_by_Nc_then_amp(opt_loop, expr)

start = gen_factored_amplitude_list(expr, opt_loop, full_dict, SUBLEADING_ORDER)[0]
others = gen_pdr_binary_list(tree_lvl_pdr_rel, full_dict)

if APPLY_REFLECTION:
    sym = [symbols(f"{el}") for el in LOOP_SEQUENCE]
    seq_amp = color_stripped_amps.abstract_cs_amp(opt_tree.cs_amp_letter, 0, sym)
    refl_amp = reflection_dict[seq_amp]
    target = reflected_target(refl_amp, full_dict)

else:
    target = target_list(LOOP_SEQUENCE, opt_tree, full_dict)

res = simplification_utils.find_sparsest_lattice_point(start, others)

init_printing(use_latex='mathjax')

display_expr = expr.subs(SUNDelta, VisualDelta)

display(full_dict)
display(display_expr)
display(tree_lvl_pdr_rel)
print(start)
print(others)
print(target)
print(is_in_int_span(start, others, target))
print(res)


#for i in range(len(target)):
#    target[i] += 1
#    for j in range(len(target)):
#        target[j] += 1
#        print(target)
#        print(can_reach_target(start, others, target))
#        target[j] -= 1
#    target[i] -= 1
