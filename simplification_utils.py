#%%
from IPython.display import display
from SUNDelta import SUNDelta
from format_utils import VisualDelta
from sympy import symbols, init_printing, expand, Matrix, Add, Mul
import sympy as sp
import amplitude
import color_stripped_amps
import photon_decoupling
import sun_utils
import permutation_utils
import numpy as np
from scipy.linalg import orth
import math
import pulp


def create_translation_dict(opt: amplitude.settings, lvl: int):
    transl_dict = {}

    for perm in permutation_utils.non_cyclic_perms(range(1, opt.n_gluons + 1)):
        sym_list = [symbols(f"{el}") for el in perm]
        transl_dict[color_stripped_amps.abstract_cs_amp(opt.cs_amp_letter, lvl, sym_list)] = 0

    return transl_dict

def get_valued_list(transl_dict: dict):
    val_lst = []

    for _, value in transl_dict.items():
        val_lst.append(value)
    
    return val_lst

def translate_pdr(transl_dict: dict, pdr_list: list):
    transl_dict = dict.fromkeys(transl_dict,0)
    transl_pdr_list = []
    for pdr in pdr_list:
        for factor in pdr:
            coeff, rest = factor.as_coeff_Mul()
            transl_dict[rest] += coeff

        transl_pdr_list.append(get_valued_list(transl_dict))
        transl_dict = dict.fromkeys(transl_dict,0)

    return transl_pdr_list

def translate_factor(transl_dict: dict, expr):
    transl_dict = dict.fromkeys(transl_dict,0)
    for addend in Add.make_args(expr):
        coeff, rest = addend.as_coeff_Mul()
        transl_dict[rest] += coeff 
    return get_valued_list(transl_dict)

def get_pdr_basis(pdr_list: list):
    M = Matrix(pdr_list)
    H,_ = M.T.hermite_normal_form()

    ind = []

    for i,row in enumerate(H.T.tolist()):
        if any(x != 0 for x in row):
            ind.append(row)

    return ind

def get_reduced_expr(start: list, basis: list):
    reachable_axis = []
    for coord, start_coord in enumerate(start):
        basis_coord_list = [el[coord] for el in basis]
        current_gcd = math.gcd(*basis_coord_list)

        if current_gcd == 0:
            reachable_axis.append(start_coord == 0)

        else:
            reachable_axis.append(start_coord % current_gcd == 0)

    return

def find_sparsest_lattice_point(start: list[int], basis: list[list[int]], c_max: int = 20):
    n = len(start)
    d = len(basis)
    
    # Initialize the Maximization problem
    prob = pulp.LpProblem("Max_Zero_Coordinates", pulp.LpMaximize)
    
    # Decision Variables
    c = [pulp.LpVariable(f"c_{j}", lowBound=-c_max, upBound=c_max, cat='Integer') for j in range(d)]
    z = [pulp.LpVariable(f"z_{i}", cat='Binary') for i in range(n)]
    
    # --- REGULARIZATION: Define variables for |c_j| ---
    abs_c = [pulp.LpVariable(f"abs_c_{j}", lowBound=0, cat='Integer') for j in range(d)]
    for j in range(d):
        prob += abs_c[j] >= c[j], f"Abs_Pos_{j}"
        prob += abs_c[j] >= -c[j], f"Abs_Neg_{j}"
    
    # --- FIXED OBJECTIVE FUNCTION ---
    # Maximize zeros primarily (huge weight), minimize coefficient sizes secondarily (small weight)
    # This guarantees the solver won't sacrifice a zero just to get smaller coefficients.
    prob += 1000000 * pulp.lpSum(z) - pulp.lpSum(abs_c)
    
    # Constraints for each coordinate
    for i in range(n):
        coord_expr = start[i] + pulp.lpSum(c[j] * basis[j][i] for j in range(d))
        M_i = abs(start[i]) + c_max * sum(abs(basis[j][i]) for j in range(d))
        
        prob += coord_expr <= M_i * (1 - z[i]), f"Upper_Bound_Coord_{i}"
        prob += coord_expr >= -M_i * (1 - z[i]), f"Lower_Bound_Coord_{i}"
        
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[status] == "Optimal":
        optimal_coefficients = [int(c[j].varValue) for j in range(d)]
        
        final_point = []
        for i in range(n):
            val = start[i] + sum(optimal_coefficients[j] * basis[j][i] for j in range(d))
            final_point.append(val)
            
        zeros_found = sum(1 for val in final_point if val == 0)
        
        return {
            "status": "Success",
            "max_zeros": zeros_found,
            "lattice_point": final_point,
            "coefficients": optimal_coefficients
        }
    else:
        return {"status": "No solution found within coefficient bounds."}

def simplify_expr(transl_dict: dict, translated_pdr: list, expr):
    new_expr = 0
    for addend in Add.make_args(expr):

        if (len(addend.args) != 3):
            new_expr += addend
            continue

        amplitude_factor = addend.args[1]

        translated_factor = translate_factor(transl_dict, amplitude_factor)

        reduced_dict = find_sparsest_lattice_point(translated_factor, translated_pdr)

        if reduced_dict["status"] == "Success":
            result = reduced_dict["lattice_point"]
            #print("Here")
            #print(result)

        else:
            result = translated_factor

        reduced_text_factor = 0
        for key, res in zip(transl_dict.keys(), result):
            if res != 0:
                reduced_text_factor += res * key

        new_expr += addend.args[0] * reduced_text_factor * addend.args[2]

    return new_expr
    