from sympy import symbols, expand
from functools import partial
import sun_utils
import permutation_utils
from color_stripped_amps import *
import fierz_identity
import math
from SUNDelta import SUNDelta
from format_utils import collect_color_structure
import itertools
from collections import defaultdict
from sympy import Mul, Add, default_sort_key, Pow

class settings:
    n_gluons = 4
    n_quarks = 0
    sun_tensor = IndexedBase('T')
    sun_delta = SUNDelta
    sun_n = symbols('Nc', positive = True, integer = True)
    sun_adjoint_idx = 'a'
    nf = symbols('nf', positive = True, integer = True)
    cs_amp_letter = 'A'
    gellman_chain_up_idx = 'i'
    gellman_chain_down_idx = 'j'
    first_tr_internal_idx = 'k'
    second_tr_internal_idx = 'p'
    q = 'q'
    qbar = r'\bar{q}'

    @property
    def n_particles(self):
        return self.n_gluons + self.n_quarks

    @property
    def base_adj_idx_list(self):
        return symbols(f'{self.sun_adjoint_idx}{self.n_quarks + 1}:{self.n_particles + 1}')
    
    @property
    def fierz(self):
        return partial(fierz_identity.abstract_fierz, self.sun_tensor, self.sun_delta, self.sun_n)

    @property
    def gellman_chain(self):
        return sun_utils.abstract_gellmann_chain(self.sun_tensor, self.base_adj_idx_list, self.gellman_chain_up_idx, self.gellman_chain_down_idx, False, self.n_quarks)
    
    @property
    def gellman_chain_up_idx_list(self):
        _, up_lst, _ = sun_utils.abstract_gellmann_chain(self.sun_tensor, self.base_adj_idx_list, self.gellman_chain_up_idx, self.gellman_chain_down_idx, True, self.n_quarks)
        return up_lst
    
    @property
    def gellman_chain_down_idx_list(self):
        _, _, down_lst = sun_utils.abstract_gellmann_chain(self.sun_tensor, self.base_adj_idx_list, self.gellman_chain_up_idx, self.gellman_chain_down_idx, True, self.n_quarks)
        return down_lst
    
    @property
    def trace(self):
        return partial(sun_utils.abstract_trace, self.sun_tensor)
    
    @property
    def first_tr_internal_idx_list(self):
        return symbols(f'{self.first_tr_internal_idx}1:{self.n_gluons+1}')
    
    @property
    def second_tr_internal_idx_list(self):
        return symbols(f'{self.second_tr_internal_idx}1:{self.n_gluons+1}')
    
    @property
    def contract_deltas_first(self):
        return partial(sun_utils.abstract_contract_deltas, self.sun_n, self.first_tr_internal_idx_list)
    
    @property
    def contract_deltas_second(self):
        return partial(sun_utils.abstract_contract_deltas, self.sun_n, self.second_tr_internal_idx_list)
    
    @property
    def leading_fierz(self):
        return partial(fierz_identity.abstract_leading_fierz, self.sun_tensor, self.sun_delta)


def generate_base_amplitude(opt:settings, loop_lvl: int, quark_loop: bool = False):
    
    expr = 0
    for perm in permutation_utils.non_cyclic_perms(range(1,opt.n_gluons + 1)):
        num_idx_list = symbols([str(n) for n in perm])
        adj_idx_list = symbols([f'{opt.sun_adjoint_idx}{n}' for n in perm])
        cs_amp = abstract_cs_amp(opt.cs_amp_letter, loop_lvl, num_idx_list, quark_loop)

        tr = opt.trace(adj_idx_list, opt.first_tr_internal_idx)
        expr_fierz = opt.fierz(tr*opt.gellman_chain*cs_amp)
        expr_contracted = opt.contract_deltas_first(expand(expr_fierz))

        expr += expr_contracted

    expr_collected = collect_color_structure(expr, opt.sun_n, opt.sun_delta)

    return expr_collected

def generate_amplitude(opt:settings, loop_lvl: int):
    
    if loop_lvl == 0:
        return generate_base_amplitude(opt, loop_lvl)
    
    if loop_lvl == 1:
        expr = opt.sun_n * generate_base_amplitude(opt, loop_lvl)
        
        max_lvl = math.floor(opt.n_gluons/2) + 1
        for lvl in range(2,max_lvl + 1):
            for perm in permutation_utils.double_trace_inv_perms(range(1,opt.n_gluons+1),lvl-1):
                
                num_idx_list = symbols([str(n) for n in perm])
                adj_idx_list = symbols([f'{opt.sun_adjoint_idx}{n}' for n in perm])
                cs_amp = abstract_cs_amp(opt.cs_amp_letter, lvl, num_idx_list)

                tr1 = opt.trace(adj_idx_list[:lvl-1], opt.first_tr_internal_idx)
                tr2 = opt.trace(adj_idx_list[lvl-1:], opt.second_tr_internal_idx)
                expr_fierz = opt.fierz(tr1*tr2*opt.gellman_chain*cs_amp)
                expr_contracted_first = opt.contract_deltas_first(expand(expr_fierz))
                expr_contracted_second = opt.contract_deltas_second(expand(expr_contracted_first))
                expr += expr_contracted_second
            
        expr_collected = collect_color_structure(expr, opt.sun_n, opt.sun_delta)

        return expr_collected

    else:
        return 0
    
def generate_leading_base_amplitude(opt:settings, loop_lvl: int, quark_loop: bool = False):

    expr = 0
    for perm in permutation_utils.non_cyclic_perms(range(1,opt.n_gluons + 1)):
        num_idx_list = symbols([str(n) for n in perm])
        adj_idx_list = symbols([f'{opt.sun_adjoint_idx}{n}' for n in perm])
        cs_amp = abstract_cs_amp(opt.cs_amp_letter, loop_lvl, num_idx_list, quark_loop)

        tr = opt.trace(adj_idx_list, opt.first_tr_internal_idx)
        expr_fierz = opt.leading_fierz(tr*opt.gellman_chain*cs_amp)
        expr_contracted = opt.contract_deltas_first(expand(expr_fierz))

        expr += expr_contracted

    expr_collected = collect_color_structure(expr, opt.sun_n, opt.sun_delta)

    return expr_collected

def generate_leading_amplitude(opt:settings, loop_lvl: int):

    if loop_lvl == 0:
        return generate_leading_base_amplitude(opt, loop_lvl)
    
    if loop_lvl == 1:
        expr = opt.sun_n * generate_leading_base_amplitude(opt, loop_lvl)
        
        max_lvl = math.floor(opt.n_gluons/2) + 1
        for lvl in range(3,max_lvl + 1):
            for perm in permutation_utils.double_trace_inv_perms(range(1,opt.n_gluons+1),lvl-1):
                
                num_idx_list = symbols([str(n) for n in perm])
                adj_idx_list = symbols([f'{opt.sun_adjoint_idx}{n}' for n in perm])
                cs_amp = abstract_cs_amp(opt.cs_amp_letter, lvl, num_idx_list)

                tr1 = opt.trace(adj_idx_list[:lvl-1], opt.first_tr_internal_idx)
                tr2 = opt.trace(adj_idx_list[lvl-1:], opt.second_tr_internal_idx)
                expr_fierz = opt.leading_fierz(tr1*tr2*opt.gellman_chain*cs_amp)
                expr_contracted_first = opt.contract_deltas_first(expand(expr_fierz))
                expr_contracted_second = opt.contract_deltas_second(expand(expr_contracted_first))
                expr += expr_contracted_second
            
        expr_collected = collect_color_structure(expr, opt.sun_n, opt.sun_delta)

        return expr_collected

    else:
        return 0
    
def generate_loop_quark_amplitude(opt:settings, n_gluons: int):
    return (opt.nf * generate_base_amplitude(opt, n_gluons, 1, True))

def generate_qq_amplitude(opt: settings, n_gluons: int, loop_lvl: int):
    n_quarks = 2
    tot_part = n_gluons + n_quarks
    all_perms = [p for p in itertools.permutations(range(n_quarks+1, tot_part + 1))]

    dummy_quark_up_idx = symbols(opt.gellman_chain_up_idx + opt.q)
    dummy_quark_down_idx = symbols(opt.gellman_chain_down_idx + opt.qbar)

    q_sym_lst = symbols(opt.q + ' ' + opt.qbar)

    gellmann_prod = partial(sun_utils.abstract_gellmann_product, opt.sun_tensor)
    gellmann_chain = partial(sun_utils.abstract_gellmann_chain, opt.sun_tensor)
    fierz = partial(fierz_identity.abstract_fierz,opt.sun_tensor, opt.sun_delta, opt.sun_n)
    contract_deltas = partial(sun_utils.abstract_contract_deltas,opt.sun_n)

    adj_idx_list = symbols(f'{opt.sun_adjoint_idx}{n_quarks + 1}:{tot_part + 1}')
    gchain = gellmann_chain(adj_idx_list, opt.gellman_chain_up_idx, opt.gellman_chain_down_idx, False, n_quarks)

    if loop_lvl == 0:
        expr = 0
        for perm in all_perms:
            num_idx_list = symbols([str(n) for n in perm])
            adj_idx_list = symbols([f'{opt.sun_adjoint_idx}{n}' for n in perm])
            cs_amp = abstract_qq_cs_amp(opt.cs_amp_letter, 0, q_sym_lst, num_idx_list)
            prod, internal_idx = gellmann_prod(adj_idx_list, opt.first_tr_internal_idx, dummy_quark_up_idx, dummy_quark_down_idx, True)
            expr_fierz = fierz(prod*gchain*cs_amp)
            expr_contracted = contract_deltas(internal_idx, expand(expr_fierz))
            expr += expr_contracted

        expr_collected = collect_color_structure(expr, opt.sun_n, opt.sun_delta)
        return expr_collected
    
    else:
        return 0

def expand_subleading(opt: settings, sub_lvl, expr):
    num_idx_list = symbols(f'1:{opt.n_gluons + 1}')
    for perm in permutation_utils.double_trace_inv_perms(range(1, opt.n_gluons + 1), sub_lvl - 1):
        sym_lst = [num_idx_list[p-1] for p in list(perm)]
        cs_amp_to_sub = abstract_cs_amp(opt.cs_amp_letter, sub_lvl, sym_lst)
        
        new_expr = 0
        for cop in permutation_utils.cop(list(perm[:sub_lvl-1])[::-1], perm[sub_lvl-1:opt.n_gluons-1]):
            right_idx = permutation_utils.cycle_order(list(cop) + [perm[opt.n_gluons-1],])
            right_sym_lst = [num_idx_list[p-1] for p in list(right_idx)]
            new_expr += ((-1)**(sub_lvl - 1)) * abstract_cs_amp(opt.cs_amp_letter, 1, right_sym_lst)

        expr = expr.subs(cs_amp_to_sub, new_expr)

    return expr

def apply_reflection(opt: settings, expr, lvl = 1):
    num_idx_list = symbols(f'1:{opt.n_gluons + 1}')
    base = []
    for perm in permutation_utils.non_cyclic_perms(range(1, opt.n_gluons + 1)):
        if tuple([perm[0],] + list(perm[1:][::-1])) in base:
            straight_idx = [num_idx_list[p-1] for p in [perm[0],] + list(perm[1:][::-1])]
            reversed_idx = [num_idx_list[p-1] for p in list(perm)]
            
            for quark in [0,1]:
                target_amp = abstract_cs_amp(opt.cs_amp_letter, lvl, reversed_idx, quark)
                to_sub_amp = abstract_cs_amp(opt.cs_amp_letter, lvl, straight_idx, quark)
                expr = expr.subs(target_amp, ((-1)**opt.n_gluons)*to_sub_amp)
 
        else:
            base.append(perm)

    return expr

def collect_by_Nc_then_amp(opt: settings, expr):
    """
    Returns a SymPy expression collected first by SUNN power, then by B factor.
    """
    # Group terms: {nc_power: {b_key: coeff}}
    grouped = defaultdict(lambda: defaultdict(int))
    
    for term in Add.make_args(expr):
        nc_power = term.as_powers_dict().get(opt.sun_n, 0)
        coeff_no_nc = term / (opt.sun_n**nc_power)
        
        b_factors = []
        rest = []
        for factor in Mul.make_args(coeff_no_nc):
            if hasattr(factor, 'base') and str(factor.base).startswith(opt.cs_amp_letter):
                b_factors.append(factor)
            else:
                rest.append(factor)
        
        b_key = tuple(sorted(b_factors, key=default_sort_key))
        grouped[nc_power][b_key] += Mul(*rest)
    
    # Reconstruct as a single SymPy expression
    total = 0
    for nc_power, b_dict in grouped.items():
        for b_key, coeff in b_dict.items():
            total += Pow(opt.sun_n, nc_power) * Mul(*b_key) * coeff
    
    return total