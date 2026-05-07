from sympy import symbols, expand
from functools import partial
import sun_utils
import permutation_utils
from color_stripped_amps import *
import fierz_identity
import math
from SUNDelta import SUNDelta
from format_utils import collect_color_structure

class settings:
    sun_tensor = IndexedBase('T')
    sun_delta = SUNDelta
    sun_n = symbols('Nc', positive = True, integer = True)
    sun_adjoint_idx = 'a'
    nf = symbols('nf', positive = True, integer = True)
    cs_amp_letter = 'A'
    gellman_chain_up_idx = 'i'
    gellman_chain_down_idx = 'j'
    first_tr_internal_idx = 'k'
    second_tr_internal_idx = 'q'

def generate_base_amplitude(opt:settings, n_gluons: int, loop_lvl: int, quark_loop: bool = False):
    
    fierz = partial(fierz_identity.abstract_fierz, opt.sun_tensor, opt.sun_delta, opt.sun_n)
    gellman_chain = partial(sun_utils.abstract_gellmann_chain, opt.sun_tensor)
    trace = partial(sun_utils.abstract_trace,opt.sun_tensor)
    contract_deltas = partial(sun_utils.abstract_contract_deltas,opt.sun_n)
    
    adj_idx_list = symbols(f'{opt.sun_adjoint_idx}1:{n_gluons + 1}')
    gchain = gellman_chain(adj_idx_list, opt.gellman_chain_up_idx, opt.gellman_chain_down_idx)

    expr = 0
    for perm in permutation_utils.non_cyclic_perms(range(1,n_gluons + 1)):
        num_idx_list = symbols([str(n) for n in perm])
        adj_idx_list = symbols([f'{opt.sun_adjoint_idx}{n}' for n in perm])
        cs_amp = abstract_cs_amp(opt.cs_amp_letter, loop_lvl, num_idx_list, quark_loop)

        tr, internal_tr_idx = trace(adj_idx_list, opt.first_tr_internal_idx, True)
        expr_fierz = fierz(tr*gchain*cs_amp)
        expr_contracted = contract_deltas(internal_tr_idx, expand(expr_fierz))

        expr += expr_contracted

    expr_collected = collect_color_structure(expr, opt.sun_n, opt.sun_delta)

    return expr_collected

def generate_amplitude(opt:settings, n_gluons: int, loop_lvl: int):
    
    fierz = partial(fierz_identity.abstract_fierz, opt.sun_tensor, opt.sun_delta, opt.sun_n)
    gellman_chain = partial(sun_utils.abstract_gellmann_chain, opt.sun_tensor)
    trace = partial(sun_utils.abstract_trace,opt.sun_tensor)
    contract_deltas = partial(sun_utils.abstract_contract_deltas,opt.sun_n)
    
    adj_idx_list = symbols(f'{opt.sun_adjoint_idx}1:{n_gluons + 1}')
    gchain = gellman_chain(adj_idx_list, opt.gellman_chain_up_idx, opt.gellman_chain_down_idx)


    if loop_lvl == 0:
        return generate_base_amplitude(opt, n_gluons, loop_lvl)
    
    if loop_lvl == 1:
        expr = opt.sun_n * generate_base_amplitude(opt, n_gluons, loop_lvl)
        
        max_lvl = math.floor(n_gluons/2) + 1
        for lvl in range(2,max_lvl + 1):
            for perm in permutation_utils.double_trace_inv_perms(range(1,n_gluons+1),lvl-1):
                
                num_idx_list = symbols([str(n) for n in perm])
                adj_idx_list = symbols([f'{opt.sun_adjoint_idx}{n}' for n in perm])
                cs_amp = abstract_cs_amp(opt.cs_amp_letter, lvl, num_idx_list)

                tr1, internal_tr1_idx = trace(adj_idx_list[:lvl-1], opt.first_tr_internal_idx, True)
                tr2, internal_tr2_idx = trace(adj_idx_list[lvl-1:], opt.second_tr_internal_idx, True)
                expr_fierz = fierz(tr1*tr2*gchain*cs_amp)
                expr_contracted_first = contract_deltas(internal_tr1_idx, expand(expr_fierz))
                expr_contracted_second = contract_deltas(internal_tr2_idx, expand(expr_contracted_first))
                expr += expr_contracted_second
            
        expr_collected = collect_color_structure(expr, opt.sun_n, opt.sun_delta)

        return expr_collected

    else:
        return 0
    
def generate_loop_quark_amplitude(opt:settings, n_gluons: int):
    return (opt.nf * generate_base_amplitude(opt, n_gluons, 1, True))