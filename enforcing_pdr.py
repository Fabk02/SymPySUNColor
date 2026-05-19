#%%
from IPython.display import display
from color_stripped_amps import *
from SUNDelta import SUNDelta
from format_utils import VisualDelta
import photon_decoupling
from sympy import symbols, init_printing
from functools import partial
from time import time
import amplitude

start = int(time()*1000)

LEVEL = "1LOOP"
N_GLUON = 5
base_num_idx_list = symbols(f'1:{N_GLUON+1}')

if LEVEL == "TREE":
    opt = amplitude.settings()
    opt.n_gluons = N_GLUON
    opt.cs_amp_letter = 'B'
    opt.gellman_chain_up_idx = 'f'

    #INIT RELATIONS AND APPLIER
    tree_lvl_pdr_rel = photon_decoupling.gen_tree_lvl_pdr(opt.cs_amp_letter, base_num_idx_list)
    tree_pdr_applier = partial(photon_decoupling.apply_tree_lvl_pdr, tree_lvl_pdr_rel)

    expr_collected = amplitude.generate_amplitude(opt, 0)

    final_expr = photon_decoupling.apply_pdr_relations(opt.sun_n, opt.sun_delta, expr_collected, tree_pdr_applier)

elif LEVEL == "1LOOP":
    
    opt = amplitude.settings()
    opt.n_gluons = N_GLUON
    opt.cs_amp_letter = 'B'
    expr_collected = amplitude.generate_amplitude(opt,1)

    if N_GLUON == 4:
        #INIT RELATIONS AND APPLIER
        g4_1loop_pdr_rel = photon_decoupling.gen_1loop_4g_pdr(opt.cs_amp_letter, base_num_idx_list)
        g4_1loop_pdr_applier = partial(photon_decoupling.apply_4g_1loop_pdr, g4_1loop_pdr_rel)
        
        #APPLY RELATIONS
        final_expr = photon_decoupling.apply_pdr_relations(opt.sun_n, opt.sun_delta, expr_collected, g4_1loop_pdr_applier)
    
    elif N_GLUON == 5:
        #INIT RELATIONS AND APPLIER
        g5_1loop_onlyA3_pdr_rel = photon_decoupling.gen_1loop_5g_onlyA3_pdr(opt.cs_amp_letter, base_num_idx_list)
        g5_1loop_A3A1_pdr_rel = photon_decoupling.gen_1loop_5g_A3A1_pdr(opt.cs_amp_letter,base_num_idx_list)
        g5_1loop_onlyA1_pdr_rel = photon_decoupling.gen_1loop_5g_onlyA1_pdr(opt.cs_amp_letter, base_num_idx_list)
        
        g5_1loop_pdr_applier = partial(photon_decoupling.apply_5g_1loop_pdr, g5_1loop_onlyA3_pdr_rel, g5_1loop_A3A1_pdr_rel, g5_1loop_onlyA1_pdr_rel)
        
        #APPLY RELATIONS
        final_expr = photon_decoupling.apply_pdr_relations(opt.sun_n, opt.sun_delta, expr_collected, g5_1loop_pdr_applier)
    
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