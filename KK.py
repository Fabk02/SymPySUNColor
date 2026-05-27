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
import KK_utils
import simplification_utils

N_GLUONS = 4
base_num_idx_list = symbols(f'1:{N_GLUONS+1}')

opt_tree_1 = amplitude.settings()
opt_tree_1.n_gluons = N_GLUONS

opt_tree_2 = amplitude.settings()
opt_tree_2.n_gluons = N_GLUONS
opt_tree_2.cs_amp_letter = 'B'

expr_tree_1 = amplitude.generate_leading_amplitude(opt_tree_1, 0)
expr_tree_1 = expr_tree_1.replace(SUNDelta, lambda a,b: SUNDelta(b, a))
expr_tree_1 = expr_tree_1.subs(KK_utils.gen_dressed_kk_rel(opt_tree_1))

pdr_1 = photon_decoupling.gen_tree_lvl_pdr(opt_tree_1.cs_amp_letter, base_num_idx_list)
kk_pdr_1 = KK_utils.gen_kk_pdr(KK_utils.gen_dressed_kk_rel(opt_tree_1), pdr_1)

expr_tree_2 = amplitude.generate_leading_amplitude(opt_tree_2, 0)
#expr_tree_2 = expr_tree_2.subs(KK_utils.gen_dressed_kk_rel(opt_tree_2))
expr_tree_2 = amplitude.apply_reflection(opt_tree_2, expr_tree_2, 0)

product = expr_tree_1 * expr_tree_2

total = sun_utils.compute_closed_cycles(opt_tree_1.sun_n, opt_tree_1.sun_delta, product)
#total = amplitude.collect_by_Nc_then_amp(opt_tree_2, total)

transl_dict = simplification_utils.create_translation_dict(opt_tree_1, 0)
simpl_pdr = simplification_utils.translate_pdr(transl_dict, kk_pdr_1)

init_printing(use_latex='mathjax')

display(KK_utils.gen_dressed_kk_rel(opt_tree_1))
#display(total)

display(expr_tree_1.subs(SUNDelta, VisualDelta))
display(expr_tree_2.subs(SUNDelta, VisualDelta))
display(total)