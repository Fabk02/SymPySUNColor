#%%
from IPython.display import display
import KK_utils
import simplification_utils
import amplitude
from SUNDelta import SUNDelta
import sun_utils
from sympy import init_printing, Add, symbols, expand
import photon_decoupling
import reflection_utils


N_GLUONS = 5

opt_tree_1 = amplitude.settings()
opt_tree_1.n_gluons = N_GLUONS

opt_2 = amplitude.settings()
opt_2.n_gluons = N_GLUONS
opt_2.cs_amp_letter = 'B'

abs_refl_dict = reflection_utils.create_abstract_reflection_dict(N_GLUONS)

refl_dict = reflection_utils.create_specific_reflection_dict(opt_tree_1.cs_amp_letter, 0, abs_refl_dict)
other_refl_dict = reflection_utils.create_specific_reflection_dict(opt_2.cs_amp_letter, 0, abs_refl_dict)

kk_dict = KK_utils.gen_dressed_kk_rel(opt_tree_1)
other_kk_dict = KK_utils.gen_dressed_kk_rel(opt_2)

base_num_idx_list = symbols(f'1:{N_GLUONS+1}')

tree_lvl_pdr_rel = photon_decoupling.gen_tree_lvl_pdr(opt_tree_1.cs_amp_letter, base_num_idx_list)
tree_lvl_pdr_rel = reflection_utils.reflect_pdr(tree_lvl_pdr_rel, refl_dict)

other_tree_lvl_pdr_rel = photon_decoupling.gen_tree_lvl_pdr(opt_2.cs_amp_letter, base_num_idx_list)
other_tree_lvl_pdr_rel = reflection_utils.reflect_pdr(other_tree_lvl_pdr_rel, other_refl_dict)

expr_tree_1 = amplitude.generate_leading_amplitude(opt_tree_1, 0)
expr_tree_1 = expr_tree_1.replace(SUNDelta, lambda a,b: SUNDelta(b, a))
expr_tree_1 = expr_tree_1.subs(refl_dict)

expr_tree_2 = amplitude.generate_leading_amplitude(opt_2, 0)
expr_tree_2 = expr_tree_2.subs(other_refl_dict)

product = expr_tree_1 * expr_tree_2

total = sun_utils.compute_closed_cycles(opt_tree_1.sun_n, opt_tree_1.sun_delta, product)
total = amplitude.collect_by_Nc_then_amp(opt_2, total)

transl_dict = simplification_utils.create_translation_dict(opt_tree_1, 0)
other_transl_dict = simplification_utils.create_translation_dict(opt_2, 0)

new_total = simplification_utils.simplify_expr(transl_dict, simplification_utils.translate_pdr(transl_dict, tree_lvl_pdr_rel), total)
new_total =  amplitude.collect_by_Nc_then_amp(opt_tree_1, expand(new_total))
new_total = simplification_utils.simplify_expr(other_transl_dict, simplification_utils.translate_pdr(other_transl_dict, other_tree_lvl_pdr_rel), new_total)

init_printing(use_latex='mathjax')

#display(KK_utils.gen_dressed_kk_rel(opt_tree_1))


#display(expr_tree_1.subs(SUNDelta, VisualDelta))
#display(expr_tree_2.subs(SUNDelta, VisualDelta))
display(total)
display(expand((new_total.subs(kk_dict)).subs(other_kk_dict)))
#display(simplify_expr(opt_2,opt_2, total))