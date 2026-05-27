#%%
from IPython.display import display
from sympy import init_printing
from SUNDelta import SUNDelta
from format_utils import VisualDelta
import KK_utils
import reflection_utils
import amplitude
import sun_utils
from time import time

start = int(time())

N_GLUONS = 6

opt_tree = amplitude.settings()
opt_tree.n_gluons = N_GLUONS

opt_loop = amplitude.settings()
opt_loop.n_gluons = N_GLUONS
opt_loop.cs_amp_letter = "B"

tree_kk_dict = KK_utils.gen_dressed_kk_rel(opt_tree)
abs_refl_dict = reflection_utils.create_abstract_reflection_dict(N_GLUONS)
loop_refl_dict = reflection_utils.create_specific_reflection_dict(opt_loop.cs_amp_letter, 1, abs_refl_dict)

expr_tree = amplitude.generate_leading_amplitude(opt_tree, 0)
expr_tree = expr_tree.replace(SUNDelta, lambda a,b: SUNDelta(b, a))
expr_tree = expr_tree.subs(tree_kk_dict)

expr_loop = amplitude.generate_leading_amplitude(opt_loop, 1)
expr_loop = amplitude.expand_subleading(opt_loop, 3, expr_loop)
if N_GLUONS > 5:
    expr_loop = amplitude.expand_subleading(opt_loop, 4, expr_loop)

expr_loop = expr_loop.subs(loop_refl_dict)


product = expr_tree * expr_loop
product = sun_utils.compute_closed_cycles(opt_tree.sun_n, opt_tree.sun_delta, product)
product = amplitude.collect_by_Nc_then_amp(opt_loop, product)

end = int(time())

init_printing(use_latex='mathjax')
display(expr_tree.subs(SUNDelta, VisualDelta))
display(expr_loop.subs(SUNDelta, VisualDelta))
display(product)
print()
print(f"Time: {end - start}")