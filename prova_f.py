#%%
from IPython.display import display
import amplitude
from sympy import symbols,init_printing,expand, IndexedBase
from SUNDelta import SUNDelta
from format_utils import VisualDelta
import permutation_utils

N_GLUONS = 4

opt = amplitude.settings()
opt.n_gluons = 3

opt.base_adj_idx_list
gchain = opt.gellman_chain

other_adj_idx_list = list(opt.base_adj_idx_list)
other_adj_idx_list[0], other_adj_idx_list[1] = other_adj_idx_list[1], other_adj_idx_list[0]

first_trace = opt.trace(opt.base_adj_idx_list,opt.first_tr_internal_idx)
second_trace = opt.trace(other_adj_idx_list, opt.second_tr_internal_idx)

f = first_trace - second_trace

expr1 = opt.fierz(first_trace*gchain)
expr2 = opt.fierz(second_trace*gchain)

contract_one = opt.contract_deltas_first(expand(expr1))
contract_two = opt.contract_deltas_second(expand(expr2))
f = contract_one - contract_two

F = 'F'

""" for perm in permutation_utils.non_cyclic_perms(range(1, N_GLUONS + 1)):
    for val in perm:

 """
init_printing(use_latex = 'mathjax')
display(first_trace)
display(second_trace)
display_expr = f.subs(SUNDelta, VisualDelta)
display(display_expr)
