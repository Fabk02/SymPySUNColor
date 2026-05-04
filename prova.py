#%%
from IPython.display import display
import permutation_utils
from color_stripped_amps import *
import sun_utils
from SUNDelta import SUNDelta
from format_utils import VisualDelta, collect_color_structure
import fierz_identity
from sympy import IndexedBase, symbols,expand, latex, pprint, init_printing
from functools import partial

T = IndexedBase('T')
d = SUNDelta
Nc = symbols('Nc', positive = True, integer = True)

fierz = partial(fierz_identity.abstract_fierz, T, d, Nc)
gellman_chain = partial(sun_utils.abstract_gellmann_chain, T)
trace = partial(sun_utils.abstract_trace,T)
contract_deltas = partial(sun_utils.abstract_contract_deltas,Nc)

N_GLUON = 4
adj_idx_list = symbols(f'a1:{N_GLUON+1}')
num_idx_list = symbols(f'1:{N_GLUON+1}')

gchain = gellman_chain(adj_idx_list, 'i', 'j')

expr = 0

for perm in permutation_utils.non_cyclic_perms(range(1,N_GLUON+1)):
    num_idx_list = symbols([str(n) for n in perm])
    adj_idx_list = symbols([f'a{n}' for n in perm])
    cs_amp = abstract_cs_amp('A', 0, num_idx_list)

    tr, internal_tr_idx = trace(adj_idx_list, 'k', True)
    expr_fierz = fierz(tr*gchain*cs_amp)
    expr_contracted = contract_deltas(internal_tr_idx, expand(expr_fierz))

    expr += expr_contracted

expr_collected = collect_color_structure(expr, Nc, SUNDelta)

init_printing(use_latex = 'mathjax')
display_expr = expr_collected.subs(SUNDelta, VisualDelta)

display(display_expr)
