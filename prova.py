import permutation_utils
import sun_utils
import fierz_identity
from sympy import IndexedBase, KroneckerDelta, symbols,expand
from functools import partial

T = IndexedBase('T')
d = KroneckerDelta
Nc = symbols('Nc', positive = True, integer = True)

fierz = partial(fierz_identity.abstract_fierz, T, d, Nc)
gellman_chain = partial(sun_utils.abstract_gellmann_chain, T)
trace = partial(sun_utils.abstract_trace,T)

N_GLUON = 4
adj_idx_list = symbols(f'a1:{N_GLUON+1}')


expr_tr = trace(adj_idx_list, 'k')
expr_gchain = gellman_chain(adj_idx_list, 'i', 'j')
expr_fierz = fierz(expr_tr*expr_gchain)

print(expr_tr)
print(expr_gchain)
print(expand(expr_fierz))