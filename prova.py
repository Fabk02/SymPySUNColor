import permutation_utils
import sun_utils
from SUNDelta import SUNDelta
import fierz_identity
from sympy import IndexedBase, KroneckerDelta, symbols,expand
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


expr_tr, internal_tr_idx = trace(adj_idx_list, 'k', True)
expr_gchain = gellman_chain(adj_idx_list, 'i', 'j')
expr_fierz = fierz(expr_tr*expr_gchain)

print(expr_tr)
print(expr_gchain)
print(expand(expr_fierz))
print("--")
print(contract_deltas(internal_tr_idx, expand(expr_fierz)))