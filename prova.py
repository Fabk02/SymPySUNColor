#%%
from IPython.display import display
from color_stripped_amps import *
from SUNDelta import SUNDelta
from format_utils import VisualDelta
import photon_decoupling
from sympy import symbols, init_printing, expand
from functools import partial
from time import time
import amplitude
import sun_utils


LEVEL = "1LOOP"
N_PART = 5
N_QUARKS = 2
N_GLUONS = N_PART - N_QUARKS
base_num_idx_list = symbols(f'3:{N_PART+1}')
q_idx_list = list(symbols(r'q \bar{q}'))

opt_1 = amplitude.settings()
opt_2 = amplitude.settings()
opt_2.cs_amp_letter = 'B'

dummy_quark_up_idx = symbols(opt_1.gellman_chain_up_idx + opt_1.q)
dummy_quark_down_idx = symbols(opt_1.gellman_chain_down_idx + opt_1.qbar)

up_idx_lst = list(symbols(f'{opt_1.gellman_chain_up_idx}{N_QUARKS + 1}:{N_PART+1}'))
up_idx_lst.append(dummy_quark_up_idx)
down_idx_lst = list(symbols(f'{opt_1.gellman_chain_down_idx}{N_QUARKS + 1}:{N_PART+1}'))
down_idx_lst.append(dummy_quark_up_idx)

contract_deltas = partial(sun_utils.abstract_contract_deltas,opt_1.sun_n)

expr_1 = amplitude.generate_qq_amplitude(opt_1, N_GLUONS, 0)
conj_expr_1 = expr_1.replace(SUNDelta, lambda a,b: SUNDelta(b, a))

expr_2 = amplitude.generate_qq_amplitude(opt_2, N_GLUONS, 0)

expr = conj_expr_1*expr_2

up_contr = contract_deltas(up_idx_lst, expand(expr))
down_contr = contract_deltas(down_idx_lst, expand(up_contr))

init_printing(use_latex='mathjax')
display_expr = down_contr.subs(SUNDelta, VisualDelta)
display(display_expr)