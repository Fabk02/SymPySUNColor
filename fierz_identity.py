import sympy as sp
from collections import defaultdict
from itertools import combinations

""" 
T = IndexedBase('T')
d = KroneckerDelta
Nc = symbols('Nc', positive = True, integer = True)
"""


def abstract_find_adj_pairs(SUNT, expr):

    T_terms = [f for f in sp.preorder_traversal(expr) if isinstance(f, sp.Indexed) and f.base == SUNT]

    by_gen = defaultdict(list)
    for t in T_terms:
        a_idx, i_idx, j_idx = t.indices
        by_gen[a_idx].append((i_idx,j_idx))

    pairs = []

    for a_idx, ij_list in by_gen.items():
        if len(ij_list) == 2:
            (i,j), (k,l) = ij_list
            pairs.append((a_idx,i,j,k,l))
        
        elif len(ij_list) > 2:
            for (i,j),(k,l) in combinations(ij_list,2):
                pairs.append((a_idx,i,j,k,l))
    
    return pairs

def abstract_fierz(SUNT, SUNDelta, SUNN, expr):
    pairs = abstract_find_adj_pairs(SUNT, expr)

    if not pairs:
        return expr
    
    for (a,i,j,k,l) in pairs:
        rhs = SUNDelta(i,l)*SUNDelta(k,j) - sp.Integer(1)/SUNN*SUNDelta(i,j)*SUNDelta(k,l)
        t1 = SUNT[a, i, j]
        t2 = SUNT[a, k, l]
        # subs(a*b) only matches if SymPy's Mul has that exact factor pair;
        # instead, divide out t1 and t2 explicitly to avoid ordering issues
        expr = expr.subs(t1, 1).subs(t2, rhs)

    return expr

def abstract_leading_fierz(SUNT, SUNDelta, expr):
    pairs = abstract_find_adj_pairs(SUNT, expr)

    if not pairs:
        return expr
    
    for (a,i,j,k,l) in pairs:
        rhs = SUNDelta(i,l)*SUNDelta(k,j)
        t1 = SUNT[a, i, j]
        t2 = SUNT[a, k, l]
        # subs(a*b) only matches if SymPy's Mul has that exact factor pair;
        # instead, divide out t1 and t2 explicitly to avoid ordering issues
        expr = expr.subs(t1, 1).subs(t2, rhs)

    return expr