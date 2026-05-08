from sympy import KroneckerDelta, preorder_traversal, symbols
from SUNDelta import SUNDelta


def abstract_gellmann_chain(SUNT, adj_idx_lst, up_idx_name, down_idx_name, with_idx = False, n_quarks = 0):
    """ 
    Symbolic product of gellmann matrices in fundamental rep

    SUNNT is expected to be a IndexBase

    Will define symbols structured like:
    up_idx_name = 'i' 
    down_idx_name = 'j'
    i1,i2,i3...in and j1,j2,j3...jn 
    where n is len(adj_idx_lst) 
    """
    n = len(adj_idx_lst) + n_quarks
    up_idx_lst = symbols(f'{up_idx_name}{n_quarks+1}:{n+1}')
    down_idx_lst = symbols(f'{down_idx_name}{n_quarks+1}:{n+1}')
    expr = 1

    for adj_idx, up_idx, down_idx in zip(adj_idx_lst, up_idx_lst, down_idx_lst):
        expr *= SUNT[adj_idx,up_idx,down_idx]

    if with_idx:
        return expr, up_idx_lst, down_idx_lst
    else:
        return expr

def abstract_trace(SUNT, adj_idx_lst, internal_idx_name, with_idx = False):

    """ 
    Symbolic trace expansion of gellmann matrices in fundamental rep

    SUNNT is expected to be a IndexBase

    Will define symbols structured like:
    internal_idx_name = 'k' 
    k1,k2,k3...kn
    where n is len(adj_idx_lst) 
    """

    n = len(adj_idx_lst)
    internal_idx_lst = symbols(f'{internal_idx_name}1:{n+1}')

    expr = 1
    for idx in range(n):
        if idx != n-1:
            expr *= SUNT[adj_idx_lst[idx],internal_idx_lst[idx],internal_idx_lst[idx+1]]
        else:
            expr *= SUNT[adj_idx_lst[idx],internal_idx_lst[idx],internal_idx_lst[0]]

    if with_idx:
        return expr, internal_idx_lst
    else:
        return expr


def abstract_gellmann_product(SUNT, adj_idx_lst, internal_idx_name, ext_up_idx_sym, ext_down_idx_sym, with_idx = False):
    n = len(adj_idx_lst)
    internal_idx_lst = symbols(f'{internal_idx_name}1:{n + 1}')

    expr = 1
    for idx in range(n):
        if idx == 0:
            expr *= SUNT[adj_idx_lst[idx],ext_up_idx_sym,internal_idx_lst[idx]]
        
        elif idx == n-1:
            expr *= SUNT[adj_idx_lst[idx], internal_idx_lst[idx-1], ext_down_idx_sym]

        else:
            expr *= SUNT[adj_idx_lst[idx],internal_idx_lst[idx-1],internal_idx_lst[idx]]
    
    if with_idx:
        return expr, internal_idx_lst
    else:
        return expr

def abstract_contract_deltas(SUNN, dummy_indices, expr):
    # 1. Handle addition (sums) by processing each term individually
    expr = expr.expand()
    if expr.is_Add:
        return expr.func(*[abstract_contract_deltas(SUNN, dummy_indices, arg) for arg in expr.args])

    # 2. Process multiplication (single terms)
    dummy_set = set(dummy_indices)
    changed = True

    while changed:
        changed = False
        for d in preorder_traversal(expr):
            if isinstance(d, SUNDelta):
                a, b = d.args
                if a == b:
                    # Safely replace SUNDelta(a, a) with SUNN across this term
                    expr = expr.subs(d, SUNN)
                    changed = True
                    break
                if b in dummy_set:
                    # Replace the specific delta with 1, then contract the index
                    expr = expr.subs(d, 1).subs(b, a)
                    dummy_set.discard(b)
                    changed = True
                    break
                if a in dummy_set:
                    # Replace the specific delta with 1, then contract the index
                    expr = expr.subs(d, 1).subs(a, b)
                    dummy_set.discard(a)
                    changed = True
                    break

    return expr