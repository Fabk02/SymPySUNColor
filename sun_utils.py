from sympy import symbols

def abstract_gellmann_chain(SUNT, adj_idx_lst, up_idx_name, down_idx_name):
    """ 
    Symbolic product of gellmann matrices in fundamental rep

    SUNNT is expected to be a IndexBase

    Will define symbols structured like:
    up_idx_name = 'i' 
    down_idx_name = 'j'
    i1,i2,i3...in and j1,j2,j3...jn 
    where n is len(adj_idx_lst) 
    """
    n = len(adj_idx_lst)
    up_idx_lst = symbols(f'{up_idx_name}1:{n+1}')
    down_idx_lst = symbols(f'{down_idx_name}1:{n+1}')
    expr = 1

    for adj_idx, up_idx, down_idx in zip(adj_idx_lst, up_idx_lst, down_idx_lst):
        expr *= SUNT[adj_idx,up_idx,down_idx]

    return expr

def abstract_trace(SUNT, adj_idx_lst, internal_idx_name):

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

    return expr