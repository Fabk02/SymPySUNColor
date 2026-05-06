import itertools

def cycle_order(idx_lst):
    min_idx = min(idx_lst)
    min_pos = idx_lst.index(min_idx)
    ordered_idx_lst = idx_lst[min_pos:] + idx_lst[:min_pos]
    return ordered_idx_lst

def non_cyclic_perms(idx_lst):
    all_perms = [p for p in itertools.permutations(idx_lst)]
    cycle_ordered_perms = []
    for perm in all_perms:
        cycle_ordered_perms.append(cycle_order(perm))
    
    return list(set(cycle_ordered_perms))

def cyclic_perms(idx_lst):
    cyclic_perms_lst = []
    cyclic_perms_lst.append(idx_lst)
    
    for pos in range(1, len(idx_lst)):
        new_lst = idx_lst[pos:] + idx_lst[:pos]
        cyclic_perms_lst.append(new_lst)

    return cyclic_perms_lst

def double_trace_inv_perms(idx_lst, split):

    n = len(idx_lst)

    if split == 0 or split == n:
        return non_cyclic_perms(idx_lst)

    distinct_reps = []
    
    all_combinations = list(itertools.combinations(idx_lst,split))
    if split == n-split:
        first_elem = idx_lst[0]
        valid_combinations = [comb for comb in all_combinations if first_elem in comb]
    else:
        valid_combinations = all_combinations

    for t1_tuple in valid_combinations:
        t1 = list(t1_tuple)
        t2 = [x for x in idx_lst if x not in t1]

        t1_cycles = non_cyclic_perms(t1)
        t2_cycles = non_cyclic_perms(t2)

        for c1 in t1_cycles:
            for c2 in t2_cycles:
                distinct_reps.append(c1+c2)
    
    return distinct_reps

def cop(a: list, b: list):
    """
    Generates all unique linear sequences by shuffling every internal 
    rotation of two lists, preserving the relative order of each rotation.
    """
    all_shuffles = set()
    
    # 1. Get all internal rotations of both lists using the provided function
    a_rots = cyclic_perms(a)
    b_rots = cyclic_perms(b)
    
    total_len = len(a) + len(b)
    indices = list(range(total_len))
    
    # 2. Iterate through every combination of internal rotations
    for perm_a in a_rots:
        for perm_b in b_rots:
            
            # 3. Generate all order-preserving linear shuffles using combinations
            for a_indices in itertools.combinations(indices, len(a)):
                res = [None] * total_len
                a_ptr = 0
                b_ptr = 0
                
                # Construct the linear shuffle
                for i in range(total_len):
                    if i in a_indices:
                        res[i] = perm_a[a_ptr]
                        a_ptr += 1
                    else:
                        res[i] = perm_b[b_ptr]
                        b_ptr += 1
                        
                # Add as a tuple so it can be hashed in the set
                all_shuffles.add(tuple(res))
                
    # Return a sorted list of unique permutations
    return sorted(list(all_shuffles))