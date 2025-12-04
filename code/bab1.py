
import time
import numpy as np
import torch
import copy
from math import *

from branching_domains1 import BatchedDomainList1
from auto_LiRPA.utils import (stop_criterion_batch_any, multi_spec_keep_func_all,
                              AutoBatchSize)
from auto_LiRPA.bound_ops import (BoundInput)
from attack.domains import SortedReLUDomainList
from attack.bab_attack import bab_loop_attack
from heuristics import get_branching_heuristic
from input_split.input_split_on_relu_domains import input_split_on_relu_domains, InputReluSplitter
from lp_mip_solver import batch_verification_all_node_split_LP
from cuts.cut_verification import cut_verification, get_impl_params
from cuts.cut_utils import fetch_cut_from_cplex, clean_net_mps_process, cplex_update_general_beta
from cuts.infered_cuts import BICCOS
from utils import (print_splitting_decisions, print_average_branching_neurons,
                   Stats, get_unstable_neurons, check_auto_enlarge_batch_size)
from prune import prune_alphas
import arguments
from bab import *
from copy import deepcopy
from collections import deque
import my_stat

mono_time = 0
non_mono_time = 0
K = 0
RQ1_list = []
RQ2_list = []
iteration = 0
def get_monotonicity_stats():
    global mono_time, non_mono_time
    return mono_time, non_mono_time

def get_RQ():
    global RQ1_list, RQ2_list
    return RQ1_list, RQ2_list

# Function to solve a node for DFS baseline
def split_domain_dfs(net, domains, d=None, batch=1, impl_params=None, stats=None,
                 set_init_alpha=False, fix_interm_bounds=True,
                 branching_heuristic=None, iter_idx=None, Q=None, tree=None, node=None, run_all=False):
    solver_args = arguments.Config['solver']
    bab_args = arguments.Config['bab']
    branch_args = bab_args['branching']
    biccos_args = bab_args['cut']['biccos']
    biccos_enable = biccos_args['enabled']
    biccos_heuristic = biccos_args['heuristic']
    stop_func = stop_criterion_batch_any
    min_batch_size = min(
        solver_args['min_batch_size_ratio'] * solver_args['batch_size'],
        batch)
    batch = 1
    
    if node.lchild is None:
        solve_node_and_split(net, domains, batch=batch, impl_params=impl_params,
            stats=stats, fix_interm_bounds=fix_interm_bounds,
            branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=node)

    if node.all_node_split:
        print('all nodes are split!!')
        print(f'{stats.visited} domains visited')
        stats.all_node_split = True
        stats.all_split_result = 'unknown'
        if not solver_args['beta-crown']['all_node_split_LP']:
            return 1, node.round_time
        
    
    if node.p <= 0:
        Q.append(node.rchild)
        Q.append(node.lchild)
    return 1, node.round_time


# Solve the node, then split the domain again
def solve_node_and_split(net, domains, d=None, batch=1, impl_params=None, stats=None,
                 set_init_alpha=False, fix_interm_bounds=True,
                 branching_heuristic=None, iter_idx=None, Q=None, tree=None, node=None, run_all=False):
    solver_args = arguments.Config['solver']
    bab_args = arguments.Config['bab']
    branch_args = bab_args['branching']
    biccos_args = bab_args['cut']['biccos']
    stop_func = stop_criterion_batch_any
    min_batch_size = min(
        solver_args['min_batch_size_ratio'] * solver_args['batch_size'],
        batch)

    batch = 1
    timer_flag = node.p is None
    if timer_flag:
        start_time = time.time()
    d = deepcopy(node.d)
    # skip the root node as it has been evaluated
    if node.parent is not None:
        split = node.parent.split
        assert split is not None
        isleft = node.isleft
        d = net.build_history_and_set_bounds1(d, split, impl_params=impl_params, mode='depth', left=isleft)
        batch = len(split['decision'])

        branching_points = split['points']
        ret = net.update_bounds(
            d, fix_interm_bounds=fix_interm_bounds,
            stop_criterion_func=stop_func(d['thresholds']),
            multi_spec_keep_func=multi_spec_keep_func_all,
            beta_bias=branching_points is not None)
        domains.add(ret, d, check_infeasibility=False)
        di = domains.pick_out(batch=batch, device=net.x.device, impl_params=impl_params)
        node.d = di
        global_lb = ret['lower_bounds'][net.final_name].max().item() 
        node.p = global_lb
        assert node.p is not None       
    print('*****lower bound', node.p)
    if node.split is None:
        depth = 1
        di = deepcopy(node.d)
        split_depth = get_split_depth(batch, min_batch_size, depth)
        # Increase the maximum number of candidates for fsb and kfsb if there are more splits needed.
        branching_decision, branching_points, split_depth = (
            branching_heuristic.get_branching_decisions(
                di, split_depth, method=branch_args['method'],
                branching_candidates=max(branch_args['candidates'], split_depth),
                branching_reduceop=branch_args['reduceop']))
        if len(branching_decision) < len(next(iter(di['mask'].values()))):
            node.all_node_split = True
            print('all nodes are split!!')
        split = {
            'decision': branching_decision,
            'points': branching_points,
        }
        if split['points'] is not None and not bab_args['interm_transfer']:
            raise NotImplementedError(
                'General branching points are not supported '
                'when interm_transfer==False')
        node.split = split
        leftnode = BaBNode(d=deepcopy(d), parent=node, isleft=True, depth=node.depth+1)
        node.lchild = leftnode
        rightnode = BaBNode(d=deepcopy(d), parent=node, isleft=False, depth=node.depth+1)
        node.rchild = rightnode
        if timer_flag:
            round_time = time.time() - start_time
            node.round_time = round_time
        return leftnode, rightnode
    # Finally, return its children nodes
    return node.lchild, node.rchild

# reuturn the node by index
def get_node_by_index(net, domains, d=None, batch=1, impl_params=None, stats=None,
                 set_init_alpha=False, fix_interm_bounds=True,
                 branching_heuristic=None, iter_idx=None, Q=None, tree=None, node=None, index=0):
    curnode = node
    if curnode.p is None:
        solve_node_and_split(net, domains, batch=batch, impl_params=impl_params,
                    stats=stats, fix_interm_bounds=fix_interm_bounds,
                    branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=curnode)
    while index > 0:
        if curnode.rchild is not None:
            curnode = curnode.lchild
            if curnode.p is None:
                solve_node_and_split(net, domains, batch=batch, impl_params=impl_params,
                    stats=stats, fix_interm_bounds=fix_interm_bounds,
                    branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=curnode)
        else:
            if curnode.all_node_split:
                return curnode
            else:
                leftnode, right_node = solve_node_and_split(net, domains, batch=batch, impl_params=impl_params,
                    stats=stats, fix_interm_bounds=fix_interm_bounds,
                    branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=curnode)
                curnode = leftnode
        index -= 1
    return curnode

# Get candidate node by index
def get_opposite_node_by_index(startnode, index):
    curnode = startnode
    while index > 1:
        curnode = curnode.lchild
        index -= 1
    if index > 0:
        curnode = curnode.rchild
    return curnode

# Path Solver for bminerE
def split_domain_exponential_search(net, domains, d=None, batch=1, impl_params=None, stats=None,
                 set_init_alpha=False, fix_interm_bounds=True,
                 branching_heuristic=None, iter_idx=None, Q=None, tree=None, node=None):
    total_round = 0
    global K
    bminere_time = 0
    il = 0
    ir = K - node.depth
    ll = [0]
    if ir > 0:
        ll += [2**ii for ii in range(floor(log2(ir))+1)]
    if ll[-1] != ir:
        ll += [ir] 
    curnode = None
    for l in ll:
        curnode = get_node_by_index(net, domains, batch=batch, impl_params=impl_params,
                    stats=stats, fix_interm_bounds=fix_interm_bounds,
                    branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=node, index=l)
        bminere_time += curnode.round_time
        
        total_round += 1
        if curnode.all_node_split and curnode.p < 0:
            stats.all_node_split = True
            stats.all_split_result = 'unknown'
            return total_round, bminere_time
        if curnode.p < 0:
            il = l
        else:
            ir = l
            break
    while ir - il > 1:
        m = ceil((il + ir) / 2)
        curnode = get_node_by_index(net, domains, batch=batch, impl_params=impl_params,
                    stats=stats, fix_interm_bounds=fix_interm_bounds,
                    branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=node, index=m)
        bminere_time += curnode.round_time
        total_round += 1
        if curnode.all_node_split and curnode.p < 0:
            stats.all_node_split = True
            stats.all_split_result = 'unknown'
            return total_round, bminere_time
        if curnode.p < 0:
            il = m
        else:
            ir = m

    for k in range(0, ir):
        curnode = get_opposite_node_by_index(node, k+1)
        Q.append(curnode)

    return total_round, bminere_time

# Function to generate data for RQs
def split_domain_rq(net, domains, d=None, batch=1, impl_params=None, stats=None,
                 set_init_alpha=False, fix_interm_bounds=True,
                 branching_heuristic=None, iter_idx=None, Q=None, tree=None, node=None):
    result = []
    is_mono = 1
    subtree_root = node
    all_node_split_flag = False
    dfs_path_round = 0
    dfs_path_time = 0
    curnode = get_node_by_index(net, domains, batch=batch, impl_params=impl_params,
                    stats=stats, fix_interm_bounds=fix_interm_bounds,
                    branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=node, index=0)
    while curnode.p <= 0 and curnode.all_node_split is False:
        dfs_path_round += 1
        assert curnode.round_time > 0 and curnode.p is not None
        dfs_path_time += curnode.round_time
        curnode = get_node_by_index(net, domains, batch=batch, impl_params=impl_params,
            stats=stats, fix_interm_bounds=fix_interm_bounds,
            branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=curnode, index=1)
        if curnode.p < curnode.parent.p:
            is_mono = 0
    if curnode.p > 0:
        dfs_path_round += 1
        dfs_path_time += curnode.round_time
    dfs_position = dfs_path_round - 1
    result.extend([is_mono, dfs_path_round, dfs_path_time])
    # exponential search    
    global K
    il = 0
    ir = K - node.depth
    ll = [0]
    bminere_path_round = 0
    bminere_path_time = 0
    if ir > 0:
        ll += [2**ii for ii in range(floor(log2(ir))+1)]
    if ll[-1] != ir:
        ll += [ir] 
    curnode = node
    for l in ll:
        curnode = get_node_by_index(net, domains, batch=batch, impl_params=impl_params,
                    stats=stats, fix_interm_bounds=fix_interm_bounds,
                    branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=node, index=l)
        bminere_path_time += curnode.round_time
        bminere_path_round += 1
        if curnode.all_node_split and curnode.p < 0:
            all_node_split_flag = True
            break
        if curnode.p < 0:
            il = l
        else:
            ir = l
            break
    while ir - il > 1 and all_node_split_flag is False:
        m = ceil((il + ir) / 2)
        curnode = get_node_by_index(net, domains, batch=batch, impl_params=impl_params,
                    stats=stats, fix_interm_bounds=fix_interm_bounds,
                    branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=node, index=m)
        bminere_path_round += 1
        bminere_path_time += curnode.round_time
        if curnode.all_node_split and curnode.p < 0:
            all_node_split_flag = True
            break
        if curnode.p < 0:
            il = m
        else:
            ir = m
    bminere_position = ir
    result.extend([bminere_path_round, bminere_path_time])
    # gradient search
    all_node_split_flag = False
    bminerg_path_round = 0
    bminerg_path_time = 0
    p_star = tree.root.p
    inc = 0
    if node.depth == 0:
        bminerg_path_time += node.round_time
        node = node.lchild
        bminerg_path_round += 1
        inc = 1
    gamma = node.depth
    il = node.depth
    t = node.depth
    ir = K
    visited = set()

    curnode = get_node_by_index(net, domains, batch=batch, impl_params=impl_params,
                    stats=stats, fix_interm_bounds=fix_interm_bounds,
                    branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=node, index=0)
    bminerg_path_time += curnode.round_time
    visited.add(curnode.depth)
    bminerg_path_round += 1
    if curnode.all_node_split and curnode.p < 0:
        all_node_split_flag = True

    while curnode.p <= 0 and all_node_split_flag is False:
        il = t
        value = (p_star - curnode.p)
        if value == 0:
            value = 0.00001
        t = min(ceil((t * p_star) / value), K)
        if t in visited:
            break
        curnode = get_node_by_index(net, domains, batch=batch, impl_params=impl_params,
            stats=stats, fix_interm_bounds=fix_interm_bounds,
            branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=node, index=t-gamma)
        bminerg_path_time += curnode.round_time
        visited.add(curnode.depth)
        bminerg_path_round += 1
        if curnode.all_node_split and curnode.p < 0:
            all_node_split_flag = True
            break
    while curnode.p > 0 and all_node_split_flag is False:
        ir = t
        value = (p_star - curnode.p)
        if value == 0:
            value = 0.00001
        t = max(ceil((t * p_star) / value), il)
        if t in visited:
            break
        curnode = get_node_by_index(net, domains, batch=batch, impl_params=impl_params,
            stats=stats, fix_interm_bounds=fix_interm_bounds,
            branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=node, index=t-gamma)
        bminerg_path_time += curnode.round_time
        visited.add(curnode.depth)
        bminerg_path_round += 1
        if curnode.all_node_split and curnode.p < 0:
            all_node_split_flag = True
            break
    if t > il:
        il = t
    while ir - il > 1 and all_node_split_flag is False:
        m = ceil((il + ir) / 2)
        curnode = get_node_by_index(net, domains, batch=batch, impl_params=impl_params,
                    stats=stats, fix_interm_bounds=fix_interm_bounds,
                    branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=node, index=m-gamma)
        bminerg_path_time += curnode.round_time
        bminerg_path_round += 1
        if curnode.all_node_split and curnode.p < 0:
            all_node_split_flag = False
            break
        if curnode.p < 0:
            il = m
        else:
            ir = m
    bminerg_position = ir - node.depth + inc
    
    result.extend([bminerg_path_round, bminerg_path_time])
    # continue the search
    if all_node_split_flag:
        stats.all_node_split = True
        stats.all_split_result = 'unknown'
        return dfs_path_time
    result.extend([dfs_position, bminere_position, bminerg_position])
    global RQ2_list
    RQ2_list.append(result)
    # print(result)
    for k in range(0, dfs_position):
        curnode = get_opposite_node_by_index(subtree_root, k+1)
        Q.append(curnode)

    return dfs_path_time

# Solver for DFS baseline
def DFS_solver(domains, net, batch, iter_idx, stats=None, impl_params=None,
                    branching_heuristic=None, Q=None, tree=None, run_all=False):
    bab_args = arguments.Config['bab']
    sort_domain_iter = bab_args['sort_domain_interval']
    recompute_interm = bab_args['recompute_interm']
    vanilla_crown = bab_args['vanilla_crown']
    spec_args = arguments.Config['specification']

    # d = domains.pick_out(batch=batch, device=net.x.device, impl_params=impl_params)
    node = Q.pop()  
    

    if vanilla_crown:
        node.d['history'] = None

  

    # if node.d['mask'] is not None:
    single_round, dfs_time = split_domain_dfs(net, domains, batch=batch, impl_params=impl_params,
                    stats=stats, fix_interm_bounds=not recompute_interm,
                    branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=node, run_all=run_all)



    if sort_domain_iter > 0 and iter_idx % sort_domain_iter == 0:
        domains.sort()


    return 0, single_round, dfs_time    


# Solver for RQs
def act_rq(domains, net, batch, iter_idx, stats=None, impl_params=None,
                    branching_heuristic=None, Q=None, tree=None):
    bab_args = arguments.Config['bab']
    sort_domain_iter = bab_args['sort_domain_interval']
    recompute_interm = bab_args['recompute_interm']
    vanilla_crown = bab_args['vanilla_crown']
    spec_args = arguments.Config['specification']

    # d = domains.pick_out(batch=batch, device=net.x.device, impl_params=impl_params)
    node = Q.pop()  
    

    if vanilla_crown:
        node.d['history'] = None

  

    # if node.d['mask'] is not None:
    single_time = split_domain_rq(net, domains, batch=batch, impl_params=impl_params,
                    stats=stats, fix_interm_bounds=not recompute_interm,
                    branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=node)



    if sort_domain_iter > 0 and iter_idx % sort_domain_iter == 0:
        domains.sort()
    return single_time

# Solver for bminerE
def act_exponential_search(domains, net, batch, iter_idx, stats=None, impl_params=None,
                    branching_heuristic=None, Q=None, tree=None):
    bab_args = arguments.Config['bab']
    sort_domain_iter = bab_args['sort_domain_interval']
    recompute_interm = bab_args['recompute_interm']
    vanilla_crown = bab_args['vanilla_crown']
    spec_args = arguments.Config['specification']

    # d = domains.pick_out(batch=batch, device=net.x.device, impl_params=impl_params)
    node = Q.pop()  
    

    if vanilla_crown:
        node.d['history'] = None

  

    # if node.d['mask'] is not None:
    total_round, bminere_time = split_domain_exponential_search(net, domains, batch=batch, impl_params=impl_params,
                    stats=stats, fix_interm_bounds=not recompute_interm,
                    branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=node)



    if sort_domain_iter > 0 and iter_idx % sort_domain_iter == 0:
        domains.sort()



    return 0, total_round, bminere_time

# Path solver for bminerG
def split_domain_gradient(net, domains, d=None, batch=1, impl_params=None, stats=None,
                 set_init_alpha=False, fix_interm_bounds=True,
                 branching_heuristic=None, iter_idx=None, Q=None, tree=None, node=None):
    total_round = 0
    global K
    bminerg_time = 0
    p_star = tree.root.p
    gamma = node.depth
    il = node.depth
    t = node.depth
    ir = K
    visited = set()

    curnode = get_node_by_index(net, domains, batch=batch, impl_params=impl_params,
                    stats=stats, fix_interm_bounds=fix_interm_bounds,
                    branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=node, index=0)
    bminerg_time += curnode.round_time
    visited.add(curnode.depth)
    total_round += 1
    if curnode.all_node_split and curnode.p < 0:
        stats.all_node_split = True
        stats.all_split_result = 'unknown'
        return total_round, bminerg_time
    while curnode.p <= 0:
        il = t
        value = (p_star - curnode.p)
        if value == 0:
            value = 0.00001
        t = min(ceil((t * p_star) / value), K)
        if t in visited:
            break
        curnode = get_node_by_index(net, domains, batch=batch, impl_params=impl_params,
            stats=stats, fix_interm_bounds=fix_interm_bounds,
            branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=node, index=t-gamma)
        bminerg_time += curnode.round_time
        visited.add(curnode.depth)
        total_round += 1
        if curnode.all_node_split and curnode.p < 0:
            stats.all_node_split = True
            stats.all_split_result = 'unknown'
            return total_round, bminerg_time
    while curnode.p > 0:
        ir = t
        value = (p_star - curnode.p)
        if value == 0:
            value = 0.00001
        t = max(ceil((t * p_star) / value), il)
        if t in visited:
            break
        curnode = get_node_by_index(net, domains, batch=batch, impl_params=impl_params,
            stats=stats, fix_interm_bounds=fix_interm_bounds,
            branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=node, index=t-gamma)
        bminerg_time += curnode.round_time
        visited.add(curnode.depth)
        total_round += 1
        if curnode.all_node_split and curnode.p < 0:
            stats.all_node_split = True
            stats.all_split_result = 'unknown'
            return total_round, bminerg_time
    if t > il:
        il = t
    while ir - il > 1:
        m = ceil((il + ir) / 2)
        curnode = get_node_by_index(net, domains, batch=batch, impl_params=impl_params,
                    stats=stats, fix_interm_bounds=fix_interm_bounds,
                    branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=node, index=m-gamma)
        bminerg_time += curnode.round_time
        total_round += 1
        if curnode.all_node_split and curnode.p < 0:
            stats.all_node_split = True
            stats.all_split_result = 'unknown'
            return total_round, bminerg_time
        if curnode.p < 0:
            il = m
        else:
            ir = m

    for k in range(0, ir-gamma):
        curnode = get_opposite_node_by_index(node, k+1)
        Q.append(curnode)


    return total_round, bminerg_time

# Solver for bminerG
def act_gradient(domains, net, batch, iter_idx, stats=None, impl_params=None,
                    branching_heuristic=None, Q=None, tree=None):
    bab_args = arguments.Config['bab']
    sort_domain_iter = bab_args['sort_domain_interval']
    recompute_interm = bab_args['recompute_interm']
    vanilla_crown = bab_args['vanilla_crown']
    spec_args = arguments.Config['specification']
    total_round = 0
    total_time = 0
    # d = domains.pick_out(batch=batch, device=net.x.device, impl_params=impl_params)
    node = Q.pop()  
    if node.depth == 0:
        leftnode, right_node = solve_node_and_split(net, domains, batch=batch, impl_params=impl_params,
            stats=stats, fix_interm_bounds=not recompute_interm,
            branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=node)
        total_round += 1
        total_time += node.round_time
        Q.append(right_node)
        node = leftnode
    if vanilla_crown:
        node.d['history'] = None

    single_round, bminerg_time = split_domain_gradient(net, domains, batch=batch, impl_params=impl_params,
                    stats=stats, fix_interm_bounds=not recompute_interm,
                    branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=node)
    total_round += single_round
    total_time += bminerg_time
    if sort_domain_iter > 0 and iter_idx % sort_domain_iter == 0:
        domains.sort()

    return 0, total_round, total_time

# entry point for Branch and Bound
def general_bab1(net, domain, x, refined_lower_bounds=None,
                refined_upper_bounds=None, activation_opt_params=None,
                reference_alphas=None, reference_lA=None, attack_images=None,
                timeout=None, max_iterations=None, refined_betas=None, rhs=0,
                model_incomplete=None, time_stamp=0, property_idx=None):
    # the crown_lower/upper_bounds are present for initializing the unstable
    # indx when constructing bounded module
    # it is ok to not pass them here, but then we need to go through a CROWN
    # process again which is slightly slower
    start_time = time.time()
    stats = Stats()
    solver_args = arguments.Config['solver']
    bab_args = arguments.Config['bab']
    branch_args = bab_args['branching']
    timeout = timeout or bab_args['timeout']
    max_domains = bab_args['max_domains']
    batch = solver_args['batch_size']
    cut_enabled = bab_args['cut']['enabled']
    biccos_args = bab_args['cut']['biccos']
    max_iterations = max_iterations or bab_args['max_iterations']


    if not isinstance(rhs, torch.Tensor):
        rhs = torch.tensor(rhs)
    stop_criterion = stop_criterion_batch_any(rhs)
    start_time = time.time()
    if refined_lower_bounds is None or refined_upper_bounds is None:
        assert arguments.Config['general']['enable_incomplete_verification'] is False
        global_lb, ret = net.build(
            domain, x, stop_criterion_func=stop_criterion, decision_thresh=rhs)
        updated_mask, lA, alpha = (ret['mask'], ret['lA'], ret['alphas'])
        global_ub = global_lb + torch.inf
    else:
        ret = net.build_with_refined_bounds(
            domain, x, refined_lower_bounds, refined_upper_bounds,
            activation_opt_params, reference_lA=reference_lA,
            reference_alphas=reference_alphas, stop_criterion_func=stop_criterion,
            cutter=net.cutter, refined_betas=refined_betas, decision_thresh=rhs)
        (global_ub, global_lb, updated_mask, lA, alpha) = (
            ret['global_ub'], ret['global_lb'], ret['mask'], ret['lA'],
            ret['alphas'])
        # release some storage to save memory
        if activation_opt_params is not None: del activation_opt_params
        torch.cuda.empty_cache()
    round_time = time.time() - start_time
    # Transfer A_saved to the new LiRPANet
    if hasattr(model_incomplete, 'A_saved'):
        net.A_saved = model_incomplete.A_saved

    impl_params = get_impl_params(net, model_incomplete, time_stamp)

    # tell the AutoLiRPA class not to transfer intermediate bounds to save time
    net.interm_transfer = bab_args['interm_transfer']
    if not bab_args['interm_transfer']:
        # Branching domains cannot support
        # bab_args['interm_transfer'] == False and bab_args['sort_domain_interval'] > 0
        # at the same time.
        assert bab_args['sort_domain_interval'] == -1

    all_label_global_lb = torch.min(global_lb - rhs).item()
    all_label_global_ub = torch.max(global_ub - rhs).item()

    if arguments.Config['debug']['lp_test'] in ['LP', 'MIP']:
        return all_label_global_lb, 0, 'unknown'

    if stop_criterion(global_lb).all():
        return all_label_global_lb, 0, 'safe'

    # If we are not optimizing intermediate layer bounds, we do not need to
    # save all the intermediate alpha.
    # We only keep the alpha for the last layer.
    if not solver_args['beta-crown']['enable_opt_interm_bounds']:
        # new_alpha shape:
        # [dict[relu_layer_name, {final_layer: torch.tensor storing alpha}]
        # for each sample in batch]
        alpha = prune_alphas(alpha, net.alpha_start_nodes)


    DomainClass = BatchedDomainList1
    # This is the first (initial) domain.
    domains = DomainClass(
        ret, lA, global_lb, global_ub, alpha,
        copy.deepcopy(ret['history']), rhs, net=net, x=x,
        branching_input_and_activation=branch_args['branching_input_and_activation'])
    domains1 = DomainClass(
        ret, lA, global_lb, global_ub, alpha,
        copy.deepcopy(ret['history']), rhs, net=net, x=x,
        branching_input_and_activation=branch_args['branching_input_and_activation'])
    domains2 = DomainClass(
        ret, lA, global_lb, global_ub, alpha,
        copy.deepcopy(ret['history']), rhs, net=net, x=x,
        branching_input_and_activation=branch_args['branching_input_and_activation'])
    domains3 = DomainClass(
        ret, lA, global_lb, global_ub, alpha,
        copy.deepcopy(ret['history']), rhs, net=net, x=x,
        branching_input_and_activation=branch_args['branching_input_and_activation'])
    num_domains = len(domains)

    # after domains are added, we replace global_lb, global_ub with the multile
    # targets 'real' global lb and ub to make them scalars
    global_lb, global_ub = all_label_global_lb, all_label_global_ub
    updated_mask, tot_ambi_nodes = get_unstable_neurons(updated_mask, net)
    net.tot_ambi_nodes = tot_ambi_nodes
    branching_heuristic = get_branching_heuristic(net)
    
    num_domains = len(domains)
    vram_ratio = 0.85 if cut_enabled else 0.9
    auto_batch_size = AutoBatchSize(
        batch, net.device, vram_ratio,
        enable=arguments.Config['solver']['auto_enlarge_batch_size'])

    total_round = 0
    result = None
    root = BaBNode(d = domains.pick_out(batch=batch, device=net.x.device, impl_params=impl_params))
    root.p = global_lb
    root.round_time = round_time
    root.depth = 0
    tree = BaBTree(root, rhs=rhs, x=x)

    # grab statistics of DFS
    start_time = time.time()
    root = tree.root
    Q = deque()
    Q.append(root)
    total_round = 0
    result = None
    domains = domains
    num_domains = len(Q)
    stats.all_node_split = False
    dfs_time = 0
    while (num_domains > 0 and (max_iterations == -1
                                or total_round < max_iterations)):
        total_round += 1
        global_lb = None
        print(f'BaB round {total_round}')

        auto_batch_size.record_actual_batch_size(min(batch, len(domains)))
    
        global_lb, single_round, single_time = DFS_solver(
            domains, net, batch, iter_idx=total_round,
            impl_params=impl_params, stats=stats,
            branching_heuristic=branching_heuristic, Q = Q, tree=tree)
        dfs_time += single_time
        batch = check_auto_enlarge_batch_size(auto_batch_size)

        if isinstance(global_lb, torch.Tensor):
            global_lb = global_lb.max().item()
    
        num_domains = len(Q)

        if stats.all_node_split:
            if stats.all_split_result == 'unsafe':
                stats.all_node_split = False
                result = 'unsafe_bab'
            else:
                stats.all_node_split = False
                result = 'unknown'
        elif num_domains > max_domains:
            print('Maximum number of visited domains has reached.')
            result = 'unknown'
        elif dfs_time > my_stat.time_out:
            print('Time out!!!!!!!!')
            result = 'unknown'
        if result:
            break
        print(f'Cumulative time: {time.time() - start_time}\n')
    dfs_round = total_round
    my_stat.dfs_time += dfs_time
    my_stat.dfs_rounds += dfs_round
    if not result:
        # No domains left and not timed out.
        result = 'safe'

    # exponential search computation
    start_time = time.time()
    root = tree.root
    Q = deque()
    Q.append(root)
    total_round = 0
    result = None
    domains = domains1
    num_domains = len(Q)
    stats.all_node_split = False
    global K
    K = tot_ambi_nodes
    bminere_time = 0
    while (num_domains > 0 and (max_iterations == -1
                                or total_round < max_iterations)):
        
        global_lb = None
        print(f'BaB bminere round {total_round}')

        auto_batch_size.record_actual_batch_size(min(batch, len(domains)))
    
        global_lb, single_round, single_time = act_exponential_search(
            domains, net, batch, iter_idx=total_round,
            impl_params=impl_params, stats=stats,
            branching_heuristic=branching_heuristic, Q = Q, tree=tree)
        total_round += single_round
        bminere_time += single_time
        batch = check_auto_enlarge_batch_size(auto_batch_size)

        if isinstance(global_lb, torch.Tensor):
            global_lb = global_lb.max().item()
        
        num_domains = len(Q)

        if stats.all_node_split:
            if stats.all_split_result == 'unsafe':
                stats.all_node_split = False
                result = 'unsafe_bab'
            else:
                stats.all_node_split = False
                result = 'unknown'
        elif num_domains > max_domains:
            print('Maximum number of visited domains has reached.')
            result = 'unknown'
        elif bminere_time > my_stat.time_out:
            print('Time out!!!!!!!!')
            result = 'unknown'
        if result:
            break
        print(f'Cumulative time: {time.time() - start_time}\n')
    bminere_round = total_round
    my_stat.bminere_time += bminere_time
    my_stat.bminere_rounds += bminere_round
    if not result:
        # No domains left and not timed out.
        result = 'safe'
    print(f'DFS round: {dfs_round}, Exponential search round: {bminere_round}')

    # gradient computation
    start_time = time.time()
    root = tree.root
    Q = deque()
    Q.append(root)
    total_round = 0
    result = None
    domains = domains2
    num_domains = len(Q)
    stats.all_node_split = False
    bminerg_time = 0

    while (num_domains > 0 and (max_iterations == -1
                                or total_round < max_iterations)):
        
        global_lb = None
        print(f'BaB bminerg round {total_round}')

        auto_batch_size.record_actual_batch_size(min(batch, len(domains)))
    
        global_lb, single_round, single_time = act_gradient(
            domains, net, batch, iter_idx=total_round,
            impl_params=impl_params, stats=stats,
            branching_heuristic=branching_heuristic, Q = Q, tree=tree)
        total_round += single_round
        bminerg_time += single_time
        batch = check_auto_enlarge_batch_size(auto_batch_size)

        if isinstance(global_lb, torch.Tensor):
            global_lb = global_lb.max().item()
        
        num_domains = len(Q)

        if stats.all_node_split:
            if stats.all_split_result == 'unsafe':
                stats.all_node_split = False
                result = 'unsafe_bab'
            else:
                stats.all_node_split = False
                result = 'unknown'
        elif num_domains > max_domains:
            print('Maximum number of visited domains has reached.')
            result = 'unknown'
        elif bminerg_time > my_stat.time_out:
            print('Time out!!!!!!!!')
            result = 'unknown'
        if result:
            break
        print(f'Cumulative time: {time.time() - start_time}\n')
    bminerg_round = total_round
    my_stat.bminerg_time += bminerg_time
    my_stat.bminerg_rounds += bminerg_round
    if not result:
        # No domains left and not timed out.
        result = 'safe'
    

    # RQ
    start_time = time.time()
    rq_time = 0
    root = tree.root
    Q = deque()
    Q.append(root)
    total_round = 0
    result = None
    domains = domains3
    num_domains = len(Q)
    stats.all_node_split = False

    while (num_domains > 0 and (max_iterations == -1
                                or total_round < max_iterations)):
        
        global_lb = None
        print(f'BaB rq round {total_round}')
        total_round += 1
        auto_batch_size.record_actual_batch_size(min(batch, len(domains)))
    
        single_time = act_rq(
            domains, net, batch, iter_idx=total_round,
            impl_params=impl_params, stats=stats,
            branching_heuristic=branching_heuristic, Q = Q, tree=tree)
        rq_time += single_time
        batch = check_auto_enlarge_batch_size(auto_batch_size)

        if isinstance(global_lb, torch.Tensor):
            global_lb = global_lb.max().item()
        
        num_domains = len(Q)

        if stats.all_node_split:
            if stats.all_split_result == 'unsafe':
                stats.all_node_split = False
                result = 'unsafe_bab'
            else:
                stats.all_node_split = False
                result = 'unknown'
        elif num_domains > max_domains:
            print('Maximum number of visited domains has reached.')
            result = 'unknown'
        elif rq_time > my_stat.time_out:
            print('Time out!!!!!!!!')
            result = 'unknown'
        if result:
            break
        print(f'Cumulative time: {time.time() - start_time}\n')
    if not result:
        # No domains left and not timed out.
        result = 'safe'

    global RQ1_list, RQ2_list, iteration
    RQ1_list.append([result, dfs_time, bminere_time, bminerg_time, dfs_round, bminere_round, bminerg_round, len(RQ1_list)])
    
    iteration += 1
    del domains
    clean_net_mps_process(net)

    return global_lb, stats.visited, result, stats


class BaBNode:
    def __init__(self, lchild = None, rchild=None, d = None, p = None, parent=None, isleft=True, split=None, depth=None, all_node_split=False, round_time=0):
        self.lchild = lchild
        self.rchild = rchild
        self.isleft = isleft
        self.d = d
        self.p = p
        self.parent = parent
        self.split = split
        self.depth = depth
        self.all_node_split = all_node_split
        self.round_time = round_time
        
class BaBTree:
    def __init__(self, root=None, rhs=None, x=None):
        self.root = root
        self.rhs = rhs
        self.x = x
