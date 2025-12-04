
import torch
import copy
from collections import defaultdict
import time
import operator
from types import SimpleNamespace
import arguments
from itertools import islice
from collections import deque

from tensor_storage import get_tensor_storage
from utils import fast_hist_copy, check_infeasible_bounds
from branching_domains import BatchedDomainList


class BatchedDomainList1(BatchedDomainList):
    """An unsorted but batched list of domain list."""

    def __init__(self, ret, lAs, global_lbs, global_ubs,
                 alphas=None, history=None, thresholds=None,
                 net=None, branching_input_and_activation=False, x=None):
        super().__init__(ret, lAs, global_lbs, global_ubs,
                 alphas=alphas, history=history, thresholds=thresholds,
                 net=net, branching_input_and_activation=branching_input_and_activation, x=x)


    def add(self, bounds, d, check_infeasibility):
        histories = d['history']
        decision_threshs = d['thresholds']
        final_lower = bounds['lower_bounds'][self.final_name]
        batch = final_lower.size(0)
        device = final_lower.device
        decision_threshs = decision_threshs.to(device)
        assert (self.all_x_Ls is None) == (bounds['x_Ls'] is None), "Inconsistent x_Ls during construction and using {type(self)}."
        assert (self.all_x_Us is None) == (bounds['x_Us'] is None), "Inconsistent x_Ls during construction and using {type(self)}."
        assert (self.all_input_split_idx is None) == (bounds['input_split_idx'] is None), "Inconsistent input_split_idx during construction and using {type(self)}"
        assert len(self.all_lAs) == len(bounds['lAs'])

        if check_infeasibility:
            infeasible = check_infeasible_bounds(
                bounds['lower_bounds'], bounds['upper_bounds'])
        else:
            infeasible = None

        # torch.all() is when taking max for multiple output specifications
        indexer = torch.ones_like(
            torch.all(bounds['lower_bounds'][self.final_name], dim=1), 
            dtype=torch.bool
            )
        if infeasible is not None:
            indexer = torch.logical_and(indexer, torch.logical_not(infeasible))
        indexer = indexer.nonzero().view(-1)
        if len(indexer) == 0:
            return
        # Add all list items in batch without for loop.
        if len(indexer) > 0:
            # itemgetter returns a value instead of tuple when length is 1, so need a special case.
            batch_indexer_lst = indexer.tolist()
            selector = (
                operator.itemgetter(*batch_indexer_lst)
                if len(batch_indexer_lst) > 1
                else lambda _arr: (_arr[batch_indexer_lst[0]], ))
            self.histories.extend(selector(histories))
            self.all_betas.extend(selector(bounds['betas']))
            self.all_intermediate_betas.extend(selector(bounds['intermediate_betas']))
            self.split_histories.extend(selector(bounds['split_history']))
            self.depths.extend(selector(d['depths']))
        for k in self.all_lAs:
            self.all_lAs[k].append(bounds['lAs'][k][indexer])
        if self.all_x_Ls is not None:
            self.all_x_Ls.append(bounds['x_Ls'][indexer])
        if self.all_x_Us is not None:
            self.all_x_Us.append(bounds['x_Us'][indexer])
        if self.all_input_split_idx is not None:
            self.all_input_split_idx.append(bounds['input_split_idx'][indexer])
        self.all_global_lbs.append(bounds['lower_bounds'][self.final_name][indexer])
        self.all_global_ubs.append(bounds['upper_bounds'][self.final_name][indexer])
        for k, v in bounds['lower_bounds'].items():
            self.all_lb_alls[k].append(v[indexer])
        for k, v in bounds['upper_bounds'].items():
            self.all_ub_alls[k].append(v[indexer])
        alpha_new = alpha_reuse = False
        for k, v in bounds['alphas'].items():
            if k not in self.all_alphas:
                self.all_alphas[k] = {}
            for kk, vv in v.items():
                if kk not in self.all_alphas[k]:
                    # This is the first time to create these alpha TensorStorage
                    self.all_alphas[k][kk] = get_tensor_storage(vv[:,:,indexer].cpu(), concat_dim=2)
                    alpha_new = True
                else:
                    # Reusing existing TensorStorage
                    self.all_alphas[k][kk].append(vv[:,:,indexer])
                    alpha_reuse = True
        assert not (alpha_new and alpha_reuse)
        self.all_thresholds.append(decision_threshs[indexer])
        self.Cs.append(bounds['c'][indexer])
        self.u = len(self.histories)


