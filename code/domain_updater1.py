

from collections import defaultdict

import torch

from utils import fast_hist_copy
import arguments
from domain_updater import *
from copy import deepcopy

class DomainUpdater1(DomainUpdaterSimple):
    def branch_node(self, d, split, mode='depth', left=True):
        """
        d: Domains
        split: Split decisions
        mode ('depth' or 'breadth'): For multiple candidate decisions, whether to
        stack them in the depth direction (to apply all the decisions) or
        breadth direction (to try different decisions separately).
        """
        d = deepcopy(d)
        self.num_domain, self.num_split = self.get_num_domain_and_split(
            d, split, self.final_name)
        if split.get('points', None) is not None and split['points'].ndim == 2:
            self.multi_branch = True
            self.num_branches = split['points'].shape[1] + 1
        else:
            self.multi_branch = False
            self.num_branches = 2
        self.num_copy = 1
        # if mode == 'depth':
        #     # TODO some branching points may be invalid and thus the actual
        #     # number of branches may be fewer (to allow some neurons to have
        #     # fewer branching points).
        #     # self.num_copy = 2 where num_branches = 2 num_split = 1
        #     self.num_copy = self.num_branches**self.num_split # TODO support multiple branches
        # else:
        #     assert mode == 'breadth', f"Unsupported splitting mode {mode}"
        #     self.num_copy = self.num_branches * self.num_split

        self.device = d['lower_bounds'][self.final_name].device
        self.node_names = [k for k in d['lower_bounds'].keys() if k != self.final_name]

        d['lower_bounds'] = {
            k: repeat(v, self.num_copy, unsqueeze=True)
            for k, v in d['lower_bounds'].items()}
        d['upper_bounds'] = {
            k: repeat(v, self.num_copy, unsqueeze=True)
            for k, v in d['upper_bounds'].items()}
        self.history = d.get('history', None)
        self.new_history = []
        if self.history is not None:
            for _ in range(self.num_copy):
                for j in range(self.num_domain):
                    self.new_history.append(fast_hist_copy(self.history[j]))
        else:
            self.new_history = [None] * (self.num_copy * self.num_domain)
        self.upd_hist_l, self.upd_hist_u = self.empty_dict(), self.empty_dict()
        self.upd_domain_l, self.upd_domain_u = self.empty_dict(), self.empty_dict()
        self.upd_idx_l, self.upd_idx_u = self.empty_dict(), self.empty_dict()
        self.upd_val_l, self.upd_val_u = self.empty_dict(), self.empty_dict()

        self._set_history(d, split, mode, left=left)

        d['lower_bounds'] = {
            k: v.view(-1, *v.shape[2:]) for k, v in d['lower_bounds'].items()}
        d['upper_bounds'] = {
            k: v.view(-1, *v.shape[2:]) for k, v in d['upper_bounds'].items()}
        d['history'] = self.new_history

        if 'depths' in d:
            if mode == 'depth':
                d['depths'] = [depth + self.num_split for depth in d['depths']]
            else:
                d['depths'] = [depth + 1 for depth in d['depths']]
            d['depths'] = d['depths'] * self.num_copy
        if 'alphas' in d:
            new_alphas = defaultdict(dict)
            for k, v in d['alphas'].items():
                new_alphas[k] = {kk: torch.cat([vv] * self.num_copy, dim=2)
                    for kk, vv in v.items()}
            d['alphas'] = new_alphas
        if 'lAs' in d:
            d['lAs'] = {
                k: repeat(v, self.num_copy)
                for k, v in d['lAs'].items()
            }
        for k in ['split_history', 'cs', 'betas', 'intermediate_betas',
                'thresholds', 'x_Ls', 'x_Us']:
            if k in d:
                d[k] = repeat(d[k], self.num_copy)
        for k in split:
            if isinstance(split[k], list):
                split[k] = split[k][-self.num_domain:] * self.num_copy
            elif isinstance(split[k], torch.Tensor):
                split[k] = split[k][-self.num_domain:].repeat(
                    self.num_copy, *[1]*(split[k].ndim - 1))
        return d


    def _set_history(self, d, split, *args, left=True):


        upd_domain, upd_idx= self.empty_dict(), self.empty_dict()
        upd = [upd_domain, upd_idx]

        branching_points = split.get('points', None) is not None

        if branching_points:
            upd_val = self.empty_dict()
            upd.append(upd_val)

        for i in range(self.num_domain):
            # FIXME Inconsistent node index for new_history (split_indices)
            # and elsewhere.
            node, idx = split['decision'][i]
            node = self.split_nodes[node].name
            points = split['points'][i] if branching_points else None
            # for j in range(2):
            j = 1 if left else 0
            history_idx = (-self.num_copy * self.num_domain
                            + j * self.num_domain + i)
            upd_domain[node].append(i)
            upd_idx[node].append(idx)
            if branching_points:
                upd_val[node].append(points)
            if self.history is not None:
                self._append_history(
                    history_idx, node, idx, 1 - j * 2, points)

        for upd_list in upd:
            for k in upd_list:
                upd_list[k] = self.as_tensor(upd_list[k])
        for k in self.node_names:
            if len(upd_domain[k]):
                if branching_points:
                    if left:
                        d['upper_bounds'][k][0].view(self.num_domain, -1)[
                        upd_domain[k], upd_idx[k]] = upd_val[k]
                    else:
                        d['lower_bounds'][k][0].view(self.num_domain, -1)[
                        upd_domain[k], upd_idx[k]] = upd_val[k]
                else:
                    if left:
                        d['upper_bounds'][k][0].view(self.num_domain, -1)[
                        upd_domain[k], upd_idx[k]] = 0.
                        
                    else:
                        d['lower_bounds'][k][0].view(self.num_domain, -1)[
                        upd_domain[k], upd_idx[k]] = 0.