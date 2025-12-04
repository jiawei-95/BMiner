
import copy
import socket
import random
import os
import sys
import time
import gc
import torch
import numpy as np
from collections import defaultdict

import arguments
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import stop_criterion_all, stop_criterion_batch_any
from auto_LiRPA.operators.convolution import BoundConv
from jit_precompile import precompile_jit_kernels
from beta_CROWN_solver1 import LiRPANet1
from lp_mip_solver import mip, check_enable_refinement
from attack import attack
from attack.attack_pgd import check_and_save_cex
from utils import Logger, print_model
from specifications import (trim_batch, batch_vnnlib, sort_targets,
                            prune_by_idx, add_rhs_offset)
from loading import load_model_and_vnnlib, parse_run_mode, adhoc_tuning, Customized  # pylint: disable=unused-import
from bab1 import general_bab1, get_monotonicity_stats, get_RQ
from input_split.batch_branch_and_bound import input_bab_parallel
from read_vnnlib import read_vnnlib
from cuts.cut_utils import terminate_mip_processes, terminate_mip_processes_by_c_matching
from lp_test import compare_optimized_bounds_against_lp_bounds
from abcrown import ABCROWN
import my_stat


class ABCROWN1(ABCROWN):
    def __init__(self, args=None, **kwargs):
        super(ABCROWN1, self).__init__(args, **kwargs)
    

    def bab(self, data_lb, data_ub, c, rhs,
            data=None, targets=None, vnnlib=None, timeout=None,
            time_stamp=0, data_dict=None, lower_bounds=None, upper_bounds=None,
            reference_alphas=None, attack_images=None, cplex_processes=None,
            activation_opt_params=None, reference_lA=None,
            model_incomplete=None, refined_betas=None,
            create_model=True, model=None, return_domains=False,
            max_iterations=None, property_idx=None, vnnlib_meta=None,
            orig_lirpa_model=None):
        # This will use the refined bounds if the complete verifier is 'bab-refine'.
        # FIXME do not repeatedly create LiRPANet which creates a new BoundedModule each time.
        start_time = time.time()
        # Save these arguments in case that they need to retrieved the next time
        # this function is called.
        if vnnlib_meta is None:
            vnnlib_meta = {
                'property_idx': 0, 'vnnlib_id': 0, 'benchmark_name': None
            }
        self.data_lb, self.data_ub, self.c, self.rhs = data_lb, data_ub, c, rhs
        self.data, self.targets, self.vnnlib = data, targets, vnnlib

        # if using input split, transpose C if there are multiple specs with shared input,
        # to improve efficiency when calling the incomplete verifier later
        if arguments.Config['bab']['branching']['input_split']['enable']:
            c_transposed = False
            if (data_lb.shape[0] == 1 and data_ub.shape[0] == 1 and c is not None
                    and c.shape[0] > 1 and c.shape[1] == 1):
                # multiple c instances (multiple vnnlibs) since c.shape[0] > 1,
                # but they share the same input (since data.shape[0] == 1）and
                # only single spec in each instance (c.shape[1] == 1)
                c = c.transpose(0, 1)
                rhs = rhs.transpose(0, 1)
                c_transposed = True

        if create_model:
            self.model = LiRPANet1(
                model, c=c, cplex_processes=cplex_processes,
                in_size=(data_lb.shape if len(targets) <= 1
                        else [len(targets)] + list(data_lb.shape[1:])),
                mip_building_proc=(orig_lirpa_model.mip_building_proc
                                   if orig_lirpa_model is not None else None)
            )
            if not model_incomplete:
                print_model(self.model.net)

        data_lb, data_ub = data_lb.to(self.model.device), data_ub.to(self.model.device)
        norm = arguments.Config['specification']['norm']
        if data_dict is not None:
            assert isinstance(data_dict['eps'], float)
            ptb = PerturbationLpNorm(
                norm=norm, eps=data_dict['eps'],
                eps_min=data_dict.get('eps_min', 0), x_L=data_lb, x_U=data_ub)
        else:
            ptb = PerturbationLpNorm(norm=norm, x_L=data_lb, x_U=data_ub)

        if data is not None:
            data = data.to(self.model.device)
            x = BoundedTensor(data, ptb).to(data_lb.device)
            output = self.model.net(x).flatten()
            print('Model prediction is:', output)

            # save output:
            if arguments.Config['general']['save_output']:
                arguments.Globals['out']['pred'] = output.cpu()

            if arguments.Config['attack']['check_clean'] and not arguments.Config['debug'][
                'sanity_check']:
                clean_rhs = c.matmul(output)
                print(f'Clean RHS: {clean_rhs}')
                if (clean_rhs < rhs).any():
                    # add and set output batch_size dimension to 1
                    verified_status, _ = check_and_save_cex(
                        x.detach(), output.unsqueeze(0), vnnlib,
                        arguments.Config['attack']['cex_path'], 'unsafe')
                    return -torch.inf, None, verified_status
        else:
            x = BoundedTensor(data_lb, ptb).to(data_lb.device)

        self.domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)
        if arguments.Config['bab']['branching']['input_split']['enable']:
            result = input_bab_parallel(
                self.model, self.domain, x, rhs=rhs,
                timeout=timeout, max_iterations=max_iterations,
                vnnlib=vnnlib, c_transposed=c_transposed,
                return_domains=return_domains, vnnlib_meta=vnnlib_meta
            )
            if return_domains:
                return result
        else:
            assert not return_domains, 'return_domains is only for input split for now'
            my_stat.other_time += time.time() - start_time
            result = general_bab1(
                self.model, self.domain, x,
                refined_lower_bounds=lower_bounds, refined_upper_bounds=upper_bounds,
                activation_opt_params=activation_opt_params, reference_lA=reference_lA,
                reference_alphas=reference_alphas, attack_images=attack_images,
                timeout=timeout, max_iterations=max_iterations,
                refined_betas=refined_betas, rhs=rhs, property_idx=property_idx,
                model_incomplete=model_incomplete, time_stamp=time_stamp)

        min_lb = result[0]
        if min_lb is None:
            min_lb = -torch.inf
        elif isinstance(min_lb, torch.Tensor):
            min_lb = min_lb.item()

        result = (min_lb, *result[1:3])
        return result


    def complete_verifier(
            self, model_ori, model_incomplete, vnnlib, batched_vnnlib, vnnlib_shape,
            index, timeout_threshold, bab_ret=None, cplex_processes=None,
            attack_images=None, attack_margins=None, results=None, vnnlib_id=None,
            benchmark_name=None, orig_lirpa_model=None
    ):
        start_time = time.time()

        enable_incomplete = arguments.Config['general']['enable_incomplete_verification']
        init_global_lb = results.get('global_lb', None)
        lower_bounds = results.get('lower_bounds', None)
        upper_bounds = results.get('upper_bounds', None)
        reference_alphas = results.get('alpha', None)
        lA = results.get('lA', None)
        cplex_cuts = (arguments.Config['bab']['cut']['enabled']
                    and arguments.Config['bab']['cut']['cplex_cuts'])
        bab_attack_enabled = arguments.Config['bab']['attack']['enabled']

        reference_alphas_cp = None
        if enable_incomplete:
            final_name = model_incomplete.final_name
            init_global_ub = upper_bounds[final_name]
            print('lA shape:', [lAitem.shape for lAitem in lA.values()])
            (batched_vnnlib, init_global_lb, init_global_ub,
            lA, attack_images) = sort_targets(
                batched_vnnlib, init_global_lb, init_global_ub,
                attack_images, attack_margins, results, model_incomplete)
            if reference_alphas is not None:
                reference_alphas_cp = copy.deepcopy(reference_alphas)

        solved_c_rows = []

        time_stamp = 0
        my_stat.flag = 'safe'
        rhs_offsets = arguments.Config['specification']['rhs_offset']
        for property_idx, properties in enumerate(batched_vnnlib):  # loop of x
            # batched_vnnlib: [x, [(c, rhs, y, pidx)]]
            
            print(f'\nProperties batch {property_idx}, size {len(properties[0])}')
            timeout = timeout_threshold - (time.time() - start_time)
            print(f'Remaining timeout: {timeout}')
            start_time_bab = time.time()
            print(f'Verifying property {property_idx} with {len(properties[0])} instances.')
            if arguments.Config['debug']['sanity_check'] == 'Full':
                rhs_offset = 0 if rhs_offsets is None else rhs_offsets[property_idx]
                timeout = timeout_threshold
                sanity_check_results = []
            else:
                rhs_offset = 0 if rhs_offsets is None else rhs_offsets
            if (arguments.Config['bab']['cut']['enabled'] and
                arguments.Config['bab']['initial_max_domains'] == 1
                and not arguments.Config['debug']['sanity_check']):
                if init_global_lb[property_idx][0] > rhs_offset:
                    print('Verified by alpha-CROWN bound!')
                    continue

            if isinstance(properties[0][0], dict):
                def _get_item(properties, key):
                    return torch.concat([
                        item[key].unsqueeze(0) for item in properties[0]], dim=0)
                x = _get_item(properties, 'X')
                data_min = _get_item(properties, 'data_min')
                data_max = _get_item(properties, 'data_max')
                # A dict to store extra variables related to the data and specifications
                for item in properties[0]:
                    assert item['eps'] == properties[0][0]['eps']
                data_dict = {
                    'eps': properties[0][0]['eps'],
                    'eps_min': properties[0][0].get('eps_min', 0),
                }
            else:
                x_range = torch.tensor(properties[0], dtype=torch.get_default_dtype())
                data_min = x_range.select(-1, 0).reshape(vnnlib_shape)
                data_max = x_range.select(-1, 1).reshape(vnnlib_shape)
                x = x_range.mean(-1).reshape(vnnlib_shape)  # only the shape of x is important.
                data_dict = None
            if 'tightened_input_bounds' in results:
                assert (
                    results['tightened_input_bounds'][0][property_idx:property_idx+1].shape
                    == data_min.shape
                )
                data_min = results['tightened_input_bounds'][0][property_idx:property_idx+1]
                data_max = results['tightened_input_bounds'][1][property_idx:property_idx+1]

            target_label_arrays = list(properties[1])  # properties[1]: (c, rhs, y, pidx)
            assert len(target_label_arrays) == 1
            c, rhs, pidx = target_label_arrays[0]

            if bab_attack_enabled:
                if arguments.Config['bab']['initial_max_domains'] != 1:
                    raise ValueError(
                        'To run Bab-attack, please set initial_max_domains to 1. '
                        f'Currently it is {arguments.Config["bab"]["initial_max_domains"]}.')
                # Attack images has shape (batch, restarts, specs, c, h, w).
                # The specs dimension should already be sorted.
                # Reshape it to (restarts, c, h, w) for this specification.
                this_spec_attack_images = attack_images[:, :, property_idx].view(
                    attack_images.size(1), *attack_images.shape[3:])
            else:
                this_spec_attack_images = None

            # FIXME Clean up.
            # Shape and type of rhs is very confusing
            rhs = torch.tensor(rhs, device=arguments.Config['general']['device'],
                               dtype=torch.get_default_dtype())
            if enable_incomplete and len(init_global_lb) > 1:
                # no need to trim_batch if batch = 1
                ret_trim = trim_batch(
                    model_incomplete, init_global_lb, init_global_ub,
                    reference_alphas_cp, lower_bounds, upper_bounds,
                    reference_alphas, lA, property_idx, c, rhs)
                lA_trim, rhs = ret_trim['lA'], ret_trim['rhs']
                trimmed_lower_bounds = ret_trim['lower_bounds']
                trimmed_upper_bounds = ret_trim['upper_bounds']
            else:
                lA_trim = lA.copy() if lA is not None else lA
                trimmed_lower_bounds = lower_bounds
                trimmed_upper_bounds = upper_bounds

            print(f'##### Instance {index} first 10 spec matrices: ')
            print(f'{c[:10]}\nthresholds: {rhs.flatten()[:10]} ######')

            torch.cuda.empty_cache()
            gc.collect()
            c = c.to(rhs)  # both device and dtype

            # compress the first dim of data_min, data_max based on duplication check
            if data_min.shape[0] > 1:
                l1_err_data_min = torch.norm((data_min[1:] - data_min[0:1]).view(-1), p=1)
                l1_err_data_max = torch.norm((data_max[1:] - data_max[0:1]).view(-1), p=1)
                if l1_err_data_min + l1_err_data_max < 1e-8:
                    # almost same x so we can use the first x
                    x, data_min, data_max = x[0:1], data_min[0:1], data_max[0:1]

            # Complete verification (BaB, BaB with refine, or MIP).
            time_stamp += 1
            input_split = arguments.Config['bab']['branching']['input_split']['enable']
            init_failure_idx = np.array([])
            if enable_incomplete and not input_split:
                if len(init_global_lb) > 1:  # if batch == 1, there is no need to filter here.
                    # Reuse results from incomplete results, or from refined MIPs.
                    # skip the prop that already verified
                    rlb = trimmed_lower_bounds[final_name]
                    # The following flatten is dangerous, each clause in OR only
                    # has one output bound.
                    assert len(rlb.shape) == len(rhs.shape) == 2
                    assert rlb.shape[1] == rhs.shape[1] == 1
                    init_verified_cond = rlb.flatten() > rhs.flatten()
                    init_verified_idx = torch.where(init_verified_cond)[0]
                    if len(init_verified_idx) > 0:
                        print('Initial alpha-CROWN verified for spec index '
                                f'{init_verified_idx} with bound '
                                f'{rlb[init_verified_idx].squeeze()}.')
                        l = init_global_lb[init_verified_idx].tolist()
                        bab_ret.append([index, l, 0, time.time() - start_time_bab, pidx])
                    init_failure_idx = torch.where(~init_verified_cond)[0]
                    if len(init_failure_idx) == 0:
                        # This batch of x verified by init opt crown
                        RQ1_list, RQ2_list = get_RQ()
                        RQ1_list.append(['safe-incomplete', my_stat.incomplete_time, my_stat.incomplete_time, my_stat.incomplete_time, 1, 1, 1, len(RQ1_list)])
                        RQ1_list
                        continue
                    print(f'Remaining spec index {init_failure_idx} with '
                            f'bounds {rlb[init_failure_idx]} need to verify.')

                    (reference_alphas, lA_trim, x, data_min, data_max,
                    trimmed_lower_bounds, trimmed_upper_bounds, c) = prune_by_idx(
                        reference_alphas, init_verified_cond, final_name, lA_trim, x,
                        data_min, data_max, lA is not None,
                        trimmed_lower_bounds, trimmed_upper_bounds, c)
                my_stat.other_time += time.time() - start_time_bab
                l, nodes, ret = self.bab(
                    data=x, targets=init_failure_idx, time_stamp=time_stamp,
                    data_ub=data_max, data_lb=data_min, data_dict=data_dict,
                    lower_bounds=trimmed_lower_bounds, upper_bounds=trimmed_upper_bounds,
                    c=c, reference_alphas=reference_alphas, cplex_processes=cplex_processes,
                    activation_opt_params=results.get('activation_opt_params', None),
                    refined_betas=results.get('refined_betas', None), rhs=rhs[0:1],
                    reference_lA=lA_trim, attack_images=this_spec_attack_images,
                    model_incomplete=model_incomplete, timeout=timeout, vnnlib=vnnlib,
                    model=model_ori, property_idx=property_idx,
                    vnnlib_meta={
                        'property_idx': property_idx,
                        'vnnlib_id': vnnlib_id,
                        'benchmark_name': benchmark_name
                    },
                    orig_lirpa_model=orig_lirpa_model,
                )
                bab_ret.append([index, float(l), nodes,
                                time.time() - start_time_bab,
                                init_failure_idx.tolist()])

            # terminate the corresponding cut inquiry process if exists
            if cplex_cuts:
                solved_c_rows.append(c)
                terminate_mip_processes_by_c_matching(cplex_processes, solved_c_rows)

            timeout = timeout_threshold - (time.time() - start_time)
            if ret == 'unsafe':
                return 'unsafe-bab'
            # elif ret == 'unknown' or timeout < 0:
            #     if arguments.Config['debug']['sanity_check'] == 'Full':
            #         sanity_check_results.append(ret)
            #         continue
            #     return 'unknown'
            elif ret == 'unknown':
                my_stat.flag = 'unknown'
            elif ret != 'safe':
                raise ValueError(f'Unknown return value of bab: {ret}')
        else:
        
            return my_stat.flag
    def main(self, interm_bounds=None):
        print(f'Experiments at {time.ctime()} on {socket.gethostname()}')
        torch.manual_seed(arguments.Config['general']['seed'])
        random.seed(arguments.Config['general']['seed'])
        np.random.seed(arguments.Config['general']['seed'])
        torch.set_printoptions(precision=8)
        device = arguments.Config['general']['device']
        if device != 'cpu':
            torch.cuda.manual_seed_all(arguments.Config['general']['seed'])
            # Always disable TF32 (precision is too low for verification).
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if arguments.Config['general']['deterministic']:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.use_deterministic_algorithms(True)
        if arguments.Config['general']['double_fp']:
            torch.set_default_dtype(torch.float64)
        if arguments.Config['general']['precompile_jit']:
            precompile_jit_kernels()

        bab_args = arguments.Config['bab']
        debug_args = arguments.Config['debug']
        timeout_threshold = bab_args['timeout']
        interm_transfer_init = bab_args['interm_transfer']
        cut_usage_init = bab_args['cut']['enabled']
        select_instance = arguments.Config['data']['select_instance']
        complete_verifier = arguments.Config['general']['complete_verifier']
        p_a_crown_init = arguments.Config['solver']['prune_after_crown']
        if bab_args['backing_up_max_domain'] is None:
            arguments.Config['bab']['backing_up_max_domain'] = bab_args['initial_max_domains']
        (run_mode, save_path, file_root, example_idx_list, model_ori,
        vnnlib_all, shape) = parse_run_mode()
        self.logger = Logger(run_mode, save_path, timeout_threshold)

        for new_idx, csv_item in enumerate(example_idx_list):
            start_time = time.time()
            my_stat.other_time = 0
            my_stat.dfs_time = 0
            my_stat.bminere_time = 0
            my_stat.bminerg_time = 0

            arguments.Globals['example_idx'] = new_idx
            vnnlib_id = new_idx + arguments.Config['data']['start']
            # Select some instances to verify
            if select_instance and not vnnlib_id in select_instance:
                continue
            self.logger.record_start_time()

            print(f'\n {"%"*35} idx: {new_idx}, vnnlib ID: {vnnlib_id} {"%"*35}')
            if arguments.Config['general']['save_output']:
                arguments.Globals['out']['idx'] = new_idx   # saved for test

            onnx_path = None
            if run_mode != 'customized_data':
                if len(csv_item) == 3:
                    # model, vnnlib, timeout
                    print((csv_item[1]), "*********************")
                    model_ori, shape, vnnlib, onnx_path = load_model_and_vnnlib(
                        file_root, csv_item)
                    arguments.Config['model']['onnx_path'] = os.path.join(file_root, csv_item[0])
                    arguments.Config['specification']['vnnlib_path'] = os.path.join(
                        file_root, csv_item[1])
                else:
                    # Each line contains only 1 item, which is the vnnlib spec.
                    vnnlib = read_vnnlib(os.path.join(file_root, csv_item[0]))
                    assert arguments.Config['model']['input_shape'] is not None, (
                        'vnnlib does not have shape information, '
                        'please specify by --input_shape')
                    shape = arguments.Config['model']['input_shape']
            else:
                vnnlib = vnnlib_all[new_idx]  # vnnlib_all is a list of all standard vnnlib

            # Skip running the actual verifier during preparation.
            if arguments.Config['general']['prepare_only']:
                continue

            # Update timeout
            bab_args['timeout'] = float(bab_args['timeout'])
            if bab_args['timeout_scale'] != 1:
                new_timeout = bab_args['timeout'] * bab_args['timeout_scale']
                print(f'Scaling timeout: {bab_args["timeout"]} -> {new_timeout}')
                bab_args['timeout'] = new_timeout
            if bab_args['override_timeout'] is not None:
                new_timeout = bab_args['override_timeout']
                print(f'Overriding timeout: {new_timeout}')
                bab_args['timeout'] = new_timeout
            timeout_threshold = bab_args['timeout']
            self.logger.update_timeout(timeout_threshold)

            model_ori.eval()
            vnnlib_shape = shape

            # Process input data
            if isinstance(vnnlib[0][0], dict):
                x = vnnlib[0][0]['X'].reshape(vnnlib_shape)
                data_min = vnnlib[0][0]['data_min'].reshape(vnnlib_shape)
                data_max = vnnlib[0][0]['data_max'].reshape(vnnlib_shape)
            else:
                x_range = torch.tensor(vnnlib[0][0])
                data_min = x_range.select(-1, 0).reshape(vnnlib_shape)
                data_max = x_range.select(-1, 1).reshape(vnnlib_shape)
                x = x_range.mean(-1).reshape(vnnlib_shape)  # only the shape of x is important.
            adhoc_tuning(data_min, data_max, model_ori)

            # Apply RHS offset if specified
            rhs_offset_init = arguments.Config['specification']['rhs_offset']
            if rhs_offset_init is not None and not arguments.Config['debug']['sanity_check']:
                vnnlib = add_rhs_offset(vnnlib, rhs_offset_init)

            model_ori = model_ori.to(device)
            x, data_max, data_min = x.to(device), data_max.to(device), data_min.to(device)
            verified_status, verified_success = 'unknown', False

            model_incomplete = None
            ret = {}
            orig_lirpa_model = None
            incomplete_time = 0
            # Run incomplete verifier if enabled
            if arguments.Config['general']['enable_incomplete_verification']:
                temp_time = time.time()
                incomplete_verification_output = self.incomplete_verifier(
                    model_ori,
                    x,
                    data_ub=data_max,
                    data_lb=data_min,
                    vnnlib=vnnlib,
                    interm_bounds=interm_bounds
                )
                incomplete_time = time.time() - temp_time
                my_stat.incomplete_time = incomplete_time
                if arguments.Config['general']['return_optimized_model']:
                    return incomplete_verification_output
                verified_status, ret, orig_lirpa_model = incomplete_verification_output

                verified_success = verified_status != 'unknown'
                model_incomplete = ret.get('model', None)

                # Auto-determine the complete verifier to use
                if arguments.Config['general']['complete_verifier'] == 'auto':
                    complete_verifier = check_enable_refinement(ret)
                    if complete_verifier in ['bab-refine', 'mip']:
                        arguments.Config['bab']['interm_transfer'] = True
                        arguments.Config['bab']['cut']['enabled'] = False
                        arguments.Config['solver']['prune_after_crown'] = False
                    else:
                        arguments.Config['bab']['interm_transfer'] = interm_transfer_init
                        arguments.Config['bab']['cut']['enabled'] = cut_usage_init
                        arguments.Config['solver']['prune_after_crown'] = p_a_crown_init
            if verified_success:
                RQ1_list, RQ2_list = get_RQ()
                for ii in range(9):
                    RQ1_list.append(['safe-incomplete', incomplete_time, incomplete_time, incomplete_time, 1, 1, 1, ii])

            # Branch and Bound verification
            if not verified_success and complete_verifier not in ['skip', 'unknown-mip']:
                # Initialize CPLEX processes for cuts if needed
                cplex_processes = None
                mip_building_proc = None
                if bab_args['cut']['enabled'] and bab_args['cut']['cplex_cuts']:
                    assert arguments.Config['bab']['initial_max_domains'] == 1
                    if model_incomplete is not None:
                        cplex_processes = model_incomplete.processes
                        print('Cut inquiry processes are launched.')
                        mip_building_proc = model_incomplete.mip_building_proc
                        
                # Prepare batched VNN specifications
                batched_vnnlib = batch_vnnlib(vnnlib)
                benchmark_name = (file_root.split('/')[-1]
                                if debug_args['sanity_check'] is not None else None)
                my_stat.other_time = time.time() - start_time

                # Run the complete verifier (Branch and Bound)
                my_stat.time_out = timeout_threshold
                verified_status = self.complete_verifier(
                    model_ori, model_incomplete, vnnlib, batched_vnnlib, vnnlib_shape,
                    new_idx, bab_ret=self.logger.bab_ret, cplex_processes=cplex_processes,
                    timeout_threshold=timeout_threshold - (time.time() - self.logger.start_time),
                    results=ret, vnnlib_id=vnnlib_id,
                    benchmark_name=benchmark_name, orig_lirpa_model=orig_lirpa_model
                )

                # Cleanup CPLEX processes
                if (bab_args['cut']['enabled'] and bab_args['cut']['cplex_cuts']
                        and model_incomplete is not None):
                    terminate_mip_processes(mip_building_proc, cplex_processes)
                    del cplex_processes
            
                
                
            # Log results
            self.logger.summarize_results(verified_status, new_idx)
            RQ1_list, RQ2_list = get_RQ()
            with open('./result/RQ3/result.csv', 'a') as f:
                for rq1 in RQ1_list:
                    print(','.join(str(x) for x in rq1), file=f)
            with open(f'./result/result_tree.csv', 'a') as f:
                result_list = copy.deepcopy(csv_item)
                result_list.pop()
                for rq1 in RQ1_list:
                    tmp_list = result_list + rq1
                    print(','.join(str(x) for x in tmp_list), file=f)
            with open(f"./result/RQ2/RQ2_instance_{time.time()}_{vnnlib_id}.csv", 'a') as f2:
                for rq2 in RQ2_list:
                    print(','.join(str(x) for x in rq2), file=f2)
            RQ1_list.clear()
            RQ2_list.clear()
            with open('./result/result.csv', 'a') as f1:
                result_list = copy.deepcopy(csv_item)
                result_list.pop()
                result_list.extend([verified_status, my_stat.dfs_time + my_stat.other_time, my_stat.bminere_time + my_stat.other_time, my_stat.bminerg_time + my_stat.other_time, my_stat.dfs_rounds, my_stat.bminere_rounds, my_stat.bminerg_rounds])
                print(','.join(str(x) for x in result_list), file = f1)
        
        self.logger.finish()
        return self.logger.verification_summary

if __name__ == '__main__':
    abcrown = ABCROWN1(args=sys.argv[1:])
    abcrown.main()
