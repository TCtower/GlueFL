import logging
import torch

class Sparsification(object):
    model_update_1_round_overhead = 0
    model_update_accurate_cache = {}

    @staticmethod
    def check_sparsification_ratio(global_g_list):
        worker_number = len(global_g_list)
        spar_ratio = 0.

        total_param = 0
        for g_idx, g_param in enumerate(global_g_list[0]):
            total_param += len(torch.flatten(global_g_list[0][g_idx]))

        for i in range(worker_number):
            non_zero_param = 0
            for g_idx, g_param in enumerate(global_g_list[i]):
                mask = g_param != 0.
                # print(mask)
                non_zero_param += float(torch.sum(mask))

            spar_ratio += (non_zero_param / total_param) / worker_number

        return spar_ratio

    @staticmethod
    def check_model_update_overhead(l, r, global_model, mask_record_list, device, use_accurate_cache=False):
        # logging.info(f"{mask_record_list}")
        if r - l < 0:
            raise RuntimeError(f"check_model_update_overhead() saw r{r} which is less than l{l}")

        if l == 0:
            return 1;

        if r - l == 0:
            return 0
            
        if r - l == 1:
            if Sparsification.model_update_1_round_overhead > 0:
                return Sparsification.model_update_1_round_overhead

        elif use_accurate_cache:
            if (r << 16 + l) in Sparsification.model_update_accurate_cache:
                return Sparsification.model_update_accurate_cache[r << 16 + l]

        mask_accum_list = []
        
        for p_idx, key in enumerate(global_model.state_dict().keys()):
            mask_accum_list.append(torch.zeros_like(global_model.state_dict()[key], dtype=torch.bool, device=torch.device("cpu")))

        for idx in range(l, r):
            for p_idx, key in enumerate(global_model.state_dict().keys()):
                mask_accum_list[p_idx] |= mask_record_list[idx][p_idx]
        
        tot_nonzero = 0
        tot_param = 0
        for p_idx, key in enumerate(global_model.state_dict().keys()):
            tot_nonzero += mask_accum_list[p_idx].sum()
            tot_param += mask_accum_list[p_idx].numel()
            
        res = float(tot_nonzero / tot_param)

        if r - l == 1:
            Sparsification.model_update_1_round_overhead = res
        elif use_accurate_cache:
            Sparsification.model_update_accurate_cache[r << 16 + l] = res

        return res