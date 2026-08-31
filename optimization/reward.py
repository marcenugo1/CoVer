import os
import ast
import json
import random
import argparse
import numpy as np


import optimization_config


# You can customize this reward.py to obtain your reward function. 
# The output of this module is a list, where each element is a dictionary with the keys 'prompt', 'response', and 'reward'.
# The reward here will be directly used as advantage, so you need to normalize them.

# read the configurations and load them as global variables

def str2bool(x):
    return x.lower() in ("1", "true", "yes")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default=optimization_config.pretrained_model)
    parser.add_argument("--dataset", type=str, default=optimization_config.train_dataset)
    parser.add_argument("--max_generation_len", type=int, default=optimization_config.max_generation_token)
    parser.add_argument("--max_len_threshold", type=int, default=optimization_config.max_len_threshold)
    parser.add_argument("--min_len_threshold", type=int, default=optimization_config.min_len_threshold)
    parser.add_argument("--separate_training", type=str2bool, default=optimization_config.separate_training)
    parser.add_argument("--enable_efficient", type=str2bool, default=optimization_config.enable_efficient)
    parser.add_argument("--post_stage", type=str2bool, default=optimization_config.post_stage)
    parser.add_argument("--cover_y_binary", type=str2bool,
                        default=optimization_config.cover_y_binary)
    return parser.parse_args()

args = parse_args()
globals().update(vars(args))



# read the inference data

outputs_name =  pretrained_model.replace("/", ".") + "-" + dataset

os.makedirs(os.path.dirname("./temp_data/outputs-rl-" + outputs_name + ".json"), exist_ok=True)
with open("./temp_data/outputs-rl-" + outputs_name + ".json", 'r') as f:
    data = json.load(f)




# obatin the rollout samples and the corresponding reward/advantages

def plug_in_mi(col, y, n_y_bins):
    """Plug-in mutual-information estimator for binary col vs discrete graded y.

    PDF Eq. 2:  Î(E_:,j ; y) = Σ_{e,v} P̂(e,v) log( P̂(e,v) / (P̂(e) P̂(v)) )
    with the convention 0·log(0) = 0.

    Args:
        col:       (m,) bool / int array — test pass/fail per code (E_:,j).
        y:         (m,) float array      — graded GT pass-rate per code, in [0,1].
        n_y_bins:  int                   — number of distinct values y can take
                                           (G_τ + 1; y is i/G_τ for i = 0..G_τ).
    Returns:
        Î(col ; y) in nats. Returns 0.0 if either marginal is degenerate.
    """
    m = len(col)
    if m == 0:
        return 0.0
    col = np.asarray(col, dtype=int)
    y = np.asarray(y, dtype=float)
    if col.std() <= 1e-12 or y.std() <= 1e-12:
        return 0.0
    # Map graded y ∈ {0, 1/G, 2/G, ..., 1} to integer bin index ∈ {0..n_y_bins-1}
    y_bin = np.round(y * (n_y_bins - 1)).astype(int)
    y_bin = np.clip(y_bin, 0, n_y_bins - 1)
    # Joint count table (2, n_y_bins), normalised to probabilities
    joint = np.zeros((2, n_y_bins), dtype=float)
    np.add.at(joint, (col, y_bin), 1.0)
    joint /= float(m)
    p_e = joint.sum(axis=1, keepdims=True)
    p_v = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = p_e * p_v
        ratio = np.where(denom > 0, joint / denom, 1.0)
        log_ratio = np.where(joint > 0, np.log(ratio), 0.0)
    return float((joint * log_ratio).sum())


def normalize_reward(reward_arr):
    if np.all(reward_arr == 1) and enable_efficient:
        return reward_arr
    mean = np.mean(reward_arr)
    std = np.std(reward_arr)
    if std.item() == 0:
        return None
    return (reward_arr - mean) / std

def normalize_balance_std(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    pos_mask = x > 0
    neg_mask = x < 0
    sum_pos = x[pos_mask].sum()
    sum_neg_abs = abs(x[neg_mask].sum())
    if sum_pos * sum_neg_abs == 0:
        return None
    scale_factor = sum_neg_abs / sum_pos
    x[pos_mask] *= scale_factor
    return x / x.std()

def length_regularize(reward_arr, response_length_list):
    reward_arr = np.sign(reward_arr)
    pos_list = np.where(reward_arr == 1)[0].tolist()
    neg_list = np.where(reward_arr == -1)[0].tolist()
    pos_response_length = np.array([response_length_list[j] for j in pos_list])
    threshold = np.median(pos_response_length).item()
    if np.sum((pos_response_length - threshold)**2) == 0: # no variance
        return normalize_balance_std(np.sign(reward_arr))
    threshold = max(min(threshold, max_len_threshold), min_len_threshold)
    length_reg_reward = np.zeros(len(reward_arr), float)
    length_reg_reward[pos_list] = - pos_response_length + threshold
    length_reg_reward[neg_list] = np.min(length_reg_reward).copy()
    length_reg_reward = normalize_balance_std(length_reg_reward)
    return length_reg_reward

code_data = []
case_data = []
index_list = []

# raw metric accumulators (pre-normalization)
_raw_code_pass_rates = []
_code_all_wrong = 0
_code_all_correct = 0
_code_filtered = 0
_code_total = 0
# COVER reward metrics
_case_mi_pos_rates = []
_case_mean_mi_values = []
_case_sign_gated_rates = []
# shared
_case_eligible = 0
_code_response_lengths = []
_case_response_lengths = []

for i in range(len(data)):
    if data[i]["all_case_bool_table"] is None:
        continue

    t = data[i]["num_ground_truth_test"]
    all_test_table_i = np.array(data[i]["all_case_bool_table"])[:, :t].copy()
    all_case_table_i = np.array(data[i]["all_case_bool_table"])[:, t:].copy()

    # reward for code
    code_reward = np.mean(all_test_table_i, 1)
    #code_reward = all_test_table_i.all(axis=1).astype(float)

    # collect raw stats before normalization
    _code_total += 1
    _raw_code_pass_rates.append(float(np.mean(code_reward)))
    if np.all(code_reward == 0):
        _code_all_wrong += 1
    if all_test_table_i.all(axis=1).all():
        _code_all_correct += 1
    if "code_response_length" in data[i]:
        _code_response_lengths.extend(data[i]["code_response_length"])
    if "case_response_length" in data[i]:
        _case_response_lengths.extend(data[i]["case_response_length"])

    code_reward_raw = code_reward.copy()   # keep raw for optional bonus augmentation below
    code_reward = normalize_reward(code_reward)
    if code_reward is None:
        _code_filtered += 1
    code_start_idx = len(code_data)   # track insertion point for in-place bonus update
    if code_reward is not None:
        if enable_efficient:
            code_reward = length_regularize(code_reward, data[i]["code_response_length"])
        if code_reward is not None:
            code_reward = code_reward.tolist()
            for j in range(len(code_reward)):
                code_data_i = {}
                code_data_i["prompt"] = data[i]["code_generation_prompt"]
                if data[i]["code_response_length"][j] < max_generation_len:
                    code_data_i["response"] = data[i]["full_code_generation"][j] + "<|im_end|>"
                else:
                    code_data_i["response"] = data[i]["full_code_generation"][j]
                code_data_i["reward"] = code_reward[j]
                code_data.append(code_data_i)

    # reward for case — CoVer reward (signed plug-in mutual information).
    # Signed plug-in mutual information against graded GT-anchored y.
    #   r^IG(t_j) = Î(E_:,j ; y)  if  Cov(E_:,j, y) > 0  else  0
    # Operates on whatever case columns survived k_case_keep pruning upstream
    # (in execute.py:prune_generated_cases_by_duplication).
    n_cases = all_case_table_i.shape[1]
    if n_cases > 0:
        _case_eligible += 1
        G_tau = all_test_table_i.shape[1]
        if cover_y_binary:
            # Ablation: collapse graded y to binary (1 iff all GT tests pass)
            y = all_test_table_i.all(axis=1).astype(float)
            n_y_bins = 2
        else:
            y = all_test_table_i.mean(axis=1)          # graded GT, (m,) in [0,1]
            n_y_bins = G_tau + 1
        y_mean = float(y.mean())
        y_centered = y - y_mean
        y_has_variance = float(y.std()) > 1e-9

        case_reward = np.zeros(n_cases, dtype=float)
        n_sign_gated = 0
        n_mi_positive = 0
        mi_values_positive = []

        if y_has_variance:
            for k_case_idx in range(n_cases):
                col = all_case_table_i[:, k_case_idx].astype(float)
                if col.std() <= 1e-9:
                    # constant column (permissive / universally-failing) → MI = 0
                    continue
                cov = float(((col - col.mean()) * y_centered).mean())
                if cov <= 0:
                    # anti-discriminative or independent → reward = 0 (sign gate)
                    n_sign_gated += 1
                    continue
                mi = plug_in_mi(col.astype(int), y, n_y_bins)
                if mi > 0:
                    case_reward[k_case_idx] = mi
                    n_mi_positive += 1
                    mi_values_positive.append(mi)

        # raw cover metrics (pre-normalization)
        _case_mi_pos_rates.append(float(n_mi_positive) / max(n_cases, 1))
        _case_sign_gated_rates.append(float(n_sign_gated) / max(n_cases, 1))
        if mi_values_positive:
            _case_mean_mi_values.append(float(np.mean(mi_values_positive)))

        case_reward = normalize_reward(case_reward)
        if case_reward is not None:
            if enable_efficient:
                case_reward = length_regularize(case_reward, data[i]["case_response_length"])
            if case_reward is not None:
                case_reward = case_reward.tolist()
                for j in range(len(case_reward)):
                    case_data_i = {}
                    case_data_i["prompt"] = data[i]["case_generation_prompt"]
                    if data[i]["case_response_length"][j] < max_generation_len:
                        case_data_i["response"] = data[i]["full_case_generation"][j] + "<|im_end|>"
                    else:
                        case_data_i["response"] = data[i]["full_case_generation"][j]
                    case_data_i["reward"] = case_reward[j]
                    case_data.append(case_data_i)





final_data = code_data + case_data
random.shuffle(final_data)

if optimization_config.use_wandb and (code_data or case_data):
    step = int(os.environ.get("COVER_STEP", 0))
    reward_metrics = {"_step": step}

    # raw code reward metrics (pre-normalization) — these show actual learning progress
    if _code_total > 0:
        reward_metrics["reward/code_raw_pass_rate"] = float(np.mean(_raw_code_pass_rates))
        reward_metrics["reward/code_all_wrong_rate"] = float(_code_all_wrong / _code_total)
        reward_metrics["reward/code_all_correct_rate"] = float(_code_all_correct / _code_total)
        reward_metrics["reward/code_filtered_rate"] = float(_code_filtered / _code_total)
        reward_metrics["reward/num_code_samples"] = len(code_data)

    # raw case reward metrics (pre-normalization)
    if _code_total > 0:
        reward_metrics["reward/case_eligible_rate"] = float(_case_eligible / _code_total)

    # CoVer reward metrics (signed plug-in MI)
    if _case_mi_pos_rates:
        reward_metrics["reward/case_mi_positive_rate"] = float(np.mean(_case_mi_pos_rates))
        reward_metrics["reward/num_case_samples"] = len(case_data)
    if _case_sign_gated_rates:
        reward_metrics["reward/case_sign_gated_rate"] = float(np.mean(_case_sign_gated_rates))
    if _case_mean_mi_values:
        reward_metrics["reward/case_mean_mi_nats"] = float(np.mean(_case_mean_mi_values))

    # response length metrics — track whether length regularization is working
    if _code_response_lengths:
        reward_metrics["reward/code_response_length"] = float(np.mean(_code_response_lengths))
    if _case_response_lengths:
        reward_metrics["reward/case_response_length"] = float(np.mean(_case_response_lengths))

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wandb_reward_metrics.json"), "w") as _f:
        json.dump(reward_metrics, _f)

if separate_training == False:
    with open("./temp_data/rl_data.json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
else:
    with open("./temp_data/rl_code_data.json", "w", encoding="utf-8") as f:
        json.dump(code_data, f, indent=2, ensure_ascii=False)
    with open("./temp_data/rl_case_data.json", "w", encoding="utf-8") as f:
        json.dump(case_data, f, indent=2, ensure_ascii=False)
