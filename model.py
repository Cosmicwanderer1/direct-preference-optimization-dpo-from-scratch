"""
Direct Preference Optimization (DPO) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - log_softmax
def log_softmax(logits, axis=-1):
    # TODO: convert logits into numerically stable log-probabilities along axis
    """
    数值稳定的 log-softmax。
    - logits: np.ndarray, 任意 shape
    - axis:   int, 沿哪个维度做 softmax（默认最后一维）
    - 返回:   与 logits 同 shape 的数组，沿 axis 的 exp 求和为 1
    """

    z = np.asarray(logits, dtype=np.float64)

    # 1. 沿 axis 取最大值，keepdims=True 保证可以广播相减
    m = np.max(z, axis=axis, keepdims=True)

    # 2. 减去最大值（数值稳定）
    z_shifted = z - m

    # 3. 指数化后求和，再取 log（同样 keepdims=True）
    log_sum_exp = np.log(np.sum(np.exp(z_shifted), axis=axis, keepdims=True))

    # 4. 相减得到 log-prob
    return z_shifted - log_sum_exp

# Step 2 - softmax
def softmax(logits, axis=-1):
    # TODO: Convert an array of logits into a probability distribution along a given axis
    
    z = np.asarray(logits, dtype=np.float64)

    m = np.max(z, axis=axis, keepdims=True)

    z_shifted = z - m
    
    exp_shifted = np.exp(z_shifted)
    
    sum_exp = np.sum(exp_shifted, axis=axis, keepdims=True)


    return exp_shifted / sum_exp

# Step 3 - gather_token_logprobs
def gather_token_logprobs(log_probs, token_ids):
    # TODO: Extract the log-probability of each observed token from a full vocab log-prob tensor...
    
    """
    log_probs: np.ndarray, shape (B, T, V)
    token_ids: np.ndarray, shape (B, T), 整数
    return:    np.ndarray, shape (B, T)
    """

    log_probs = np.asarray(log_probs, dtype=np.float64)
    token_ids = np.asarray(token_ids)

    # 1. 把 token_ids 扩成 (B, T, 1)，与 log_probs 的维度数对齐
    idx = token_ids[..., None]

    # 2. 沿最后一维（词表轴）按 idx 取值 → (B, T, 1)
    gathered = np.take_along_axis(log_probs, idx, axis=-1)

    # 3. 去掉尾部的 1 → (B, T)
    return gathered[..., 0]

# Step 4 - masked_sequence_logprob
def masked_sequence_logprob(token_logprobs, mask):
    # TODO: Sum per-token log-probabilities under a binary mask to obtain a single sequence log-probability per example.
    
    token_logprobs = np.asarray(token_logprobs, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)

    return np.sum(token_logprobs * mask, axis=-1, keepdims=False)

# Step 5 - init_policy_params (not yet solved)
# TODO: implement

# Step 6 - policy_token_logits (not yet solved)
# TODO: implement

# Step 7 - policy_sequence_logprob (not yet solved)
# TODO: implement

# Step 8 - sequence_logprob_grad (not yet solved)
# TODO: implement

# Step 9 - bradley_terry_loss (not yet solved)
# TODO: implement

# Step 10 - reward_accuracy (not yet solved)
# TODO: implement

# Step 11 - build_preference_pairs (not yet solved)
# TODO: implement

# Step 12 - sample_preference_batch (not yet solved)
# TODO: implement

# Step 13 - freeze_reference_logprobs (not yet solved)
# TODO: implement

# Step 14 - policy_reference_logratio (not yet solved)
# TODO: implement

# Step 15 - dpo_pair_margin (not yet solved)
# TODO: implement

# Step 16 - dpo_loss (not yet solved)
# TODO: implement

# Step 17 - dpo_loss_grad (not yet solved)
# TODO: implement

# Step 18 - dpo_train_step (not yet solved)
# TODO: implement

# Step 19 - train_dpo (not yet solved)
# TODO: implement

# Step 20 - length_normalized_logprob (not yet solved)
# TODO: implement

# Step 21 - ipo_loss (not yet solved)
# TODO: implement

# Step 22 - implicit_reward (not yet solved)
# TODO: implement

# Step 23 - preference_accuracy (not yet solved)
# TODO: implement

# Step 24 - kl_to_reference (not yet solved)
# TODO: implement

# Step 25 - reward_margin_stats (not yet solved)
# TODO: implement

# Step 26 - evaluate_dpo (not yet solved)
# TODO: implement

# Step 27 - run_dpo_pipeline (not yet solved)
# TODO: implement

