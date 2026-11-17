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

# Step 3 - gather_token_logprobs (not yet solved)
# TODO: implement

# Step 4 - masked_sequence_logprob (not yet solved)
# TODO: implement

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

