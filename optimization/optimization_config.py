# ========================================= config for optimization process ==========================================
# ====================================================================================================================


# the model you want to optimize
pretrained_model = "Qwen/Qwen2.5-14B-Instruct"
#Qwen/Qwen2.5-7B-Instruct
#"Qwen/Qwen2.5-14B-Instruct"

# the training data and evaluation data
train_dataset = "CodeContests_train"
eval_dataset = "CodeContests"

# total steps for optimization
total_steps = 350

# evaluate every eval_interval steps
eval_interval = 20

# save optimized model every save_interval steps
save_interval = 50









# ============= config for sampling in each step =================

# number of codes and unit tests sampled in each step
k_code = 16
k_case = 32
k_case_keep = 16    #set None to disable pruning

# temperature
temp = 1.0
#0.8

# number of tasks for sampling in each step
n_sample_per_step = 100

# GPU usage for vllm inference, [[0]] represents only one engine with one GPU; [[0, 1], [2, 3]] represents two engines each with 2 GPUs
# each engine loads a model, so for <=7B model, you can only set GPU numbers for each engine <= 2
# Default 7B setup: two independent 1-GPU vLLM workers for higher sampling
# throughput. For larger models, switch back to [[0, 1]] to use tensor
# parallelism across both GPUs.
gpu_groups = [[0,1]]

# max ground-truth unit test we can use here
max_ground_truth_test = 8

# set to 1 by default
max_input_examples = 1

# maximum number of tokens the vLLM engine can handle in a single sequence
max_model_len = 6000

# max token model can generate for each quiry
max_generation_token = 3000

# the probability for providing public unit test example in prompt
p_give_example = 1.0

# the prompt design for code generation and unit test generation
# === MUST_MIRROR_BEGIN(code_prompt): evaluation/evaluation_config.py:system_prompts ===
# Train/eval prompts must stay byte-identical or the model sees a different task at eval.
system_prompts = """<|im_start|>You are a helpful assistant that helps users solve programming problems. \
<|im_end|>\n<|im_start|>User: Think through the problem before writing your {{language}} solution. {{special_requirements}}
Here is the problem:\n{{problem}} <|im_end|>\n<|im_start|>Assistant: """
# === MUST_MIRROR_END(code_prompt) ===

# === MUST_MIRROR_BEGIN(case_prompt): evaluation/evaluation_config.py:system_case_prompts ===
# Train/eval prompts must stay byte-identical. The eval parser depends on the format
# this prompt asks for; changing one without the other causes silent UT-acc collapse.
system_case_prompts = """<|im_start|>You are a helpful assistant specialized in generating test examples for coding tasks. \
<|im_end|>\n<|im_start|>User: Given a coding task, your goal is not to write the solution, but to generate a new test example consisting of an input, an expected output, and an explanation.
Here is the problem:\n{{problem}}\n
{{example_intro}}
Your test example must be completely accurate and conform to the problem's format requirements, while also being discriminative enough to distinguish correct code from incorrect code.
Begin by thinking carefully and reasoning step by step inside <reasoning> tags to derive an input and output you are confident are correct. A good approach is to first design an input you can reliably work through, then compute the output step by step. If you are unsure about the output, revise or redesign the input until you are certain. Skipping this process and directly providing input/output pairs is strongly discouraged, as it frequently leads to incorrect results.
Once you have thoroughly completed your reasoning and derivation, your final response MUST follow this exact format:\n
<reasoning>\nyour explanation here.\n</reasoning>\n\n<answer>\n<input>\nyour raw input here\n</input>\n<output>\nyour raw output here\n</output>\n</answer>\n\n <|im_end|>\n<|im_start|>Assistant: <reasoning>"""
# === MUST_MIRROR_END(case_prompt) ===

# some special requirements for code generation
special_requirements = """You should use input() to input and print() to output in your script. """















# ============= config for execution in each step =================

# how many parts the execution tasks are divided into (too small may get stuck), should be proportion to k_code * k_case * n_sample_per_step
num_chunks = 48 * 4  # kept for eval.py compatibility

# max concurrent worker threads for execute.py (each thread manages one subprocess)
# defaults to cpu_count() if not set; tune based on your machine
num_executors = 128

# Reuse a bounded pool of execution worker processes instead of spawning one
# process per generated-code/test pair.  Set False to fall back to the legacy
# chunked process-per-task runner.
use_executor_pool = True

# the BoN setting you want to see in each step's output
scale_tuple_list = [(4, 4), (16, 16)]















# ============= config for oracle correction (Proposal 1) =================

# Master toggle — False means identical behaviour to baseline.
# Keep this False for the CoVer reward: CoVer computes signed MI on the raw column
# and does not need consensus rewriting. Leaving this True adds a silent upstream
# filter (low-consensus tests get their outputs zero'd) plus ~5x wasted execution
# per step.
enable_oracle_correction = False

# Number of highest-GT-scoring codes used to vote on the corrected oracle
oracle_correction_topk = 5

# Minimum weighted-consensus ratio required to accept the execution oracle
# (max_vote_weight / total_valid_weight).  Range [0, 1].
oracle_correction_confidence = 0.6

# Minimum GT pass rate of the *best* available code before attempting correction.
# Guards against early-training noise when all codes are terrible.
oracle_correction_min_quality = 0.25

# Minimum sum of quality² weights across valid-executing codes before trusting the vote.
# Filters single-marginal-voter cases (one code barely above min_quality, rest fail).
oracle_correction_min_total_weight = 0.1

# Reasoning bonus multiplier: when the model's own oracle matches the execution
# oracle, the base case reward is scaled by (1 + alpha) before normalisation.
oracle_correction_alpha = 0.2




# ============= config for reward assignment in each step =================


# Ablation: when True, use binary y ∈ {0,1} (1 iff code passes all GT tests) instead of
# graded y ∈ {0, 1/Gτ, ..., 1}. Reduces n_y_bins from G_tau+1 to 2 in the plug-in MI estimator.
cover_y_binary = False

# set True by default
separate_training = True

# set False for standard base model, True for long-CoT model
enable_efficient = False
# when enable_efficient = True, responses with length >= max_len_threshold enforce negative reward, responses with length <= min_len_threshold no need for length penalty
max_len_threshold = 8000
min_len_threshold = 1000

# Keep False — we always include the example in the prompt (p_give_example=1.0)
post_stage = False

















# ============= config for training in each step =================

# number of GPUs for training
# 2 nodes × 1 GPU each: ZeRO-3 data parallel across both GPUs
total_num_nodes = 2

# learning rate
actor_learning_rate = 1e-6

# 0 by default
num_warmup_steps = 0

# number of updates each step, 1 by default
policy_update_steps = 1

# KL loss setting
use_kl_loss = True
kl_loss_coef = 0.01
use_kl_estimator_k3 = True

# max prompt (inquiry) length in collected data
prompt_max_len = 2000

# generation token limit
generate_max_len = 3000

# we use packing here instead of batching for training, and we need packing_max_len >= generate_max_len + prompt_max_len
packing_max_len = 20000

# number of epoch for this training, 1 by default
max_epochs = 1

# the output model name
optimized_model_name = "Qwen_14B_CoVer"

# ============= wandb logging =================
use_wandb = True
wandb_project = "CoVer_TEST"
wandb_run_name = "Qwen_14B_CoVer"

# Set to an existing W&B run id to resume that run; leave None to start a fresh run.
# Set to an existing W&B run id to resume that run; leave None to start a fresh run.
wandb_run_id = None






# ============= config for evaluation during the optimization =================

eval_k_code = 16
eval_num_chunks = 128 * 4
eval_no_example = True
eval_max_test = 8
