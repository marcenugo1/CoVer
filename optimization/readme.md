# Optimization

We introduce the main hyperparameters used in each module that you can customize in `optimization_config.py`. You can also customize these modules to achieve different optimization methods. When you start the training with `python run.py` (from the repository root), you can monitor the rollout experience and summarized results in the newly created `CoVer/optimization/temp_data` and  `CoVer/optimization/results` directories, respectively. The evaluation results every `eval_interval` steps will be save in the directory `CoVer/evaluation/results`.

## Main Configurations

`pretrained_model`: name of the model to be trained

`train_dataset` and `eval_dataset`: training and evaluation dataset

`eval_interval`: evaluate every eval_interval steps

`save_interval`: save optimized model every save_interval steps

## Sampling

Module: `sample.py`

`k_code` and `k_case`: number of codes and unit tests sampled in each step

`n_sample_per_step`: number of tasks for sampling in each step

`gpu_groups`: GPU usage for vllm inference. For example, [[0]] represents only one engine with one GPU; [[0, 1], [2, 3]] represents two engines each with 2 GPUs


## Execution

Module: `execute.py`

`num_chunks`: how many parts the execution tasks are divided into (too small may get stuck), should be proportion to k_code * k_case * n_sample_per_step


## Reward Assignment

Module: `reward.py`

`enable_efficient`: set False if it's standard model, set True if it's long-CoT model.

## Train

Module: `train.py`

`total_num_nodes`: number of GPUs

`actor_learning_rate`: learning rate

`prompt_max_len`: max prompt (inquiry) length in collected data

`generate_max_len`: generation token limit

`packing_max_len`: we use packing here instead of batching for training, and we need packing_max_len >= generate_max_len + prompt_max_len

`optimized_model_name`: the output model name, the model will be saved under `./ckpt`







