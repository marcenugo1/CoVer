# the code base is adapted from open-reasoner-zero (orz) and openrlhf

import os
import re
import sys
import ast
import ray
import json
import copy
import torch
import asyncio
import argparse
import time
import numpy as np
from typing import List
from loguru import logger
from jinja2 import Template
from dataclasses import dataclass
from collections import defaultdict
from functools import cached_property
from typing_extensions import override
from itertools import islice, zip_longest
from omegaconf.listconfig import ListConfig
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable, List, Optional, Tuple


from train_utils.dataset import PromptDataset
from train_utils.rl.trainer import RayPPOTrainer
from train_utils.base_exp import BasePPOExp, BasePPOExpConfig



import optimization_config


class TrainTiming:
    def __init__(self, phase):
        self.phase = phase

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        timing_file = os.environ.get("COVER_TRAIN_TIMING_FILE")
        duration = end_time - self.start_time
        logger.info(f"{self.phase}, time cost: {duration:.2f}s")
        if timing_file:
            try:
                os.makedirs(os.path.dirname(timing_file), exist_ok=True)
                record = {
                    "step": os.environ.get("COVER_STEP"),
                    "pid": os.getpid(),
                    "phase": self.phase,
                    "duration_s": duration,
                    "start_time": self.start_time,
                    "end_time": end_time,
                    "status": "error" if exc_type else "ok",
                }
                if exc_type:
                    record["error_type"] = getattr(exc_type, "__name__", str(exc_type))
                    record["error"] = str(exc_val)
                with open(timing_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"Failed to write train timing record: {e}")



# our optimization scripts are in ./train_utils/rl/...

@dataclass
class PPOExpConfig(BasePPOExpConfig):
    use_compute_reward_fn: bool = True

    # Conditional settings with production values first
    total_num_nodes: int = optimization_config.total_num_nodes

    # resource related settings
    ref_num_nodes: int = total_num_nodes
    ref_num_gpus_per_node: int = 1
    actor_num_nodes: int = total_num_nodes
    actor_num_gpus_per_node: int = 1
    critic_num_nodes: int = total_num_nodes
    critic_num_gpus_per_node: int = 1
    colocate_all: bool = True
    colocate_critic_reward: bool = True
    colocate_actor_ref: bool = True
    vllm_num_engines: int = total_num_nodes
    vllm_tensor_parallel_size: int = 1
    adam_offload: bool = False
    offload_param: bool = False  # B200 has 183GB VRAM — no need to offload
    zero_stage: int = 3

    # path related settings
    pretrain: Optional[str] = optimization_config.pretrained_model

    rl_data: Optional[str] = "temp_data/rl_data.json"
    separate_training: bool = optimization_config.separate_training
    rl_code_data: Optional[str] = "temp_data/rl_code_data.json"
    rl_case_data: Optional[str] = "temp_data/rl_case_data.json"

    os.makedirs(os.path.dirname("./ckpt"), exist_ok=True)
    ckpt_path: str = f"ckpt"
    save_path: str = f"ckpt"
    tensorboard_log_dir: str = f"ckpt"

    # ppo related settings
    actor_learning_rate: float = optimization_config.actor_learning_rate
    num_warmup_steps: int = optimization_config.num_warmup_steps
    prompt_max_len: int = optimization_config.prompt_max_len
    enable_prefix_caching: bool = True
    advantage_normalize: bool = False

    policy_update_steps: int = optimization_config.policy_update_steps
    micro_train_batch_size: int = 1
    micro_forward_batch_size: int = 1
    freezing_actor_steps: int = -1
    init_kl_coef: float = 0
 
    kl_loss_coef: float = optimization_config.kl_loss_coef
    use_kl_loss: bool = optimization_config.use_kl_loss
    use_kl_estimator_k3: bool = optimization_config.use_kl_estimator_k3

    # generate related settings
    generate_max_len: int = optimization_config.generate_max_len
    packing_max_len: int = optimization_config.packing_max_len
    # Max tokens per forward+backward to reduce OOM; 0 = no chunking (one backward per batch) 4096
    train_max_tokens_per_backward: int = 32768

    max_epochs: int = optimization_config.max_epochs # number of iteration of training

    gamma: float = 1.0
    lambd: float = 1.0

    optimized_model_name: str = optimization_config.optimized_model_name



class PPOExp(BasePPOExp):
    @cached_property
    def trainer(self):
        return RayPPOTrainer(
            cfg=self.cfg,
            strategy=self.strategy,
            tokenizer=self.tokenizer,
            colocate_pg=self.get_colocate_pg,
        )


if __name__ == "__main__":

    import argparse, sys, asyncio, os
    with TrainTiming("train/process_total"):
        with TrainTiming("train/parse_args"):
            parser = argparse.ArgumentParser()
            parser.add_argument("--pretrain", type=str)
            args, unknown = parser.parse_known_args()

            sys.argv = [sys.argv[0]] + unknown

        with TrainTiming("train/build_config"):
            cfg = PPOExpConfig()
            if args.pretrain is not None:
                cfg.pretrain = args.pretrain

        with TrainTiming("train/create_exp"):
            exp = PPOExp().set_cfg(cfg)

        with TrainTiming("train/log_config_and_dirs"):
            import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
            import optimization_config

            logger.info(exp.get_cfg_as_str(exp.cfg))
            if not os.path.exists(exp.cfg.save_path):
                os.makedirs(exp.cfg.save_path, exist_ok=True)
            if not os.path.exists(exp.cfg.tensorboard_log_dir):
                os.makedirs(exp.cfg.tensorboard_log_dir, exist_ok=True)
            if not os.path.exists(exp.cfg.ckpt_path):
                os.makedirs(exp.cfg.ckpt_path, exist_ok=True)

        with TrainTiming("train/asyncio_run"):
            asyncio.run(exp.run())
