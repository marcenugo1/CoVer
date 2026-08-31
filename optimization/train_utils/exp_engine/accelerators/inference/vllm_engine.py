try:
    from vllm.core.scheduler import Scheduler
except ImportError:
    # vLLM 0.16+ uses v1 API; Scheduler lives in v1 and has a different constructor.
    # We only need it for the legacy scheduler rebuild in backload_to_gpu.
    Scheduler = None


class LLMActor:
    def __init__(self, *args, **kwargs):
        import vllm

        self.__version__ = vllm.__version__
        assert self.__version__ >= "0.4.1", "OpenRLHF only supports vLLM >= 0.4.1"

        self.use_gpu_executor = kwargs["tensor_parallel_size"] == 1

        # See https://github.com/vllm-project/vllm/blob/main/vllm/executor/gpu_executor.py
        if self.use_gpu_executor:
            from .vllm_worker_wrap import OffloadableVLLMWorker

            vllm.worker.worker.Worker = OffloadableVLLMWorker
        else:
            # RayGPUExecutor
            # See the patch https://github.com/vllm-project/vllm/commit/479d69fad0538f04cb22bf13e76ff91cfeb8a4e5
            kwargs["worker_use_ray"] = True

            if vllm.__version__ > "0.6.4.post1":
                # https://github.com/vllm-project/vllm/pull/10555
                kwargs[
                    "worker_cls"
                ] = "orz.exp_engine.accelerators.inference.vllm_worker_wrap.OffloadableVLLMWorker"
            else:
                RayWorkerWrapperPath = vllm.executor.ray_utils

                class RayWorkerWrapper(RayWorkerWrapperPath.RayWorkerWrapper):
                    def __init__(self, *args, **kwargs) -> None:
                        kwargs[
                            "worker_module_name"
                        ] = "orz.exp_engine.accelerators.inference.vllm_worker_wrap"
                        kwargs["worker_class_name"] = "OffloadableVLLMWorker"
                        super().__init__(*args, **kwargs)

                RayWorkerWrapperPath.RayWorkerWrapper = RayWorkerWrapper

        kwargs["enforce_eager"] = True
        self.llm = vllm.LLM(*args, **kwargs)
        engine = self.llm.llm_engine
        # vLLM 0.16+ (v1 API) stores config on vllm_config
        vllm_config = getattr(engine, "vllm_config", None)
        self.scheduler_config = getattr(engine, "scheduler_config", None) or (
            vllm_config.scheduler_config if vllm_config else None
        )
        self.model_config = engine.model_config
        self.cache_config = engine.cache_config
        self.lora_config = getattr(engine, "lora_config", None) or (
            vllm_config.lora_config if vllm_config else None
        )
        self.parallel_config = getattr(engine, "parallel_config", None) or (
            vllm_config.parallel_config if vllm_config else None
        )

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                master_address, master_port, rank_offset, world_size, group_name, backend
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers(
                "init_process_group", master_address, master_port, rank_offset, world_size, group_name, backend
            )

    def get_ip_and_port(self):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.get_ip_and_port()
        else:
            return self.llm.llm_engine.model_executor._run_workers("get_ip_and_port")

    def offload_to_cpu(self):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.offload_cpu()
        else:
            return self.llm.llm_engine.model_executor._run_workers("offload_cpu")

    def backload_to_gpu(self):
        if self.use_gpu_executor:
            self.llm.llm_engine.model_executor.driver_worker.load_gpu()
        else:
            self.llm.llm_engine.model_executor._run_workers("load_gpu")
        # rebuild scheduler (legacy vLLM only; v1 API uses different engine structure)
        if Scheduler is not None and hasattr(self.llm.llm_engine, "scheduler"):
            self.llm.llm_engine.scheduler = [
                Scheduler(
                    self.scheduler_config,
                    self.cache_config,
                    self.lora_config,
                    self.parallel_config.pipeline_parallel_size,
                    self.async_callbacks[v_id] if self.model_config.use_async_output_proc else None,
                )
                for v_id in range(self.parallel_config.pipeline_parallel_size)
            ]

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self.stop_remote_worker_execution_loop()

        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.update_weight(name, dtype, shape, empty_cache)
        else:
            return self.llm.llm_engine.model_executor._run_workers("update_weight", name, dtype, shape, empty_cache)

    def update_weight_internal_with_cuda_ipc(self, name, dtype, shape, cudaipc_handler, empty_cache=False):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.update_weight_internal_with_cuda_ipc(
                name, dtype, shape, cudaipc_handler, empty_cache
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers(
                "update_weight_internal_with_cuda_ipc", name, dtype, shape, cudaipc_handler, empty_cache
            )

    def stop_remote_worker_execution_loop(self):
        # Fix error for using 2 communication group
        # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
        if self.__version__ > "0.4.2":
            self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()

    def get_gpu_memory(self):
        """获取当前Actor使用的GPU内存"""
        import torch

        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated() / 1024**2  # 转换为MB

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        stats = {}
        model_runner = self.llm.llm_engine.model_executor.driver_worker.model_runner
        for name, param in model_runner.model.named_parameters():
            tensor_stats = {
                "mean": param.mean().item(),
                "std": param.std().item(),
                "norm": param.norm().item(),
                "shape": tuple(param.shape),
                "max": param.max().item(),
                "min": param.min().item(),
            }
            stats[name] = tensor_stats
        return stats