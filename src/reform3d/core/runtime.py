"""Deterministic execution policy for the public evaluation runner."""
from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module
import os
import random


def seed_runtime(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextmanager
def deterministic_runtime():
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits

    # Threadpool limits apply to loaded libraries; initialize the probe backends first.
    for module in ("scipy.linalg", "sklearn.linear_model"):
        import_module(module)
    thread_environment = {name: os.environ.get(name) for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
    )}
    previous_env = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if torch.cuda.is_initialized() and previous_env not in {":4096:8", ":16:8"}:
        raise RuntimeError("Start evaluation in a fresh process: CUDA was initialized without deterministic cuBLAS settings")
    workspace = previous_env if previous_env in {":4096:8", ":16:8"} else ":4096:8"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    previous = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "fp16_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "threads": torch.get_num_threads(),
    }
    attention_backends = {name: getattr(torch.backends.cuda, f'{name}_sdp_enabled')()
                          for name in ('flash', 'math', 'mem_efficient', 'cudnn')
                          if hasattr(torch.backends.cuda, f'{name}_sdp_enabled')}
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        # Spawned workers and subsequently loaded numerical libraries inherit this policy.
        for name in thread_environment:
            os.environ[name] = "1"
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        # Frozen inference uses deterministic Flash forward, with a math fallback.
        for name in attention_backends:
            getattr(torch.backends.cuda, f'enable_{name}_sdp')(name in {'flash', 'math'})
        torch.set_num_threads(1)
        seed_runtime(0)
        with threadpool_limits(limits=1):
            yield {
                "policy": "deterministic-v2", "inference_seed": 0,
                "deterministic_algorithms": True, "warn_only": False,
                "cudnn_benchmark": False, "tf32": False,
                "fp16_reduced_precision_reduction": False,
                "cublas_workspace_config": workspace, "cpu_threads": 1,
                "thread_environment": {name: "1" for name in thread_environment},
                "attention_backends": [name for name in ('flash', 'math') if name in attention_backends],
            }
    finally:
        torch.use_deterministic_algorithms(previous["deterministic"], warn_only=previous["warn_only"])
        torch.backends.cudnn.benchmark = previous["benchmark"]
        torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]
        torch.backends.cudnn.allow_tf32 = previous["cudnn_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = previous["matmul_tf32"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous["fp16_reduction"]
        for name, enabled in attention_backends.items():
            getattr(torch.backends.cuda, f'enable_{name}_sdp')(enabled)
        torch.set_num_threads(previous["threads"])
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        for name, value in thread_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        # cuBLAS can stay initialized after this context exits.
        if not torch.cuda.is_initialized():
            if previous_env is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_env
