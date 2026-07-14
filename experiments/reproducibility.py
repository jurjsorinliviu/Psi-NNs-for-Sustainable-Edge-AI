import copy
import hashlib
import json
import os
import random
from typing import Any, Dict, Tuple

import numpy as np
import torch


def configure_reproducibility(seed: int, strict: bool = True) -> Dict[str, Any]:
    """Configure per-seed reproducibility for Python, NumPy, and PyTorch."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")

    status = {
        "seed": seed,
        "strict_requested": strict,
        "strict_enabled": False,
        "warn_only_fallback": False,
    }

    try:
        torch.use_deterministic_algorithms(strict)
        status["strict_enabled"] = strict
    except Exception:
        torch.use_deterministic_algorithms(True, warn_only=True)
        status["warn_only_fallback"] = True

    return status


def select_experiment_device() -> torch.device:
    """Allow reproducibility-sensitive reruns to force CPU via environment."""
    force_cpu = os.environ.get("EDGE_AI_FORCE_CPU", "").strip().lower() in {"1", "true", "yes"}
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Clone a state dict onto CPU so it can be safely reused across regimes."""
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def load_cloned_state(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load a cloned state dict without mutating the stored reference."""
    model.load_state_dict(copy.deepcopy(state_dict))


def assert_state_dicts_equal(
    left: Dict[str, torch.Tensor],
    right: Dict[str, torch.Tensor],
    label_left: str,
    label_right: str,
) -> None:
    """Assert bitwise equality between two model state dicts."""
    if left.keys() != right.keys():
        raise AssertionError(f"State dict keys differ: {label_left} vs {label_right}")

    for name in left:
        if not torch.equal(left[name], right[name]):
            raise AssertionError(
                f"Initial weights differ for {name}: {label_left} vs {label_right}"
            )


def make_spatiotemporal_test_tensors(
    seed: int,
    n_test: int,
    device: torch.device,
    t_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a fixed test tensor pair reused across regimes within one seed."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x_test = torch.rand((n_test, 1), generator=generator)
    t_test = torch.rand((n_test, 1), generator=generator) * t_scale
    return x_test.to(device), t_test.to(device)


def tensor_digest(*tensors: torch.Tensor) -> str:
    """Return a stable hash for one or more tensors."""
    hasher = hashlib.sha256()
    for tensor in tensors:
        cpu_tensor = tensor.detach().cpu().contiguous()
        hasher.update(str(cpu_tensor.dtype).encode("utf-8"))
        hasher.update(str(tuple(cpu_tensor.shape)).encode("utf-8"))
        hasher.update(cpu_tensor.numpy().tobytes())
    return hasher.hexdigest()


def json_digest(payload: Any) -> str:
    """Return a stable hash for JSON-serializable content."""
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
