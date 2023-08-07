from typing import Callable, Dict, List

import torch

_REGISTRY: Dict[str, Callable[..., torch.nn.Module]] = {}


def register_encoder(name: str):
    def _decorator(func):
        assert name not in _REGISTRY, f"model {name} already registered"
        _REGISTRY[name] = func
        return func

    return _decorator


def create_encoder(name: str, *args, **kwargs) -> torch.nn.Module:
    assert name in _REGISTRY, f"model {name} not registered"
    model = _REGISTRY[name](*args, **kwargs)
    return model


def list_encoders() -> List[str]:
    return list(_REGISTRY.keys())
