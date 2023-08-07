from dataclasses import dataclass
from typing import Tuple

import torch
from timm.data import resolve_model_data_config, str_to_interp_mode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model
from torchvision.transforms import InterpolationMode


@dataclass
class DataConfig:
    img_size: int = 224
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD
    interp_mode: InterpolationMode = InterpolationMode.BICUBIC


def create_trunk(name: str, **kwargs) -> Tuple[torch.nn.Module, DataConfig]:
    # Special cases
    if "hiera" in name:
        model = torch.hub.load(
            "facebookresearch/hiera",
            name,
            pretrained=True,
            checkpoint="mae_in1k_ft_in1k",
        )
        cfg = DataConfig()
    elif "dinov2" in name:
        model = torch.hub.load("facebookresearch/dinov2", name)
        cfg = DataConfig()

    # Fall back to timm as default
    # Note, this is the only case where kwargs are passed through
    else:
        model = create_model(name, pretrained=True, **kwargs)
        data_config = resolve_model_data_config(model)

        img_size = data_config["input_size"][1]
        interp_mode = str_to_interp_mode(data_config["interpolation"])
        cfg = DataConfig(
            img_size=img_size,
            mean=data_config["mean"],
            std=data_config["std"],
            interp_mode=interp_mode,
        )
    return model, cfg
