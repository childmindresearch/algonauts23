import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from algonauts23 import FMRI_DIM, NUM_SUBS

from .registry import register_encoder

Shapes = List[Tuple[int, ...]]


class GroupLinearEncoder(nn.Module):
    """
    A multi-subject multi-layer linear encoding head for predicting fMRI brain activity.

    Args:
        feature_shapes: list of feature shapes, excluding the batch dimension. Accepts
            either convolutional (C, H, W), or transformer (L, C) shapes.
        proj_dim: latent dimension before final embedding
        out_dim: output dimension
        hidden_dim: latent dimension after feature projection
        group_size: number of subjects
        dropout: dropout rate applied to input features
        norm: apply batch norm after feature projection
    """

    def __init__(
        self,
        *,
        feature_shapes: List[Tuple[int, ...]],
        proj_dim: int,
        out_dim: int = FMRI_DIM,
        hidden_dim: int = 1024,
        group_size: int = NUM_SUBS,
        dropout: float = 0.0,
        norm: bool = True,
    ) -> None:

        super().__init__()
        self.feature_shapes = feature_shapes
        self.group_size = group_size
        self.proj_dim = proj_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.feat_projs = nn.ModuleList(
            [
                FeatureProjection(shape, hidden_dim, norm=norm, dropout=dropout)
                for shape in feature_shapes
            ]
        )

        # Shared and subject-specific projection
        self.shared_proj = nn.Linear(hidden_dim, proj_dim, bias=False)
        self.subj_proj = GroupLinear(hidden_dim, proj_dim, group_size, bias=True)
        self.embed = nn.Linear(proj_dim, out_dim, bias=True)

    def forward(
        self, features: List[torch.Tensor], indices: Optional[torch.Tensor]
    ) -> torch.Tensor:
        pred, _ = self.predict(features, indices)
        return pred

    def predict(
        self, features: List[torch.Tensor], indices: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict outputs from feature list. Returns a tuple of (pred, group_pred). If
        indices is None, pred == group_pred. Note that pred = group_pred +
        subj_residual.
        """

        # Project each feature down to the hidden dimension and average
        features = [
            feat_proj(feat) for feat_proj, feat in zip(self.feat_projs, features)
        ]
        latent = torch.stack(features).mean(dim=0)

        group_pred = self.embed(self.shared_proj(latent))
        if indices is None:
            return group_pred, group_pred

        subj_res = self.embed(self.subj_proj(latent, indices))
        pred = group_pred + subj_res
        return pred, group_pred

    def load_embed(self, state_dict: Dict[str, torch.Tensor], freeze: bool = True):
        """
        Load final embedding weight. E.g. from pre-computed PCA.
        """
        # Slice to match embed dimension
        state_dict = state_dict.copy()
        state_dict["weight"] = state_dict["weight"][:, : self.embed.in_features]

        self.embed.load_state_dict(state_dict)
        if freeze:
            for p in self.embed.parameters():
                p.requires_grad_(False)


class FeatureProjection(nn.Module):
    """
    Project a feature tensor down to a 1D vector.
    """

    def __init__(
        self,
        feature_shape: Tuple[int, ...],
        embed_dim: int,
        norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        # (N, C, H, W) or (N, *, C)
        channel_dim = 1 if len(feature_shape) == 3 else -1

        self.flat = Flatten(channel_dim)
        self.drop = nn.Dropout(dropout)
        self.proj = FactorLinear2D(feature_shape, embed_dim, bias=False)
        self.norm = nn.BatchNorm1d(embed_dim) if norm else nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.flat(x)
        x = self.drop(x)
        x = self.proj(x)
        x = self.norm(x)
        return x


class FactorLinear2D(nn.Module):
    """
    A linear layer for 2D tensor input features with a rank-one weight tensor.

    Args:
        in_shape: input feature shape, (L, C)
        out_features: output feature dimension K
        bias: with additive bias

    Shape:
        input: (N, L, C)
        output: (N, K)
    """

    def __init__(
        self,
        in_shape: Tuple[int, int],
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_shape = in_shape
        self.out_features = out_features

        T, C = in_shape
        self.weight1 = nn.Parameter(torch.empty((out_features, T)))
        self.weight2 = nn.Parameter(torch.empty((out_features, C)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Copied from nn.Linear, except for the bias init which is copied from
        # batchnorm.

        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        # x: (N, L, C)
        assert x.ndim == 3 and x.shape[1:] == self.in_shape

        # We could use einsum here, but it takes too much memory

        # (N, L, K)
        x = torch.matmul(x, self.weight2.t())
        # (N, K)
        x = torch.sum(x * self.weight1.t(), dim=-2)

        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self) -> str:
        return (
            f"in_shape={self.in_shape}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


class GroupLinear(nn.Module):
    """
    Per-subject linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.bias = bias

        self.fc = nn.Linear(in_features, group_size * out_features, bias=bias)

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, "Invalid x shape"
        N = x.size(0)
        indices = indices.view(N, 1, 1)
        x = self.fc(x)
        x = x.view(N, self.group_size, self.out_features)
        x = torch.take_along_dim(x, indices, 1)
        x = x.squeeze(1)
        return x


class Flatten(nn.Module):
    """
    Flatten an input tensor of shape (N, *, C, *) to (N, L, C)
    """

    def __init__(self, channel_dim: int):
        super().__init__()
        self.channel_dim = channel_dim

    def forward(self, x: torch.Tensor):
        x = torch.movedim(x, self.channel_dim, -1)
        x = torch.flatten(x, 1, -2)
        return x


@register_encoder("grouplin")
def group_linear_encoder(
    *,
    feature_shapes: List[Tuple[int, ...]],
    proj_dim: int,
    out_dim: int = FMRI_DIM,
    hidden_dim: int = 1024,
    group_size: int = NUM_SUBS,
    dropout: float = 0.0,
    norm: bool = True,
    embed_state: Optional[Union[str, Path]] = None,
    freeze_embed: bool = True,
    **kwargs,
):
    model = GroupLinearEncoder(
        feature_shapes=feature_shapes,
        proj_dim=proj_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        group_size=group_size,
        dropout=dropout,
        norm=norm,
    )

    if embed_state is not None:
        logging.info("Loading embed state: %s", embed_state)
        state = torch.load(embed_state, map_location="cpu")
        model.load_embed(state, freeze=freeze_embed)
    return model
