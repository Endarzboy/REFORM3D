"""Require complete encoder weights while permitting explicitly unused heads."""

from collections.abc import Mapping

import torch
from torch import nn


def load_encoder_state(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    allowed_extra_prefixes: tuple[str, ...] = (),
) -> None:
    expected = set(model.state_dict())
    actual = set(state_dict)
    missing = sorted(expected - actual)
    unexpected = sorted(
        key for key in actual - expected
        if not key.startswith(allowed_extra_prefixes)
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match the encoder: missing={missing}; "
            f"unexpected={unexpected}"
        )
    model.load_state_dict({key: state_dict[key] for key in expected}, strict=True)
