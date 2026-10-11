"""Small shared helpers: seeding, device choice, config loading, parameter counting."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides) -> dict:
    """Apply 'section.key=value' strings, e.g. 'model.n_layers=2'. Values are parsed as YAML."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        section, _, name = key.partition(".")
        if section not in cfg or not name:
            raise ValueError(f"unknown config section in override {item!r}")
        cfg[section][name] = yaml.safe_load(raw)
    return cfg


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
