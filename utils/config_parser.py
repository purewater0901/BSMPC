import os
import re
import yaml


def parse_yaml(yaml_file):
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)


def parse_cfg(cfg, cfg_path: str):
    """Parses a config file and returns an OmegaConf object."""
    data = parse_yaml(cfg_path)
    for key, value in data.items():
        if key == "bisim_coef" and hasattr(cfg, key) and cfg.bisim_coef is not None:
            continue
        setattr(cfg, key, value)

    default_data = parse_yaml(cfg.default_config_path)
    for key, value in default_data.items():
        setattr(cfg, key, value)

    cfg.episode_length = int(cfg.episode_length / cfg.action_repeat)
    cfg.train_steps = int(cfg.train_steps / cfg.action_repeat)
    print(cfg)
    return cfg
