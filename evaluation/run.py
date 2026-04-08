#!/usr/bin/env python3

import argparse
import importlib
import os
import random

import numpy as np
import torch
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from vlnce_baselines.config.default import get_config
from vlnce_baselines.nonlearning_agents import evaluate_agent, nonlearning_inference


def _load_trainer_module(trainer_name: str) -> None:
    trainer_modules = {
        "dagger": "vlnce_baselines.dagger_trainer",
        "cma": "vlnce_baselines.cma_trainer",
        "recollect": "vlnce_baselines.recollect_trainer",
        "ddppo-waypoint": "vlnce_baselines.ddppo_waypoint_trainer",
        "navila": "vlnce_baselines.navila_trainer",
        "qwen": "vlnce_baselines.qwen_trainer",
    }

    module_name = trainer_modules.get(trainer_name)
    if module_name is not None:
        importlib.import_module(module_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "inference"],
        required=True,
        help="run type of the experiment (train, eval, inference)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="optional log file path that overrides config.LOG_FILE for this run",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(
    exp_config: str,
    run_type: str,
    num_chunks: int,
    chunk_idx: int,
    log_file: str = None,
    opts=None,
) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """
    config = get_config(exp_config, opts)

    if log_file is not None and len(log_file) > 0:
        config.defrost()
        config.LOG_FILE = log_file
        config.freeze()

    logger.info(f"config: {config}")
    logdir = "/".join(config.LOG_FILE.split("/")[:-1])
    if logdir:
        os.makedirs(logdir, exist_ok=True)
    logger.add_filehandler(config.LOG_FILE)

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    if run_type == "eval":
        torch.backends.cudnn.deterministic = True
        if config.EVAL.EVAL_NONLEARNING:
            evaluate_agent(config)
            return

    if run_type == "inference" and config.INFERENCE.INFERENCE_NONLEARNING:
        nonlearning_inference(config)
        return

    _load_trainer_module(config.TRAINER_NAME)
    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"

    trainer = trainer_init(config, num_chunks, chunk_idx)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == "inference":
        trainer.inference()


if __name__ == "__main__":
    main()
