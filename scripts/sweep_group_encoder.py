import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from hf_argparser import HfArg, HfArgumentParser

import wandb
from scripts.train_group_encoder import Args as TrainArgs
from scripts.train_group_encoder import main as train_main

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

PROJECT = "algonauts23-group-encoder"


@dataclass
class Args:
    sweep_cfg: Path = HfArg(aliases=["--scfg"], help="wandb sweep config")
    train_cfg: Path = HfArg(aliases=["--tcfg"], help="train config")
    sweep_id: Optional[str] = HfArg(aliases=["--sid"], default=None, help="sweep ID")
    count: int = HfArg(default=1, help="number of trials for this agent")


def main(args: Args):
    with args.sweep_cfg.open() as f:
        sweep_cfg = yaml.safe_load(f)
    with args.train_cfg.open() as f:
        train_cfg = yaml.safe_load(f)

    if args.sweep_id is None:
        sweep_id = wandb.sweep(sweep_cfg, project=PROJECT)
        logging.info(
            "Initialized new sweep: %s\n%s", sweep_id, yaml.safe_dump(sweep_cfg)
        )
        return

    logging.info("Launching sweep agent for sweep: %s", args.sweep_id)
    defaults = TrainArgs(**train_cfg)
    task = get_task(args.sweep_id, defaults)
    wandb.agent(args.sweep_id, function=task, project=PROJECT, count=args.count)


def get_task(sweep_id: str, defaults: TrainArgs):
    def task():
        with wandb.init(config=defaults.__dict__) as run:
            args = TrainArgs(**run.config)
            args.name = f"{sweep_id}/{run.name}"
            args.wandb = True
            args.sweep = True
            train_main(args)

    return task


if __name__ == "__main__":
    args: Args
    parser = HfArgumentParser(Args)
    (args,) = parser.parse_args_into_dataclasses()
    main(args)
