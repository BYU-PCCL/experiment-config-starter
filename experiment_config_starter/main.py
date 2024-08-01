import json
import os
from hashlib import sha256
from pprint import pprint

import sys
import torch
import torch.distributed
import wandb
import wandb.env
from accelerate import Accelerator
from omegaconf import OmegaConf
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    TrainingArguments,
)

from .config import ModelConfig, TrainConfig
from .trainer import initialize_trainer
from .data import load_dataset


def initialize_model(
    tokenizer: PreTrainedTokenizer, model_cfg: ModelConfig
) -> AutoModelForCausalLM:
    config = AutoConfig.from_pretrained(
        model_cfg.model_type,
        vocab_size=len(tokenizer),
        **OmegaConf.to_container(model_cfg),  # type: ignore
    )

    model_cfg = AutoModelForCausalLM.from_config(config)

    return model_cfg  # type: ignore


def train(cfg: TrainConfig):
    # Set WANDB_PROJECT based on config
    os.environ["WANDB_PROJECT"] = cfg.wandb.project

    accelerator = Accelerator()

    # If we are running a sweep, broadcast the config to all processes. This is super
    # important.
    if os.environ.get(wandb.env.SWEEP_ID) is not None:
        if accelerator.is_main_process:
            config_list = [OmegaConf.to_container(cfg)]
        else:
            config_list = [None]

        torch.distributed.broadcast_object_list(config_list, src=0)

        cfg = OmegaConf.create(config_list[0])  # type: ignore

    # Hash config dict so you don't overwrite your experiment output
    config_hash = sha256(json.dumps(OmegaConf.to_container(cfg)).encode()).hexdigest()[
        :8
    ]
    run_name = cfg.experiment.run_name + "_" + config_hash

    wandb_config_dict = OmegaConf.to_container(cfg, resolve=True)

    if os.environ.get(wandb.env.SWEEP_ID) is not None:
        if accelerator.is_main_process:
            wandb.init(name=cfg.experiment.run_name)

            # Only in the case that you're running a sweep: update the config with the
            # sweep config. Otherwise this is done by
            # .wandb_callback.ConfigurableWandbCallback
            wandb.config.update(dict(rosetta=wandb_config_dict))

    train_arguments = TrainingArguments(
        **OmegaConf.to_container(cfg.trainer, resolve=True),  # type: ignore
        run_name=run_name,
        output_dir=cfg.experiment.parent_output_dir + run_name,
        #
        # These are all irrelevant to this example, but you can use them
        #
        # fp16=torch.cuda.is_available(),
        # group_by_length=True,
        # auto_find_batch_size=False,
        # evaluation_strategy="steps",
        # do_eval=True,
        # include_inputs_for_metrics=True,
    )

    if accelerator.is_main_process:
        # Pretty print out config in terminal
        pprint(dict(rosetta=OmegaConf.to_container(cfg, resolve=True)))

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.path)
    train_dataset, eval_dataset = load_dataset(cfg.experiment)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=cfg.model.is_mlm)
    model = initialize_model(tokenizer, cfg.model)

    trainer = initialize_trainer(
        model,
        tokenizer,
        data_collator,
        train_dataset,
        eval_datasets,
        experiment_cfg=cfg.experiment,
        trainer_args=train_arguments,
        wandb_config=wandb_config_dict,
    )

    trainer.train()

    trainer._save_checkpoint(trainer.model, None)


if __name__ == "__main__":
    # From OLMo
    def clean_opt(arg: str) -> str:
        if "=" not in arg:
            arg = f"{arg}=True"
        name, val = arg.split("=", 1)
        name = name.strip("-").replace("-", "_")
        return f"{name}={val}"

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise RuntimeError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [])
