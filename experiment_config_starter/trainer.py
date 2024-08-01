from typing import Dict

from config import ExperimentConfig
from datasets import Dataset
from transformers import (
    AutoModel,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_pt_utils import AcceleratorConfig, LengthGroupedSampler
from transformers.trainer_utils import has_length, seed_worker
from wandb_callback import ConfigurableWandbCallback


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True


class CheckpointFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_save = True


def initialize_trainer(
    model: AutoModel,
    tokenizer: PreTrainedTokenizerFast,
    data_collator: DataCollatorForLanguageModeling,
    train_dataset: Dataset,
    eval_datasets: Dict[str, Dataset],
    experiment_cfg: ExperimentConfig,
    trainer_args: TrainingArguments,
    wandb_config=None,
):
    trainer_args.accelerator_config = AcceleratorConfig(
        dispatch_batches=True, split_batches=True
    )  # type: ignore

    trainer = Trainer(
        model=model,  # type: ignore
        tokenizer=tokenizer,
        args=trainer_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,  # type: ignore
    )

    custom_wandb_callback = ConfigurableWandbCallback(custom_config=wandb_config)

    trainer.add_callback(custom_wandb_callback)
    trainer.add_callback(EvaluateFirstStepCallback())
    trainer.add_callback(CheckpointFirstStepCallback())

    return trainer
