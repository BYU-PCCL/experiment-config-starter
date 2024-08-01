from copy import deepcopy
from dataclasses import dataclass
from glob import glob
from os import PathLike
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar, Union, cast

from omegaconf import DictConfig, ListConfig
from omegaconf.errors import OmegaConfBaseException
from omegaconf import OmegaConf as om

C = TypeVar("C", bound="BaseConfig")
D = TypeVar("D", bound="DictConfig|ListConfig")


PathOrStr = Union[str, PathLike]


def is_url(path: PathOrStr) -> bool:
    return re.match(r"[a-z0-9]+://.*", str(path)) is not None


class ConfigurationError(Exception):
    """
    An error with a configuration file.
    """


class BaseConfig:
    @classmethod
    def _register_resolvers(cls, validate_paths: bool = True):
        # Expands path globs into a list.
        def path_glob(*paths) -> List[str]:
            out = []
            for path in paths:
                matches = sorted(glob(path))
                if not matches and validate_paths:
                    raise FileNotFoundError(f"{path} does not match any files or dirs")
                out.extend(matches)
            return out

        # Chooses the first path in the arguments that exists.
        def path_choose(*paths) -> str:
            for path in paths:
                if is_url(path) or Path(path).exists():
                    return path
            if validate_paths:
                raise FileNotFoundError(", ".join(paths))
            else:
                return ""

        om.register_new_resolver("path.glob", path_glob, replace=True)
        om.register_new_resolver("path.choose", path_choose, replace=True)

    @classmethod
    def new(cls: Type[C], **kwargs) -> C:
        cls._register_resolvers()
        conf = om.structured(cls)
        try:
            if kwargs:
                conf = om.merge(conf, kwargs)
            return cast(C, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise ConfigurationError(str(e))

    @classmethod
    def load(
        cls: Type[C],
        path: PathOrStr,
        overrides: Optional[List[str]] = None,
        key: Optional[str] = None,
        validate_paths: bool = True,
    ) -> C:
        """Load from a YAML file."""
        cls._register_resolvers(validate_paths=validate_paths)
        schema = om.structured(cls)
        try:
            raw = om.load(str(path))
            if key is not None:
                raw = raw[key]  # type: ignore
            conf = om.merge(schema, raw)
            if overrides:
                conf = om.merge(conf, om.from_dotlist(overrides))
            return cast(C, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise ConfigurationError(str(e))

    def save(self, path: PathOrStr) -> None:
        """Save to a YAML file."""
        om.save(config=self, f=str(path))

    def asdict(self, exclude: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        out = asdict(self)  # type: ignore
        if exclude is not None:
            for name in exclude:
                if name in out:
                    del out[name]
        return out

    def update_with(self, **kwargs):
        result = deepcopy(self)
        for key, value in kwargs.items():
            setattr(result, key, value)
        return result


@dataclass
class ModelConfig:
    model_type: str
    is_mlm: bool
    num_hidden_layers: int
    intermediate_size: int
    hidden_size: int
    num_attention_heads: int


@dataclass
class TokenizerConfig:
    path: str


@dataclass
class ExperimentConfig:
    # TODO: Make sure to fix this; run names should be the same as their datasets
    run_name: str
    data_dir: str
    train_file: str
    # We just evaluate from every dataset in this directory
    eval_dir: str
    # TODO: Do we keep this one?
    separate_vocab_batches: bool
    parent_output_dir: str
    train_size: Optional[int]


@dataclass
class TrainerConfig:
    learning_rate: float
    weight_decay: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    report_to: str
    logging_steps: int
    num_train_epochs: int
    eval_steps: Union[float, int]
    lr_scheduler_type: str


@dataclass
class WandbConfig:
    project: str
    notes: Optional[str]
    sweep_id: Optional[str]


@dataclass
class WandbTrainerSweepConfig:
    learning_rate: float
    batch_size: int
    num_epochs: int


@dataclass
class WandbExperimentConfig:
    separate_vocab_batches: bool


@dataclass
class WandbSweepConfig:
    trainer: WandbTrainerSweepConfig
    experiment: WandbExperimentConfig


@dataclass
class TrainConfig(BaseConfig):
    model: ModelConfig
    tokenizer: TokenizerConfig
    experiment: ExperimentConfig
    trainer: TrainerConfig
    wandb: WandbConfig
