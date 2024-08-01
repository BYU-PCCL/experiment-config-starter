from transformers.integrations import WandbCallback


class ConfigurableWandbCallback(WandbCallback):
    def __init__(self, custom_config, *args, **kwargs):
        """Just do standard wandb init, but save the arguments for setup."""
        super().__init__(*args, **kwargs)
        self._custom_config = custom_config

    def setup(self, args, state, model, **kwargs):
        super().setup(args, state, model, **kwargs)
        if state.is_world_process_zero:
            self._wandb.config.update(self._custom_config, allow_val_change=True)
