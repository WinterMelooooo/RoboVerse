from diffusers.optimization import (
    TYPE_TO_SCHEDULER_FUNCTION,
    Optimizer,
    Optional,
    SchedulerType,
    Union,
)
import math
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import OmegaConf


def get_composite_scheduler(
    default_scheduler_name: Union[str, SchedulerType],
    optimizer: Optimizer,
    default_num_warmup_steps: Optional[int] = None,
    default_num_training_steps: Optional[int] = None,
    last_epoch: int = -1,
    step_per_epoch: Optional[int] = None,
    params_groups: Optional[list] = None,
):
    lr_lambda = [get_lambda_function(
        default_scheduler_name,
        num_warmup_steps=default_num_warmup_steps,
        num_training_steps=default_num_training_steps,
        step_per_epoch=step_per_epoch)] # Initialize with the default scheduler

    if params_groups is not None:
        for i, scheduler_kwargs in enumerate(params_groups):

            scheduler_kwargs = preprocess_scheduler_kwargs(
                scheduler_kwargs,
                default_scheduler_lambda_name=default_scheduler_name,
                default_num_warmup_steps=default_num_warmup_steps,
                default_num_training_steps=default_num_training_steps,
                step_per_epoch=step_per_epoch,
            )

            lambda_func = get_lambda_function(**scheduler_kwargs)
            lr_lambda.append(lambda_func)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def preprocess_scheduler_kwargs(
    scheduler_kwargs: dict,
    default_scheduler_lambda_name: Optional[str] = None,
    default_num_warmup_steps: Optional[int] = None,
    default_num_training_steps: Optional[int] = None,
    step_per_epoch: Optional[int] = None,
):
    scheduler_kwargs = scheduler_kwargs.copy()
    scheduler_kwargs = OmegaConf.to_container(
        scheduler_kwargs, resolve=True, enum_to_str=True
    )
    scheduler_kwargs.pop("name")
    scheduler_kwargs.pop("lr", None)
    scheduler_kwargs.pop("weight_decay", None)
    if "scheduler_lambda_name" not in scheduler_kwargs:
        scheduler_kwargs["scheduler_lambda_name"] = default_scheduler_lambda_name
    if "num_warmup_steps" not in scheduler_kwargs:
        scheduler_kwargs["num_warmup_steps"] = default_num_warmup_steps
    if "num_training_steps" not in scheduler_kwargs:
        scheduler_kwargs["num_training_steps"] = default_num_training_steps
    if "step_per_epoch" not in scheduler_kwargs:
        scheduler_kwargs["step_per_epoch"] = step_per_epoch
    return scheduler_kwargs


def get_lambda_function(
    scheduler_lambda_name: Union[str, SchedulerType],
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs,
):
    if scheduler_lambda_name == "cosine":
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(kwargs.get("num_cycles", 0.5)) * 2.0 * progress)))
        return lr_lambda
    elif scheduler_lambda_name == "freeze":
        after_freeze_func_name = kwargs["after_freeze_func_name"]
        freeze_steps = kwargs["freeze_epochs"] * kwargs["step_per_epoch"]
        after_freeze = get_lambda_function(
            after_freeze_func_name,
            num_warmup_steps,
            num_training_steps - freeze_steps,
            **kwargs)
        return lambda current_step: after_freeze(current_step-freeze_steps) if current_step >= freeze_steps else 0.0
    else:
        raise ValueError(f"Unknown scheduler name: {scheduler_lambda_name}.")


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs,
):
    """
    Added kwargs vs diffuser's original implementation

    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(
            f"{name} requires `num_warmup_steps`, please provide that argument."
        )

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(
            f"{name} requires `num_training_steps`, please provide that argument."
        )

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **kwargs,
    )
