import collections
from typing import Callable, Dict, List

import torch
import torch.nn as nn


def dict_apply(x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def pad_remaining_dims(x, target):
    assert x.shape == target.shape[: len(x.shape)]
    return x.reshape(x.shape + (1,) * (len(target.shape) - len(x.shape)))


def dict_apply_split(
    x: Dict[str, torch.Tensor],
    split_func: Callable[[torch.Tensor], Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    results = collections.defaultdict(dict)
    for key, value in x.items():
        result = split_func(value)
        for k, v in result.items():
            results[k][key] = v
    return results


def dict_apply_reduce(
    x: List[Dict[str, torch.Tensor]],
    reduce_func: Callable[[List[torch.Tensor]], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key in x[0].keys():
        result[key] = reduce_func([x_[key] for x_ in x])
    return result


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer


def update_optimizer(optimizer, model, param_groups: List[Dict], debug=False):
    if param_groups is None or len(param_groups) == 0:
        return optimizer
    for module in param_groups:
        prefix = module["name"]
        special_params = [p for n,p in model.named_parameters() if (n == prefix or n.startswith(prefix + "."))]
        assert len(special_params) > 0, f"no parameters found for prefix {prefix}"
        default_group = optimizer.param_groups[0]
        filtered = []
        for p in default_group["params"]:
            if not any(p is sp for sp in special_params):
                filtered.append(p)
        default_group["params"] = filtered
        lr, weight_decay = module["lr"], module["weight_decay"]
        optimizer.add_param_group({
            "params": special_params,
            "lr": lr,
            "weight_decay": weight_decay,
            "name": prefix
        })

    default_group = optimizer.param_groups[0]
    default_group["name"] = "default"
    if debug:
        dump_path = "./tmp/debug/optimizer_params.txt"
        param_to_name = {p: n for n, p in model.named_parameters()}
        with open(dump_path, "w") as f:
            for pg in optimizer.param_groups:
                f.write(f"Group `{pg['name']}`:\n")
                for p in pg["params"]:
                    name = param_to_name.get(p, "<unknown>")
                    f.write(f"  {name}\n")
                f.write("\n")

    return optimizer
