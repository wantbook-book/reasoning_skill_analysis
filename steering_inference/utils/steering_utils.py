import torch
import contextlib
import functools

from typing import List, Tuple, Callable, Optional
from torch import Tensor

from sae_lens import SAE

class GlobalSteering:
    enable = True

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

def get_steering_hook(
    alpha: float = 1.0,
    sae: SAE = None,
    feature_idx: int = None,
    c_m: float = None,
    hs_vec: Tensor = None,
):
    def hook_fn(module, input, output, tokens_to_skip: int = 0):
        nonlocal alpha, sae, feature_idx, c_m, hs_vec
        if not GlobalSteering.enable:
            return output

        if torch.is_tensor(output):
            activations = output.clone()
        else:
            activations = output[0].clone()

        if torch.is_tensor(input):
            pos_ids = input.clone()
        else:
            pos_ids = input[0].clone()
        assert pos_ids.dtype == torch.int64, "Input position ids must be of type int64"
        assert input[0].shape[0] == activations.shape[0], "Batch size mismatch between input tokens and activations"
        mask = pos_ids >= tokens_to_skip

        if sae is not None:
            if sae.device != activations.device:
                sae.device = activations.device
                sae.to(sae.device)
            sae_acts = sae.encode(activations)
            reconstructed = sae.decode(sae_acts)
            error = activations.to(sae_acts.dtype) - reconstructed
            if mask is not None:
                sae_acts[..., feature_idx] += c_m * alpha * mask.squeeze(-1)
            else:
                sae_acts[..., feature_idx] += c_m * alpha
            sae_acts[..., feature_idx] = torch.clamp(sae_acts[..., feature_idx], min=0)
            activations_hat = sae.decode(sae_acts) + error
            activations_hat = activations_hat.type_as(activations)
            
        elif hs_vec is not None:
            hs_vec_local = hs_vec
            if hs_vec_local.device != activations.device or hs_vec_local.dtype != activations.dtype:
                hs_vec_local = hs_vec_local.to(device=activations.device, dtype=activations.dtype)
            if mask is not None:
                activations_hat = activations + mask.unsqueeze(-1) * (hs_vec_local * c_m * alpha)
            else:
                activations_hat = activations + hs_vec_local * c_m * alpha
        else:
            print("no steering method") 
            
        if torch.is_tensor(output):
            return activations_hat
        else:
            return (activations_hat,) + output[1:] if len(output) > 1 else (activations_hat,)

    return hook_fn


def get_multi_steering_hook_from_lists(
    feature_idxs: List[Optional[int]],
    c_ms: List[Optional[float]],
    hs_vecs: List[Optional[Tensor]],
    alphas: List[float],
    saes: List[Optional[SAE]],
):
    """
    Combine multiple steering operations that reside on the same layer into a single hook.
    The lists must all be the same length; each index represents one steering config.
    """

    lengths = {len(feature_idxs), len(c_ms), len(hs_vecs), len(alphas), len(saes)}
    if len(lengths) != 1:
        raise ValueError("All steering configuration lists must be the same length.")
    total = lengths.pop()
    if total == 0:
        raise ValueError("At least one steering configuration is required.")

    def combined_hook(module, input, output, tokens_to_skip: int = 0):
        if not GlobalSteering.enable:
            return output

        output_is_tensor = torch.is_tensor(output)
        activations = output.clone() if output_is_tensor else output[0].clone()

        if torch.is_tensor(input):
            pos_ids = input.clone()
        else:
            pos_ids = input[0].clone()
        assert pos_ids.dtype == torch.int64, "Input position ids must be of type int64"
        assert input[0].shape[0] == activations.shape[0], "Batch size mismatch between input tokens and activations"
        mask = pos_ids >= tokens_to_skip

        current_activations = activations

        # Group SAE steering by SAE instance to encode/decode once per SAE.
        sae_groups: dict[int, List[int]] = {}
        for idx, sae in enumerate(saes):
            if sae is not None:
                sae_groups.setdefault(id(sae), []).append(idx)

        for indices in sae_groups.values():
            sae = saes[indices[0]]
            if sae.device != current_activations.device:
                sae.device = current_activations.device
                sae.to(sae.device)
            sae_acts = sae.encode(current_activations)
            reconstructed = sae.decode(sae_acts)
            error = current_activations.to(sae_acts.dtype) - reconstructed
            mask_feature = mask.squeeze(-1) if mask is not None else None
            for idx in indices:
                feature_idx = feature_idxs[idx]
                c_m = c_ms[idx]
                alpha = alphas[idx]
                if feature_idx is None:
                    raise ValueError("SAE steering requires feature_idx.")
                if c_m is None:
                    raise ValueError("SAE steering requires c_m.")
                if mask_feature is not None:
                    sae_acts[..., feature_idx] += c_m * alpha * mask_feature
                else:
                    sae_acts[..., feature_idx] += c_m * alpha
                sae_acts[..., feature_idx] = torch.clamp(sae_acts[..., feature_idx], min=0)
            current_activations = sae.decode(sae_acts) + error
            current_activations = current_activations.type_as(activations)

        # Sum HS steering contributions before applying once.
        hs_delta = None
        for idx, hs_vec in enumerate(hs_vecs):
            if hs_vec is None:
                continue
            c_m = c_ms[idx]
            alpha = alphas[idx]
            if c_m is None:
                raise ValueError("HS steering requires c_m.")
            hs_vec_local = hs_vec
            if hs_vec_local.device != current_activations.device or hs_vec_local.dtype != current_activations.dtype:
                hs_vec_local = hs_vec_local.to(device=current_activations.device, dtype=current_activations.dtype)
            delta = hs_vec_local * c_m * alpha
            if mask is not None:
                delta = mask.unsqueeze(-1) * delta
            hs_delta = delta if hs_delta is None else hs_delta + delta

        if hs_delta is not None:
            current_activations = current_activations + hs_delta

        if output_is_tensor:
            return current_activations
        else:
            return (current_activations,) + output[1:] if len(output) > 1 else (current_activations,)

    return combined_hook
