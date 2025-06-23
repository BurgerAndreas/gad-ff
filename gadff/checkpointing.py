import torch
from typing import Any, List, Mapping, OrderedDict
from collections import namedtuple

class _IncompatibleKeys(
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"]),
):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()

    __str__ = __repr__
    
# torch/nn/modules/module.py
def load_state_dict(
    self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
):
    r"""Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

    If :attr:`strict` is ``True``, then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :meth:`~torch.nn.Module.state_dict` function.

    .. warning::
        If :attr:`assign` is ``True`` the optimizer must be created after
        the call to :attr:`load_state_dict` unless
        :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

    Args:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        strict (bool, optional): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
        assign (bool, optional): When ``False``, the properties of the tensors
            in the current module are preserved while when ``True``, the
            properties of the Tensors in the state dict are preserved. The only
            exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s
            for which the value from the module is preserved.
            Default: ``False``

    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing any keys that are expected
                by this module but missing from the provided ``state_dict``.
            * **unexpected_keys** is a list of str containing the keys that are not
                expected by this module but present in the provided ``state_dict``.

    Note:
        If a parameter or buffer is registered as ``None`` and its corresponding key
        exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
        ``RuntimeError``.
    """
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"Expected state_dict to be dict-like, got {type(state_dict)}."
        )

    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    error_msgs: List[str] = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = OrderedDict(state_dict)
    if metadata is not None:
        # mypy isn't aware that "_metadata" exists in state_dict
        state_dict._metadata = metadata  # type: ignore[attr-defined]

    def load(module, local_state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        if assign:
            local_metadata["assign_to_params_buffers"] = assign
        module._load_from_state_dict(
            local_state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + "."
                child_state_dict = {
                    k: v
                    for k, v in local_state_dict.items()
                    if k.startswith(child_prefix)
                }
                load(child, child_state_dict, child_prefix)  # noqa: F821

        # Note that the hook can modify missing_keys and unexpected_keys.
        incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
        for hook in module._load_state_dict_post_hooks.values():
            out = hook(module, incompatible_keys)
            assert out is None, (
                "Hooks registered with ``register_load_state_dict_post_hook`` are not"
                "expected to return new values, if incompatible_keys need to be modified,"
                "it should be done inplace."
            )

    load(self, state_dict)
    del load

    print(f"Unexpected keys: {len(unexpected_keys)} / {len(state_dict.keys())}")
    print(f"Missing keys: {len(missing_keys)} / {len(state_dict.keys())}")

    if strict:
        if len(unexpected_keys) > 0:
            error_msgs.insert(
                0,
                "Unexpected key(s) in state_dict: {}. ".format(
                    ", ".join(f'"{k}"' for k in unexpected_keys)
                ),
            )
        if len(missing_keys) > 0:
            error_msgs.insert(
                0,
                "Missing key(s) in state_dict: {}. ".format(
                    ", ".join(f'"{k}"' for k in missing_keys)
                ),
            )

    if len(error_msgs) > 0:
        raise RuntimeError(
            "Error(s) in loading state_dict for {}:\n\t{}".format(
                self.__class__.__name__, "\n\t".join(error_msgs)
            )
        )
    return _IncompatibleKeys(missing_keys, unexpected_keys)