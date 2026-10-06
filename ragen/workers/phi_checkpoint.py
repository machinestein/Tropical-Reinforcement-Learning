"""Checkpoint compatibility for native Transformers Phi models."""

import logging


def prepare_native_phi_checkpoint(model):
    """Remove stale Hub code mappings only when the loaded Phi class is native.

    Phi-4-mini's Hub config advertises remote modeling files even when loaded
    with trust_remote_code=False. VERL interprets the mere presence of auto_map
    as a custom model and tries to copy Transformers' package-relative imports
    into the checkpoint, failing on paths such as ``phi3/..utils.py``.
    Native Phi configs reload through Transformers' built-in model registry;
    they do not need these mappings. Qwen and actual custom classes are untouched.
    """
    model = getattr(model, "_fsdp_wrapped_module", model)
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "phi3" or not hasattr(config, "auto_map"):
        return False

    from torch.distributed.fsdp import FSDPModule

    model_class = type(model)
    if isinstance(model, FSDPModule):
        model_class = model_class.__bases__[1]
    if (
        model_class.__module__ != "transformers.models.phi3.modeling_phi3"
        or model_class.__name__ not in {"Phi3ForCausalLM", "Phi3ForTokenClassification"}
        or type(config).__module__ != "transformers.models.phi3.configuration_phi3"
    ):
        return False

    del config.auto_map
    logging.getLogger(__name__).info("Removed stale remote-code mappings for native %s checkpoints", model_class.__name__)
    return True
