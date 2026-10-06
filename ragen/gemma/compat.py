"""Compatibility loaded only by the separate Gemma environment's .pth file."""


def activate():
    import transformers

    if int(transformers.__version__.split(".")[0]) < 5:
        raise RuntimeError(
            "Gemma 4 requires the separate gemma_venv (Transformers 5.6.0)"
        )
    if not hasattr(transformers, "AutoModelForVision2Seq"):
        transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText
    transformers.Gemma4ForCausalLM._no_split_modules = ["Gemma4TextDecoderLayer"]
    transformers.Gemma4TextModel._no_split_modules = ["Gemma4TextDecoderLayer"]
    import sys
    import vllm.lora.lora_model

    sys.modules.setdefault("vllm.lora.models", vllm.lora.lora_model)
    from ragen.gemma.model import register_critic

    register_critic()
