"""A scalar value head on Gemma 4's unchanged native text backbone."""

from torch import nn
from transformers import AutoModelForTokenClassification, Gemma4TextConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4PreTrainedModel,
    Gemma4TextModel,
)


class Gemma4ForTokenClassification(Gemma4PreTrainedModel):
    config_class = Gemma4TextConfig
    base_model_prefix = "model"
    _no_split_modules = ["Gemma4TextDecoderLayer"]

    @classmethod
    def _can_set_experts_implementation(cls):
        return Gemma4TextModel._can_set_experts_implementation()

    def __init__(self, config):
        super().__init__(config)
        self.model = Gemma4TextModel(config)
        self._keys_to_ignore_on_load_unexpected = [
            f"model.{name}" for name in self.model._keys_to_ignore_on_load_unexpected
        ]
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        return TokenClassifierOutput(
            logits=self.score(outputs.last_hidden_state),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def register_critic():
    AutoModelForTokenClassification.register(
        Gemma4TextConfig, Gemma4ForTokenClassification, exist_ok=True
    )
