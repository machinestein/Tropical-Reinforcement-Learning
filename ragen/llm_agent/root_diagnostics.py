"""Reasoning diagnostics from exact first-response tokens, before any filtering."""

import numpy as np
from verl import DataProto


def make_root_diagnostics(inputs, outputs, tokenizer):
    if outputs.batch is None or 'responses' not in outputs.batch:
        return None
    closing = tuple(tokenizer.encode('</think>', add_special_tokens=False))
    prompts, reasoning = [], []
    for row, response in enumerate(outputs.batch['responses']):
        prompt = inputs.batch['input_ids'][row][inputs.batch['attention_mask'][row].bool()].tolist()
        mask = outputs.batch['attention_mask'][row, -len(response):].bool()
        tokens = tuple(response[mask].tolist())
        end = next((i for i in range(len(tokens) - len(closing) + 1)
                    if tokens[i:i + len(closing)] == closing), 0)
        prompts.append(prompt)
        reasoning.append(list(tokens[:end]))
    def objects(rows):
        result = np.empty(len(rows), dtype=object)
        result[:] = rows
        return result
    return DataProto.from_dict(tensors={
        'input_ids': inputs.batch['input_ids'].clone(),
        'attention_mask': inputs.batch['attention_mask'].clone(),
    }, non_tensors={
        'group_ids': inputs.non_tensor_batch['group_ids'].copy(),
        'first_turn_prompt_ids': objects(prompts),
        'first_turn_reasoning_ids': objects(reasoning),
    }, meta_info={'root_only_diagnostics': True})
