"""Phi chat-format support, kept separate from the legacy Qwen/Llama masks."""

import json
from pathlib import Path

import torch


PHI_ROLES = ("<|system|>", "<|user|>", "<|assistant|>")
PHI_SPECIAL_TOKENS = (*PHI_ROLES, "<|end|>", "<|endoftext|>")


def is_phi_model_path(path):
    """Recognize the supported checkpoint or a locally saved Phi3 configuration."""
    name = str(path).lower()
    if "phi-4-mini-instruct" in name:
        return True
    config = Path(path) / "config.json"
    if config.is_file():
        return json.loads(config.read_text()).get("model_type") == "phi3"
    return False


def is_phi_tokenizer(tokenizer):
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if not callable(get_added_vocab):
        return False
    vocab = get_added_vocab()
    return all(token in vocab for token in PHI_SPECIAL_TOKENS)


def phi_response_spans(input_ids, tokenizer, attention_mask=None):
    """Return inclusive assistant-content/end-token spans in unshifted coordinates.

    Role headers are prompt tokens, not generated content. Attention masks keep
    real EOS tokens distinct from padding (both use <|endoftext|> in Phi).
    """
    vocab = tokenizer.get_added_vocab()
    role_ids = [vocab[token] for token in PHI_ROLES]
    assistant_id, end_id = vocab["<|assistant|>"], vocab["<|end|>"]
    if attention_mask is None:
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
    spans = []
    for row, valid in zip(input_ids, attention_mask.bool()):
        boundaries = ((row == role_ids[0]) | (row == role_ids[1]) | (row == role_ids[2])) & valid
        starts = boundaries.nonzero(as_tuple=True)[0].tolist()
        valid_positions = valid.nonzero(as_tuple=True)[0]
        limit = int(valid_positions[-1]) + 1 if len(valid_positions) else 0
        row_spans = []
        for index, start in enumerate(starts):
            if row[start].item() != assistant_id:
                continue
            stop = starts[index + 1] if index + 1 < len(starts) else limit
            endings = ((row[start + 1:stop] == end_id) |
                       (row[start + 1:stop] == tokenizer.eos_token_id)).nonzero(as_tuple=True)[0]
            end = start + 1 + int(endings[0]) if len(endings) else stop - 1
            if end > start:
                row_spans.append((start + 1, end))
        spans.append(row_spans)
    return spans


def phi_masks(input_ids, tokenizer, attention_mask=None, mode="full", enable_response_mask=True):
    """Masks align with the predicted tokens, input_ids[:, 1:]."""
    if attention_mask is None:
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
    response = torch.zeros_like(input_ids, dtype=torch.float32)
    for row_index, spans in enumerate(phi_response_spans(input_ids, tokenizer, attention_mask)):
        for start, end in (spans[-1:] if mode in ("single_turn", "limited_multi_turn") else spans):
            response[row_index, start:end + 1] = 1
    loss = response.clone()
    if mode == "full" and not enable_response_mask:
        user_id = tokenizer.get_added_vocab()["<|user|>"]
        for i, row in enumerate(input_ids):
            users = ((row == user_id) & attention_mask[i].bool()).nonzero(as_tuple=True)[0]
            if len(users):
                loss[i, int(users[0]):] = 1
    loss *= attention_mask
    response *= attention_mask
    return loss[:, 1:], response[:, 1:]


def phi_masks_and_scores(input_ids, tokenizer, all_scores, use_turn_scores=False,
                         enable_response_mask=False, attention_mask=None):
    loss, response = phi_masks(input_ids, tokenizer, attention_mask,
                               enable_response_mask=enable_response_mask)
    scores = torch.zeros_like(loss)
    spans = phi_response_spans(input_ids, tokenizer, attention_mask)
    if len(all_scores) != len(spans):
        raise ValueError("One reward sequence is required per Phi episode")
    for i, (turns, rewards) in enumerate(zip(spans, all_scores)):
        if not turns:
            raise ValueError("Phi training episode has no assistant response")
        if use_turn_scores:
            if len(rewards) < len(turns):
                raise ValueError("Missing per-turn rewards for Phi responses")
            for (_, end), reward in zip(turns, rewards[-len(turns):]):
                scores[i, end - 1] = reward
        else:
            scores[i, turns[-1][1] - 1] = sum(rewards)
    return scores, loss, response


def is_response_end(tokenizer, token_id):
    from ragen.llm_agent.gemma import is_gemma_tokenizer, gemma_end_ids
    if is_gemma_tokenizer(tokenizer):
        return token_id in gemma_end_ids(tokenizer)
    if is_phi_tokenizer(tokenizer):
        return token_id in (tokenizer.eos_token_id, tokenizer.get_added_vocab()["<|end|>"])
    return token_id == tokenizer.eos_token_id
