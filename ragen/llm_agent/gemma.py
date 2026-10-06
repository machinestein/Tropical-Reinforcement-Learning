"""Gemma 4's native turn format, isolated from the Qwen/Phi protocols."""

import json
from pathlib import Path

import torch


GEMMA_SPECIAL_TOKENS = (
    "<bos>",
    "<eos>",
    "<|turn>",
    "<turn|>",
    "<|channel>",
    "<channel|>",
    "<|think|>",
)


def is_gemma_model_path(path):
    if "gemma-4" in str(path).lower():
        return True
    config = Path(path) / "config.json"
    return config.is_file() and json.loads(config.read_text()).get("model_type") in {
        "gemma4",
        "gemma4_text",
    }


def is_gemma_tokenizer(tokenizer):
    get_vocab = getattr(tokenizer, "get_added_vocab", None)
    return callable(get_vocab) and all(
        token in get_vocab() for token in ("<|turn>", "<turn|>", "<|channel>")
    )


def gemma_end_ids(tokenizer):
    vocab = tokenizer.get_added_vocab()
    return {tokenizer.eos_token_id, vocab["<turn|>"]} | (
        {vocab["<|tool_response>"]} if "<|tool_response>" in vocab else set()
    )


def gemma_response_spans(input_ids, tokenizer, attention_mask=None):
    """Inclusive content/terminal spans; role headers and trailing newlines are excluded."""
    start_id = tokenizer.convert_tokens_to_ids("<|turn>")
    header = tokenizer.encode("<|turn>model\n", add_special_tokens=False)
    endings = gemma_end_ids(tokenizer)
    attention_mask = (
        input_ids.ne(tokenizer.pad_token_id)
        if attention_mask is None
        else attention_mask.bool()
    )
    spans = []
    for row, valid in zip(input_ids, attention_mask):
        starts = ((row == start_id) & valid).nonzero(as_tuple=True)[0].tolist()
        positions = valid.nonzero(as_tuple=True)[0]
        limit = int(positions[-1]) + 1 if len(positions) else 0
        turns = []
        for index, start in enumerate(starts):
            if row[start : start + len(header)].tolist() != header:
                continue
            first = start + len(header)
            stop = starts[index + 1] if index + 1 < len(starts) else limit
            terminal = next(
                (
                    first + i
                    for i, token in enumerate(row[first:stop].tolist())
                    if token in endings
                ),
                stop - 1,
            )
            if first <= terminal:
                turns.append((first, terminal))
        spans.append(turns)
    return spans


def gemma_masks(
    input_ids, tokenizer, attention_mask=None, mode="full", enable_response_mask=True
):
    attention_mask = (
        input_ids.ne(tokenizer.pad_token_id)
        if attention_mask is None
        else attention_mask
    )
    response = torch.zeros_like(input_ids, dtype=torch.float32)
    for i, turns in enumerate(
        gemma_response_spans(input_ids, tokenizer, attention_mask)
    ):
        for start, end in (
            turns[-1:] if mode in ("single_turn", "limited_multi_turn") else turns
        ):
            response[i, start : end + 1] = 1
    loss = response.clone()
    if mode == "full" and not enable_response_mask:
        header = tokenizer.encode("<|turn>user\n", add_special_tokens=False)
        for i, row in enumerate(input_ids):
            starts = (
                ((row == header[0]) & attention_mask[i].bool())
                .nonzero(as_tuple=True)[0]
                .tolist()
            )
            for start in starts:
                if row[start : start + len(header)].tolist() == header:
                    loss[i, start:] = 1
                    break
    return (loss * attention_mask)[:, 1:], (response * attention_mask)[:, 1:]


def gemma_masks_and_scores(
    input_ids,
    tokenizer,
    all_scores,
    use_turn_scores=False,
    enable_response_mask=False,
    attention_mask=None,
):
    loss, response = gemma_masks(
        input_ids, tokenizer, attention_mask, enable_response_mask=enable_response_mask
    )
    scores = torch.zeros_like(loss)
    spans = gemma_response_spans(input_ids, tokenizer, attention_mask)
    if len(all_scores) != len(spans):
        raise ValueError("One reward sequence is required per Gemma episode")
    for i, (turns, rewards) in enumerate(zip(spans, all_scores)):
        if not turns:
            raise ValueError("Gemma training episode has no assistant response")
        if use_turn_scores:
            if len(rewards) < len(turns):
                raise ValueError("Missing per-turn rewards for Gemma responses")
            for (_, end), reward in zip(turns, rewards[-len(turns) :]):
                scores[i, end - 1] = reward
        else:
            scores[i, turns[-1][1] - 1] = sum(rewards)
    return scores, loss, response
