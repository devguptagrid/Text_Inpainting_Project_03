import torch

def simple_guidance(logits, tokenizer, strength=1.0):
    """
    Simple guidance: boost common/simple words of size <=4

    
    Args:
        logits: [num_masked_tokens, vocab_size]
        tokenizer: tokenizer
        strength: guidance strength

    Returns:
        modified logits
    """

    # Example: boost short tokens (simple words)
    vocab_size = logits.size(-1)

    bias = torch.zeros_like(logits)

    for token_id in range(vocab_size):
        token = tokenizer.convert_ids_to_tokens(token_id)

        # simple heuristic: short tokens = simpler words
        if len(token) <= 4:
            bias[:, token_id] += strength

    return logits + bias


def span_guidance_with_penalty(logits, tokenizer, span_ids, strength=1.0):
    """
    Span-level guidance for different spans with reward + penalty
    """

    
    bias = torch.zeros_like(logits)

    num_tokens = logits.size(0)
    vocab_size = logits.size(-1)

    for i in range(num_tokens):
        span_id = span_ids[i].item()

        # Define behavior per span
        if span_id == 1:
            reward_strength = strength * 2.0
            penalty_strength = strength * 2.0
        elif span_id == 2:
            reward_strength = strength * 0.5
            penalty_strength = 0.0   # no penalty
        else:
            reward_strength = 0.0
            penalty_strength = 0.0

        for token_id in range(vocab_size):
            token = tokenizer.convert_ids_to_tokens(token_id)

            if len(token) <= 4:
                # reward short tokens
                bias[i, token_id] += reward_strength
            else:
                # penalize long tokens
                bias[i, token_id] -= penalty_strength

    return logits + bias
    
