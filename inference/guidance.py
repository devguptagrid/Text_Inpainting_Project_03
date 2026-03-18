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

