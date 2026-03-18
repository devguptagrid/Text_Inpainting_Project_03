def compute_short_token_percentage(generated_ids, mask_positions, tokenizer, max_len=4):
    """
    Compute % of generated tokens that are short (len <= max_len)
    """

    total = 0
    short_count = 0

    tokens = tokenizer.convert_ids_to_tokens(generated_ids)

    for token, is_mask in zip(tokens, mask_positions):
        if is_mask:
            total += 1
            if len(token) <= max_len:
                short_count += 1

    return short_count / total if total > 0 else 0

