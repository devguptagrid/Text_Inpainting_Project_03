import torch

def extract_top_transitions(probs_steps, mask_positions, top_k=10):
    """
    Extract top-k token transitions per timestep
    """

    transitions = []

    for t, probs in enumerate(probs_steps):

        # take first sample only
        probs = probs[0]

        # select masked positions
        mask=mask_positions[0].cpu()
        probs_masked = probs[mask]

        # average across masked tokens
        avg_probs = probs_masked.mean(dim=0)

        # top-k tokens
        top_probs, top_indices = torch.topk(avg_probs, top_k)

        transitions.append({
            "timestep": t,
            "tokens": top_indices.cpu(),
            "probs": top_probs.cpu()
        })

    return transitions


def decode_tokens(token_ids, tokenizer):
    return [tokenizer.convert_ids_to_tokens(t.item()) for t in token_ids]
