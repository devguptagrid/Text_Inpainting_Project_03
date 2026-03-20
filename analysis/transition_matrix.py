import torch

def compute_transition_matrix(
    probs_steps,
    target_ids,
    mask_positions,
    vocab_size
):
    """
    Approximate transition matrix T[i][j]
    """

    
    T = torch.zeros((vocab_size, vocab_size))

    # use final timestep (most stable)
    probs = probs_steps[-1]  # [batch, seq_len, vocab]

    for b in range(probs.size(0)):
        for i in range(probs.size(1)):

            if mask_positions[b, i]:

                true_token = target_ids[b, i].item()
                pred_probs = probs[b, i]  # [vocab]

                T[true_token] += pred_probs.cpu()

    # normalize rows
    row_sums = T.sum(dim=1, keepdim=True) + 1e-8
    T = T / row_sums

    return T
    

def print_transition_row(T, tokenizer, token_id, top_k=10):
    probs = T[token_id]

    values, indices = probs.topk(top_k)

    print(f"\nTransitions for token: {tokenizer.convert_ids_to_tokens(token_id)}")

    for v, i in zip(values, indices):
        print(f"{tokenizer.convert_ids_to_tokens(i.item())}: {v.item():.4f}")
    

