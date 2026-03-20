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


def compute_stationary_distribution(probs_steps, mask_positions):
    """
    Approximate stationary distribution using final timestep
    """

    
    # take last timestep (most stable)
    final_probs = probs_steps[-1][0]  # [seq_len, vocab]

    # select masked tokens
    mask = mask_positions[0].cpu()
    masked_probs = final_probs[mask]

    # average across masked positions
    stationary = masked_probs.mean(dim=0)

    return stationary
    
def print_top_stationary_tokens(stationary, tokenizer, top_k=10):
    values, indices = stationary.topk(top_k)

    print("\nTop Stationary Tokens:\n")
    for v, i in zip(values, indices):
        token = tokenizer.convert_ids_to_tokens(i.item())
        print(f"{token}: {v.item():.4f}")
    


from collections import Counter
import torch

def compute_unigram_distribution(dataset, tokenizer, vocab_size):
    """
    Compute unigram frequency distribution over dataset
    """
    counter = Counter()

    for sample in dataset:
        input_ids = sample["target_ids"]  # or input_ids
        for token_id in input_ids:
            counter[token_id.item()] += 1

    # convert to tensor
    unigram = torch.zeros(vocab_size)

    total = sum(counter.values())

    for token_id, count in counter.items():
        unigram[token_id] = count / total

    return unigram


def compare_stationary_unigram(stationary, unigram, tokenizer, top_k=10):
    """
    Compare top tokens from both distributions
    """

    stat_vals, stat_idx = stationary.topk(top_k)
    uni_vals, uni_idx = unigram.topk(top_k)

    print("\nTop Stationary Tokens:")
    for v, i in zip(stat_vals, stat_idx):
        print(f"{tokenizer.convert_ids_to_tokens(i.item())}: {v.item():.4f}")

    print("\nTop Unigram Tokens:")
    for v, i in zip(uni_vals, uni_idx):
        print(f"{tokenizer.convert_ids_to_tokens(i.item())}: {v.item():.4f}")
    
