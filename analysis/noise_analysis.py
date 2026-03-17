import torch

def compute_confidence(probs_steps):
    
    """
    Compute confidence (max probability) per token per timestep

    ```
    Args:
        probs_steps: list of tensors
            each tensor shape = [batch_size, seq_len, vocab_size]

    Returns:
        confidence_steps: list of tensors
            each tensor shape = [batch_size, seq_len]
    """

    confidence_steps = []

    for probs in probs_steps:
        # max over vocabulary dimension
        confidence = torch.max(probs, dim=-1).values

        confidence_steps.append(confidence)

    return confidence_steps



def compute_entropy(probs_steps, eps=1e-9):
    """
    Compute entropy per token per timestep

    ```
    Args:
        probs_steps: list of tensors
            each tensor shape = [batch_size, seq_len, vocab_size]

    Returns:
        entropy_steps: list of tensors
            each tensor shape = [batch_size, seq_len]
    """

    entropy_steps = []

    for probs in probs_steps:
        # add small epsilon to avoid log(0)
        probs_safe = probs + eps

        # entropy formula: -sum(p * log(p))
        entropy = -torch.sum(probs_safe * torch.log(probs_safe), dim=-1)

        entropy_steps.append(entropy)

    return entropy_steps

