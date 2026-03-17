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

