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


def compute_confident_mistakes(probs_steps, confidence_steps, ground_truth_ids, mask_positions, threshold=0.9):
    """
    Identify confident mistakes

    Args:
        probs_steps: list of tensors [batch, seq_len, vocab]
        confidence_steps: list of tensors [batch, seq_len]
        ground_truth_ids: tensor [batch, seq_len]
        mask_positions: boolean tensor [batch, seq_len]
        threshold: confidence threshold

    Returns:
        mistakes_per_step: list of counts
        total_tokens_per_step: list of counts
    """

    mistakes_per_step = []
    total_tokens_per_step = []

    for step_idx in range(len(probs_steps)):

        probs = probs_steps[step_idx]
        confidence = confidence_steps[step_idx]

        # predicted tokens (argmax)
        pred_tokens = torch.argmax(probs, dim=-1)

        # only evaluate masked positions
        pred_masked = pred_tokens[mask_positions]
        gt_masked = ground_truth_ids[mask_positions]
        conf_masked = confidence[mask_positions]

        # condition: high confidence but wrong
        confident_wrong = (conf_masked > threshold) & (pred_masked != gt_masked)

        mistakes = confident_wrong.sum().item()
        total = mask_positions.sum().item()

        mistakes_per_step.append(mistakes)
        total_tokens_per_step.append(total)

    return mistakes_per_step, total_tokens_per_step


def aggregate_metrics(confidence_steps, entropy_steps, mask_positions):
    """
    Compute average confidence and entropy over masked tokens per timestep


    Args:
        confidence_steps: list of tensors [batch, seq_len]
        entropy_steps: list of tensors [batch, seq_len]
        mask_positions: tensor [batch, seq_len]

    Returns:
        avg_confidence: list of floats
        avg_entropy: list of floats
    """

    avg_confidence = []
    avg_entropy = []

    for conf, ent in zip(confidence_steps, entropy_steps):

        # select only masked positions
        conf_masked = conf[mask_positions]
        ent_masked = ent[mask_positions]

        # compute averages
        avg_confidence.append(conf_masked.mean().item())
        avg_entropy.append(ent_masked.mean().item())

    return avg_confidence, avg_entropy



def compute_confidence_histogram(confidence_steps, mask_positions, threshold=0.9):
    """
    For each token, find the first step where confidence exceeds threshold

    
    Returns:
        step_counts: dict {step: count}
    """

    num_steps = len(confidence_steps)
    batch_size, seq_len = confidence_steps[0].shape

    # initialize tracking
    first_reach = torch.full((batch_size, seq_len), -1) ##When did each token become confident?

    for step_idx, conf in enumerate(confidence_steps):

        # only masked tokens
        conf_masked = conf.clone()
        conf_masked[~mask_positions] = -1

        reached = (conf_masked > threshold) & (first_reach == -1)

        first_reach[reached] = step_idx

    # count occurrences
    step_counts = {}

    valid_positions = mask_positions.cpu()

    for step in range(num_steps):
        count = ((first_reach == step) & valid_positions).sum().item()
        step_counts[step] = count

    return step_counts
    
