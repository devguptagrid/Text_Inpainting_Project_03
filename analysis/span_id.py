import torch
def get_span_ids(mask_positions):
    """
    Assign span IDs to masked positions
    """
    span_ids = torch.zeros_like(mask_positions, dtype=torch.long)


    current_span = 0
    prev = False

    for i in range(mask_positions.size(1)):
        if mask_positions[0, i]:
            if not prev:
                current_span += 1
            span_ids[0, i] = current_span
            prev = True
        else:
            prev = False

    return span_ids

