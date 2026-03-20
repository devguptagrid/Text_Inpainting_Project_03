import torch
from collections import defaultdict

def compute_confusion_matrix(
    generated_ids,
    target_ids,
    mask_positions,
    tokenizer,
    top_k=50
    ):
    """
    Compute confusion counts between true and predicted tokens
    """

    confusion = defaultdict(lambda: defaultdict(int))

    for i in range(generated_ids.size(0)):  # batch

        gen = generated_ids[i]
        tgt = target_ids[i]
        mask = mask_positions[i]

        for g, t, m in zip(gen, tgt, mask):

            if m:  # only masked tokens

                pred_token = tokenizer.convert_ids_to_tokens(g.item())
                true_token = tokenizer.convert_ids_to_tokens(t.item())

                confusion[true_token][pred_token] += 1

    return confusion


def print_top_confusions(confusion, top_n=10):
    """
    Print most common confusions
    """

    pairs = []

    for true_token in confusion:
        for pred_token in confusion[true_token]:
            count = confusion[true_token][pred_token]
            if true_token != pred_token:
                pairs.append((true_token, pred_token, count))

    # sort by frequency
    pairs.sort(key=lambda x: x[2], reverse=True)

    print("\nTop Confusions:\n")
    for t, p, c in pairs[:top_n]:
        print(f"{t} → {p}: {c}")
