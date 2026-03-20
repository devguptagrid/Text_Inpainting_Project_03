import nltk
from collections import defaultdict

def compute_pos_transitions(confusion_matrix):
    """
    Convert token transitions → POS transitions
    """
    pos_transitions = defaultdict(lambda: defaultdict(int))

    for true_token in confusion_matrix:
        for pred_token in confusion_matrix[true_token]:

            count = confusion_matrix[true_token][pred_token]

            # get POS tags
            true_pos = nltk.pos_tag([true_token])[0][1]
            pred_pos = nltk.pos_tag([pred_token])[0][1]

            pos_transitions[true_pos][pred_pos] += count

    return pos_transitions
    

def print_pos_transitions(pos_transitions, top_n=10):

    pairs = []

    for pos1 in pos_transitions:
        for pos2 in pos_transitions[pos1]:
            count = pos_transitions[pos1][pos2]
            pairs.append((pos1, pos2, count))

    pairs.sort(key=lambda x: x[2], reverse=True)

    print("\nTop POS Transitions:\n")
    for p1, p2, c in pairs[:top_n]:
        print(f"{p1} → {p2}: {c}")
    
