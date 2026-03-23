import math
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu

def compute_self_bleu(generations, tokenizer):
    texts = [
    tokenizer.convert_ids_to_tokens(seq.tolist())
    for seq in generations
    ]

    
    scores = []

    for i in range(len(texts)):
        reference = texts[:i] + texts[i+1:]
        hypothesis = texts[i]

        if len(reference) == 0:
            continue

        score = sentence_bleu(reference, hypothesis)
        scores.append(score)

    return sum(scores) / len(scores)
    

def compute_ngram_entropy(generations, n=2):
    counter = Counter()
    total = 0

    
    for seq in generations:
        tokens = seq.tolist()

        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            counter[ngram] += 1
            total += 1

    entropy = 0.0

    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p + 1e-8)

    return entropy
    

def compute_unique_bigrams(generations):
    bigrams = set()
    total = 0

    
    for seq in generations:
        tokens = seq.tolist()

        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i+1])
            bigrams.add(bigram)
            total += 1

    return len(bigrams) / total if total > 0 else 0
    
