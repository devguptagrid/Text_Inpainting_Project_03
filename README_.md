## task 2

1. Two vectors created to store model prected token probability distribution per step
    logits_per_step = [] 
    probs_per_step = []


2. Observation:

confidence_vs_step.png - Confidence rapidly increases from ~0.6 to ~0.99 within the first diffusion step.

entropy_vs_step.png - Entropy correspondingly drops from ~2.1 to near zero.

Subsequent steps show minimal change.

Confident Mistakes per Step:
Step 0: 0/25 (rate=0.0000)
Step 1: 16/25 (rate=0.6400)
Step 2: 15/25 (rate=0.6000)
Step 3: 15/25 (rate=0.6000)
Step 4: 16/25 (rate=0.6400)
Step 5: 16/25 (rate=0.6400)
Step 6: 15/25 (rate=0.6000)
Step 7: 15/25 (rate=0.6000)
Step 8: 15/25 (rate=0.6000)
Step 9: 15/25 (rate=0.6000)
Step 10: 15/25 (rate=0.6000)
Step 11: 16/25 (rate=0.6400)

3. Observation from above table or confident mistake_vs_step.png

The model exhibits high confidence early in the diffusion process (confidence ≈ 0.99)

However, this confidence is misleading, as ~60% of predictions are incorrect at early steps

Critical Insight

“The diffusion model does not know when it is wrong.”

6. Confidence Histogram Analysis

The confidence histogram shows that approximately 68% of tokens reach a confidence level above 90% within the first diffusion step, while the remaining tokens achieve this threshold at the initial step. Despite this rapid confidence saturation, earlier analysis indicates that a large proportion of these predictions are incorrect at early steps. This demonstrates that confidence is not a reliable indicator of correctness in diffusion-based text generation models.

4. “The model collapses to low-entropy states prematurely, even when predictions are incorrect.” - entropy_correct_vs_incorrect.png/ entropy_heatmap.png


5. Noise vs Accuracy Analysis

The accuracy does not increase smoothly across diffusion steps. Instead, it remains relatively constant during intermediate steps and improves only at the final step. This indicates that the model does not progressively refine predictions but instead performs a significant correction at the final stage of the diffusion process.- accuracy_vs_step.png

## task 4

1. We implement a steering mechanism by modifying token logits during reverse diffusion sampling. This allows us to guide generation toward desired attributes without retraining the model. - with_guidance.png

2. "We implemented a heuristic guidance mechanism that biases the diffusion process toward simpler and more common tokens by modifying logits during reverse diffusion." - with_guidance.png

4. 
mask_weights = torch.ones_like(input_ids,dtype=torch.float).to(device)
    mask_weights[mask_positions] = 1.0
    mask_weights[~mask_positions] = 0.2


5. 
--- Guidance Strength: 0.5 ---
BLEU: 0.0826
% Short Tokens: 0.8359

--- Guidance Strength: 1.0 ---
BLEU: 0.0845
% Short Tokens: 0.8760

--- Guidance Strength: 1.5 ---
BLEU: 0.0799
% Short Tokens: 0.9106

--- Guidance Strength: 2.0 ---
BLEU: 0.0804
% Short Tokens: 0.9296

As guidance strength increases, the proportion of short tokens(constraint satisfaction) increases significantly, demonstrating effective control over generation.
BLEU is low overall (~0.08)
So: small changes don’t affect it much


3.  simple_word_reward_and_penalty.png
We implement span-level steering by assigning different constraints to each masked span. For spans requiring simpler language, we apply both a reward to short tokens and a penalty to longer tokens during sampling by modifying logits. This enables fine-grained control over generated content.


6. 
| Setting  | Train Acc | Val Acc |
| -------- | --------- | ------- | without dropout
| Span 10% | 49.84     | 50.82   |


| Setting  | Train Acc | Val Acc |
| -------- | --------- | ------- | with dropout
| Span 10% | 49.80     | 50.97   |



## task 3


Step 0:
,: 0.0618
is: 0.0422
##us: 0.0380
##rus: 0.0374
the: 0.0312
lobster: 0.0286
.: 0.0279
in: 0.0222
known: 0.0207
=: 0.0202

Step 1:
the: 0.1193
##us: 0.0975
is: 0.0785
,: 0.0615
a: 0.0527
length: 0.0393
lobster: 0.0392
or: 0.0392
it: 0.0376
in: 0.0203

Step 5:
the: 0.1176
##us: 0.0978
is: 0.0788
,: 0.0589
a: 0.0562
lobster: 0.0392
length: 0.0392
or: 0.0391
it: 0.0285
of: 0.0202

Step 6:
the: 0.1175
##us: 0.0971
is: 0.0783
,: 0.0588
a: 0.0439
lobster: 0.0392
length: 0.0392
or: 0.0391
##rus: 0.0206
in: 0.0196

Step 10:
the: 0.1172
##us: 0.0784
is: 0.0783
,: 0.0588
a: 0.0395
lobster: 0.0393
length: 0.0392
##rus: 0.0392
or: 0.0388
crab: 0.0198

Step 11:
the: 0.1175
##us: 0.0784
is: 0.0783
,: 0.0588
a: 0.0392
lobster: 0.0392
length: 0.0392
##rus: 0.0392
or: 0.0390
crab: 0.0197

4. The diffusion process exhibits a clear diversity pattern: high diversity at early steps, followed by gradual reduction as the model converges.

5. 
Top Confusions:

the → ,: 6
and → ,: 4
is → the: 4
sisters → was: 4
on → in: 4
american → japanese: 4
that → ,: 4
the → .: 3
the → she: 3
the → a: 3

The confusion matrix reveals that the diffusion model frequently substitutes high-frequency tokens such as punctuation (e.g., “the → ,”, “and → ,”) and common function words (e.g., “the → a”, “is → the”), indicating a bias toward simpler and more probable tokens under uncertainty. At the same time, the model demonstrates partial semantic understanding, as seen in substitutions like “american → japanese,” suggesting it captures broad semantic categories.

2. graph for T=0 and T=11.png

T = 11 (late)
Strong edges:
the → 0.12
is → 0.08
Few dominant tokens
👉 peaked distribution

🔴 T = 0 (early)
All edges ~0.02–0.06
No clear winner
👉 flat distribution

late step → confident  
early step → 

🟢 T = 11
meaningful tokens:
the, is, lobster, a

🔴 T = 0
noisy tokens:
= , ##us, .

The transition graphs show that at early diffusion steps (t=0), the model produces a flat and uncertain distribution over many tokens, including punctuation and subword fragments, indicating high noise. As the process progresses to later steps (t=11), the distribution becomes more concentrated, with higher probabilities assigned to a few meaningful tokens such as “the” and “is.”

3. 
Top Stationary Tokens:
the: 0.1175
##us: 0.0784
is: 0.0783
,: 0.0588
a: 0.0392
lobster: 0.0392
length: 0.0392
##rus: 0.0392
or: 0.0390
crab: 0.0197

Top Unigram Tokens:
the: 0.0605
,: 0.0430
.: 0.0368
of: 0.0243
and: 0.0220
@: 0.0207
in: 0.0197
to: 0.0171
a: 0.0152
=: 0.0119

Cosine Similarity: 0.5594

model balances:
- frequency bias
- contextual understanding

not purely frequency-based  
not purely context-based  
→ hybrid behavior

The stationary distribution shows partial alignment with the unigram token frequency distribution, with common tokens such as “the” and “a” appearing in both. However, the stationary distribution also includes context-specific tokens such as “lobster” and “crab,” which are absent from the unigram distribution. The cosine similarity of 0.56 indicates moderate correlation, suggesting that the diffusion model is influenced by token frequency but is not solely driven by it.

1. 
Transitions for token: the
the: 0.4289
an: 0.0957
,: 0.0952
to: 0.0476
@: 0.0476
armored: 0.0476
turret: 0.0476
3rd: 0.0470
two: 0.0469
a: 0.0468

The transition matrix shows a strong self-loop for frequent tokens such as “the”, indicating that the diffusion process tends to preserve tokens during denoising. At the same time, the model allows transitions to grammatically similar tokens such as “a” and “an”, demonstrating an understanding of linguistic equivalence.

Transitions for token: a
in: 0.4007
with: 0.2000
the: 0.1979
received: 0.1903
suffered: 0.0091
of: 0.0003
one: 0.0002
two: 0.0001
got: 0.0001
sustained: 0.0001

The transition matrix for the token “a” shows highly diverse transitions, with strong probabilities toward prepositions such as “in” and “with”, as well as substitutions to other articles like “the”. Unlike “the”, which exhibits a strong self-loop, “a” demonstrates less stability and frequently transitions to different syntactic categories, including verbs such as “received”.

6. 

Top POS Transitions:

NN → NN: 81
NN → IN: 20
IN → IN: 20
NN → DT: 17
DT → DT: 15
CD → NN: 10
. → .: 10
NN → CD: 9
NN → JJ: 8
NN → NNS: 7

The POS transition analysis reveals that the model partially captures linguistic structure, with strong consistency in noun-to-noun (NN → NN) and preposition-to-preposition (IN → IN) transitions, indicating preservation of semantic categories. The presence of determiner-to-determiner (DT → DT) transitions further suggests that the model learns common grammatical substitutions such as “the” and “a”. However, the model also produces invalid transitions such as noun-to-determiner (NN → DT) and noun-to-number (NN → CD), demonstrating that it does not strictly enforce grammatical constraints.


## task 5
1. 
2. 
Model Parameters: 109,525,050
Model Size: 417.80 MB

Initial Memory: 417.82 MB

Peak Memory Usage: 417.92 MB
Estimated Activation Memory: 144.00 MB

The model occupies approximately 417.8 MB of memory, corresponding to its 109 million parameters. While MPS-reported peak memory usage remains close to the model size (~417.9 MB), this underestimates actual runtime memory due to backend limitations. Therefore, activation memory is estimated analytically and found to be approximately 144 MB, reflecting the memory required for intermediate representations during forward passes. Since the architecture is based on BERT, no KV cache is used, and memory usage is primarily dominated by activations.
