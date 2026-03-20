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