Observation:

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

Observation from above table or confident mistake_vs_step.png

The model exhibits high confidence early in the diffusion process (confidence ≈ 0.99)

However, this confidence is misleading, as ~60% of predictions are incorrect at early steps

Critical Insight

“The diffusion model does not know when it is wrong.”