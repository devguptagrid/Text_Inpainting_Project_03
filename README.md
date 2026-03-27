# 44. Text Inpainting (Span Fill)

# Demo on Hugging Face
[https://huggingface.co/spaces/Devanshu2402/text-inpainting-diffusion](https://huggingface.co/spaces/Devanshu2402/text-inpainting-diffusion)
# 📘 Task 1: Batched Inpainting with Parallel Mask Conditioning

## 🎯 Objective

This task focuses on improving the efficiency and scalability of diffusion-based text inpainting by introducing **parallel mask conditioning**, **curriculum learning**, and **batch-level optimization**.

---

## 🧩 1. Parallel Mask Conditioning (Batch-Level Support)

The conditioning mechanism was redesigned to support **multiple masking patterns within a single batch**.

### Implementation

- Each sample in a batch is assigned a mask ratio from:
[0.1, 0.25, 0.4] 
- Enables heterogeneous masking within the same batch.

### Example Output
[0.4, 0.1, 0.25, 0.25, 0.25, 0.1, 0.25, 0.4, 0.1, 0.25, 0.25, 0.4, 0.1, 0.1, 0.25, 0.25]  
[0.25, 0.1, 0.4, 0.4, 0.25, 0.4, 0.4, 0.25, 0.1, 0.4, 0.25, 0.4, 0.1, 0.25, 0.25, 0.25]  
[0.1, 0.1, 0.1, 0.25, 0.4, 0.1, 0.4, 0.25, 0.4, 0.4, 0.25, 0.25, 0.1, 0.25, 0.25, 0.25]



### Insight

- Different mask difficulties are processed simultaneously  


---

## 🧠 2. Mask Encoder (Parallel Conditioning)

A **mask encoder** was introduced to explicitly indicate masked positions.

### Implementation
mask_embedding = Embedding(2, hidden_dim)  
embeddings = token_emb + timestep_emb + mask_emb



### Benefit

- Helps model identify masked tokens explicitly  
- Improves denoising performance  

---

## 🔗 3. Variable-Length Span Masking

Supports dynamic span masking between **1–20 tokens**.

### Debug Output
Span length: 3, Remaining: 99  
Span length: 17, Remaining: 96  
Span length: 7, Remaining: 79  
Span length: 19, Remaining: 69

## 📚 4. Mask Batch Sampler + Curriculum Learning

A **MaskBatchSampler** groups samples by difficulty and combines with epoch-wise curriculum learning.

### Curriculum Strategy

| Epoch | Mask Ratios |
|------|-------------|
| 1–2  | 0.10        |
| 3–4  | 0.10, 0.25  |
| 5–6  | 0.10, 0.25, 0.40 |

---

### Training Results

| Epoch | Mask Ratios Used | Train Loss | Train Acc | Val Loss | Val Acc |
|------|------------------|-----------|----------|----------|---------|
| 1    | 0.10             | 4.2730    | 0.3560   | 3.6811   | 0.4202  |
| 2    | 0.10             | 3.4452    | 0.4375   | 3.6229   | 0.4251  |
| 3    | 0.10, 0.25       | 3.3974    | 0.4351   | 3.5513   | 0.4333  |
| 4    | 0.10, 0.25       | 3.3171    | 0.4410   | 3.5175   | 0.4370  |
| 5    | 0.10, 0.25, 0.40 | 3.3215    | 0.4347   | 3.5135   | 0.4371  |
| 6    | 0.10, 0.25, 0.40 | 3.2411    | 0.4413   | 3.4756   | 0.4421  |



---

## ⚡ 5. Tokens/sec Benchmark

### Results

| Mask Ratio | Batch 1 | Batch 4 | Batch 8 | Batch 16 | Batch 32 |
|-----------|--------|--------|--------|---------|---------|
| 0.10      | 1005.37 | 1258.02 | 1228.35 | 967.95  | 491.73  |
| 0.25      | 849.66  | 1220.08 | 1008.75 | 901.06  | 494.60  |
| 0.40      | 896.66  | 1006.57 | 1133.80 | 942.99  | 597.95  |

### Observations

- Performance improves from Batch 1 → 4  
- Optimal throughput at Batch 4–8  
- Larger batches degrade due to memory limits  

### Key Insight
Optimal batch size ≈ 4–8

---

## ⏱️ 6. Latency Comparison (Batch vs Sequential)

### Results

Batch Time: 0.3413 sec  
Sequential Time: 67.4623 sec  
Speedup: 197.65×


---

### Interpretation

- Batch inference uses parallel computation  
- Sequential simulation recomputes full diffusion per token  

---

### Important Note

- The ~200× speedup is an upper bound  
- Sequential method is naive that recomputes the full diffusion process for each token independently(no caching)

--- 
<br>
<br>

# 📘 Task 2: Token-Level Noise Analysis — Confidence and Error Tracking

## 🎯 Objective

This task analyzes how the diffusion model behaves at the **token level across diffusion steps**, focusing on:

- Confidence evolution  
- Entropy reduction  
- Error patterns  
- Relationship between noise and prediction accuracy  

---

## 🧩 1. Tracking Token Distributions Across Steps

To analyze model behavior, the predicted token distributions were stored at each diffusion step.

### Implementation

- Two vectors were created to store model predictions at each diffusion step:
  - logits_per_step → stores raw logits
  - probs_per_step → stores probability distributions

### Insight

- Enables step-wise tracking of:
  - confidence  
  - entropy  
  - prediction stability  

---

## 📊 2. Confidence and Entropy Evolution


### Observations
*confidence_vs_step.png*  
*entropy_vs_step.png*
- Confidence rapidly increases from ~0.6 → ~0.99 within the first diffusion step
- Entropy drops sharply from ~2.1 → near 0
- Subsequent steps show minimal change  

### Interpretation

- Most uncertainty is resolved **very early**  
- Later diffusion steps contribute **minimal refinement**  

---

## ⚠️ 3. Confident Mistakes Analysis

### Results 
*confident mistake_vs_step.png*

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

### Critical Insight

The model exhibits very high confidence early in the diffusion process (~0.99),  
however ~60% of predictions are incorrect at early steps. 

### Key Conclusion

"The diffusion model does not know when it is wrong."

---

## 🔥 4. Entropy Heatmap Analysis

### Observation
*entropy_heatmap_correct.png*  
*entropy_heatmap_incorrect.png*
- Entropy drops rapidly even for incorrect predictions  
- Model converges to low-entropy (high confidence) states prematurely  

### Insight

The model collapses to confident predictions too early,  
even when those predictions are incorrect. 

---

## 📉 5. Noise vs Accuracy Analysis

### Observation
*accuracy_vs_step.png*
- Accuracy does not increase smoothly across steps  
- It remains relatively constant during intermediate steps  
- Significant improvement occurs only at the final step  

### Interpretation

The model does not progressively refine predictions.  
Instead, a major correction happens at the final diffusion step.

---

## 📊 6. Confidence Histogram Analysis

### Observation
*confidence histogram.png*
- ~68% of tokens reach 90%+ confidence within the first diffusion step  
- Remaining tokens reach high confidence shortly after  

### Key Finding

Despite rapid confidence saturation, many predictions are still incorrect.

### Insight

Confidence is NOT a reliable indicator of correctness in diffusion-based models.

---


<br>
<br>



# 📘 Task 3: Transition Probability Inspection and Markov Chain Visualization

## 🎯 Objective

This task analyzes the learned transition behavior of the diffusion model by treating it as a Markov process. The goal is to understand:

- Token-to-token transition probabilities  
- Evolution of predictions across diffusion steps  
- Stationary distribution vs unigram frequency  
- Common prediction errors (confusions)  
- Alignment with linguistic (POS) patterns  

---

## 🧩 1. Transition Matrix Inspection

### Example: Transitions for token "the"
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


### Observation

- Strong **self-loop** for frequent tokens like "the"  
- Transitions to similar tokens ("a", "an")  
- Indicates grammatical awareness  

---

### Example: Transitions for token "a"
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


### Observation

- Highly diverse transitions  
- Includes:
  - prepositions ("in", "with")  
  - articles ("the")  
  - verbs ("received")  

### Insight

"the" → stable token  
"a" → flexible token


---


## 📈 2. Transition Probability Graph 

*graph for T=0.png*  
*Graph for T=11.png*

### Early Step (T = 0)

- All probabilities ~0.02–0.06  
- No dominant token  
- Includes noisy tokens:
=,##us,.


---

### Late Step (T = 11)

- Few dominant tokens:
the, is, lobster, a

- Strong edges:
the -> 0.12
is -> 0.08

---
### Insight
Early → uncertain and noisy
Late → confident and meaningful


---

## 🔄 3. Stationary Distribution vs Unigram Frequency

### Stationary Distribution

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


---

### Unigram Distribution
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


---

### Similarity
Cosine similarity: 0.5594


---

### Insight

- Overlap in common tokens:
the, a, ,

- Additional context-specific tokens:
lobster, crab


---

### Conclusion
Model behavious= Hybrid


---

## 🔥 4. Diversity Across Diffusion Steps

### 📊 Step-wise Examples

#### Step 0 (Early / Noisy)

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


#### Step 5 (Mid)
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


#### Step 11 (Late / Converged)

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


### Observation

- Early steps → high diversity  
- Late steps → reduced diversity  

---

### Insight
Diffusion = Explore → Collapse → Converge


---

## 5. Confusion Matrix Analysis

### Top Confusions
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


---

### Insight

- Bias toward:
  - punctuation (",", ".")  
  - frequent words ("the", "a")  

- Some semantic awareness:
american -> japanese


---

### Conclusion
Model prefers high-probability tokens under uncertainty


---

## 🧠 6. POS Transition Analysis

### Results

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


---

### Insight

- Strong grammatical consistency:

NN → NN  
IN → IN  
DT → DT


- Some invalid transitions:

NN → DT  
NN → CD


---

### Conclusion
Model captures grammar partially, not strictly

<br>
<br>

# 📘 Task 4: Controlled Generation via Latent Manipulation — Span-Level Steering

## 🎯 Objective

This task introduces **controlled generation** in diffusion-based text inpainting by manipulating the sampling process without retraining. The goal is to:

- Guide generation toward desired attributes (e.g., simplicity)  
- Apply span-level constraints  
- Control refinement using soft masking  
- Evaluate trade-offs between quality and control  

---

## 🧩 1. Steering Mechanism (Logit Manipulation)

A steering mechanism was implemented by modifying token logits during reverse diffusion sampling.

### Implementation

- Logits are adjusted at each step to bias token selection  
- No retraining required  


---

## 🔧 2. Guidance Function (Heuristic Control)

A heuristic guidance function was designed to favor **simple and common tokens**.

### Description

- Reward short/common tokens  
- Penalize longer/complex tokens  
- Modify logits before sampling 

*with_guidance_1.0_20%.png*  
*with_guidance_1.5_20%.png*  
*with_guidance_2.0_20%.png*
*without_guidance_20%.png*

### Insight
The model can be steered toward desired attributes without changing weights


---

## 🔗 3. Span-Level Steering

Different masked spans are assigned different constraints.

### Implementation

- Apply reward + penalty selectively per span  
- Example:
  - Encourage simple words in specific spans  

*simple_word_reward_and_penalty.png*

### Insight
Fine-grained control at span level enables targeted generation


---

## 🧪 4. Soft Masking (Confidence-Based Control)

Instead of binary masking, continuous weights are used.


### Insight

- Masked tokens → strong refinement (1.0)  
- Unmasked tokens → weaker updates (0.2)  


---

## 📊 5. Guidance Strength vs Generation Quality

### Results
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


---

### Observations

- Increasing guidance strength:
  - increases constraint satisfaction (% short tokens)  
  - slightly reduces BLEU (quality)  

---
## 🔬 6. Comparison with Classifier-Free Guidance

Performance compared with and without mask dropout.

---

### Without Dropout

| Setting  | Train Acc | Val Acc |
|----------|----------|---------|
| Span 10% | 49.84    | 50.82   |

---

### With Dropout 0.1

| Setting  | Train Acc | Val Acc |
|----------|----------|---------|
| Span 10% | 49.80    | 50.97   |

---

### Insight

- Slight improvement in validation accuracy with dropout  
- Indicates better generalization  

---

### Conclusion
Classifier-free guidance (dropout) improves robustness slightly

<br>
<br>

# 📘 Task 5: Memory Profiling and Generation Diversity Analysis Across Mask Ratios

## 🎯 Objective

This task evaluates the **memory efficiency** and **generation diversity** of the diffusion model across different mask ratios. The goal is to:

- Profile memory usage across masking settings  
- Analyze diversity metrics across generated samples  
- Study the trade-off between diversity and accuracy  
- Identify system bottlenecks and propose optimizations  

---

## 🧩 1, 2. Memory Profiling Across Mask Ratios

Memory usage was analyzed for different mask ratios (10%, 25%, 40%, 60%).

### Results
Model Parameters: 109,525,050  
Model Size: 417.80 MB  
Initial Memory: 417.82 MB  
Peak Memory Usage: 417.92 MB  
Estimated Activation Memory: 144.00 MB

### Key Insight


Memory usage is independent of mask ratio

### Note

- BERT-based architecture → no KV cache 

---

## 📊 3. Diversity Metrics Across Mask Ratios

50 samples were generated per mask ratio and evaluated using:

- Self-BLEU (↓ lower = more diverse)  
- N-gram Entropy (↑ higher = more diverse)  
- Unique Bigrams % (↑ higher = more diverse)  

---

###  Results

| Mask Ratio | Self-BLEU ↓ | Entropy ↑ | Unique Bigrams ↑ |
|-----------|------------|----------|------------------|
| 0.10      | 0.9227     | 6.86     | 0.1467           |
| 0.25      | 0.8263     | 7.10     | 0.2095           |
| 0.40      | 0.7375     | 7.21     | 0.2473           |
| 0.60      | 0.6418     | 7.12     | 0.2598           |

---

###  Insight

- Increasing mask ratio:
  - ↓ Self-BLEU  
  - ↑ Entropy  
  - ↑ Unique bigrams  

---

### Conclusion
Higher mask ratio → higher diversity


---

## 📉 4. Diversity vs Accuracy Trade-off

###  Results

| Mask Ratio | Accuracy ↓ | Entropy ↑ | Self-BLEU ↓ |
|-----------|-----------|----------|------------|
| 0.10      | 0.2320    | 6.86     | 0.92       |
| 0.25      | 0.1922    | 7.10     | 0.83       |
| 0.40      | 0.1639    | 7.21     | 0.74       |
| 0.60      | 0.1207    | 7.12     | 0.64       |

---

### Observations
*diversity_vs_accuracy.png*
- Accuracy decreases steadily with mask ratio  
- Diversity increases (entropy ↑, BLEU ↓)  

---

### Interpretation
More tokens to generate → harder task → more randomness


---

## 🔄 5. Effect of Mask Ratio on Diversity

###  Trends
Entropy: 6.86 → 7.21 → 7.12  
Self-BLEU: 0.92 → 0.64  
Unique bigrams: 0.14 → 0.26

### Insight

- More masked tokens:
  - Less context  
  - More uncertainty  
  - More variation  


---

## ⚠️ 6. Memory Bottlenecks

### 1. Model Weights (Largest)

- ~418 MB  
- Comes from:
  - transformer layers  
  - embeddings  

---

### 2. Activations (Second Largest)

- ~144 MB  
- Depends on:
batch_size × seq_len × hidden_size × layers

---

### 3. Diffusion Steps

- T = 12  
- Each step:
- forward pass  
- stores logits/probabilities  

---

## Optimization Strategies

### 1. Mixed Precision (FP16)
- Reduces memory by ~50%  

---

### 2. Reduce Sequence Length

- Truncate long inputs  
- Reduces activation memory  

---

### 3. Gradient Checkpointing

- Stores fewer activations  
- Recomputes during backward pass  

---
