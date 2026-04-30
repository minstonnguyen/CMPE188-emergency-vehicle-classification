# Police Vehicle Detection — Architecture Options

**Task:** Classify vehicle crop → civilian / overt LE / covert LE  
**Constraint:** Max 2 models in pipeline, end-to-end learned

---

## 1. CNN Binary Classifier
ResNet-50 / EfficientNetV2 fine-tuned with CE loss.

- **Pro:** Simple, fast (~5ms), well-understood
- **Con:** Pooling destroys spatial relationships — cannot reason about sparse co-occurring cues (antenna + tint + plate across the image). Learns overt signals only. Fails on covert.

---

## 2. Vehicle Re-ID Embedding + Cosine Similarity
Metric learning backbone (ArcFace / SupCon) producing 512-dim normalized embeddings. Classification = cosine distance to class prototype centroids.

- **Pro:** Generalizes to unseen LE models without retraining. Open-set capable (low confidence = uncertain). Make/model identity is a byproduct.
- **Con:** Generic embeddings cluster by make/model first, LE status second — covert vehicles sit dangerously close to their civilian counterparts. Requires careful margin tuning and balanced batching to carve out covert separation.

---

## 3. Masked Autoencoder (MAE) + Classification Head
Single ViT-B, MAE-pretrained. Encoder + decoder reconstruct randomly masked patches. CLS token feeds a 3-class head. Fine-tuned jointly end-to-end.

Civilian vehicles reconstruct well (abundant in training). LE vehicles produce **high per-patch reconstruction loss in anomalous regions** — antenna mounts, tinted glass, specialty plates. The loss map is a free spatial evidence heatmap. Overt vehicles flag many patches; covert vehicles flag a few specific subtle ones.

- **Pro:** Covert detection emerges from the reconstruction objective — no covert-specific labels needed, just a strong civilian prior. Generalizes to unseen covert configs. Anomaly heatmap = built-in explainability.
- **Con:** Sensitive to masking ratio and data balance (LE-skewed training degrades the civilian prior). Decoder adds inference overhead (can be dropped post-training at cost of explainability).

---

## 4. ViT with Register Tokens
DINOv2 ViT-B/14 with 4 learnable register tokens prepended to the patch sequence. Pool [CLS + R1–R4] → Linear → 3 classes. Fine-tuned end-to-end.

Register tokens absorb global reasoning and emergently specialize during fine-tuning — some accumulate make/model priors, others accumulate sparse covert cue evidence. Cross-attention lets the model reason about cue co-occurrence across the full image without any engineered branches.

- **Pro:** Lowest implementation complexity of the ViT options. Truly one model, one loss, one forward pass. DINOv2's pretraining already distinguishes windshield / roofline / door panel in zero-shot.
- **Con:** Register specialization is emergent — not guaranteed. Purely discriminative; won't generalize to covert configs absent from training data.

---

## 5. Hyperspherical Prototype Network (ArcFace ViT)
DINOv2 ViT-B fine-tuned with ArcFace loss (angular margin on unit hypersphere). Classification = nearest prototype centroid.

Angular margin forces separation between visually-identical classes (civilian Charger vs. Charger Pursuit) by maximizing the angle between their embeddings. The model must find and encode whatever signal causes that separation — and since covert cues are the only available signal, it discovers them end-to-end.

- **Pro:** Best long-term generalization — learns the *geometry of deviation from civilian*, not specific appearances. New LE models added via prototype insertion, no retraining.
- **Con:** Margin hyperparameters need tuning. No spatial explainability.

---

## Recommended Pipelines

| Goal | Pipeline |
|---|---|
| Simplest single model | Option 4 — DINOv2 + register tokens |
| Best covert generalization, sparse labels | Option 3 — MAE + classification head |
| Best overall (2-model budget) | Option 3 → Option 5: MAE flags anomalous regions, ArcFace ViT classifies. Stage 2 runs only when Stage 1 confidence is low. |

---

## Relation to this repository (current milestone)

The codebase today implements the **binary** task described in the main `README.md` (`emergency_vehicle` vs `non_emergency`) using a **small CNN** and a standard **train / val / test** folder layout. That keeps the **ML pipeline** straightforward on coursework timelines: config → prepare data → train → evaluate → checkpoint → infer, with JSON metrics for reporting.

Moving to the **three-class** problem above (civilian / overt LE / covert LE) is primarily a **data and labeling** change: new class folders or a label manifest, then either **three logits on the same CNN** or a backbone swap (e.g. **Option 4 — DINOv2 + register tokens**) without changing those stages. The optional **two-model** design (e.g. MAE screen → ArcFace classifier) adds a **second checkpoint** and a **gate** on low stage-one confidence.
