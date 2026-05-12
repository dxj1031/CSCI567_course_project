# Paper Narrative With Sample Figure Insertions

## One-Sentence Thesis

Camera-trap generalization across location and time is only partly improved by model capacity and simple image interventions; the strongest evidence points to object localization and background-context entanglement as the central bottleneck.

## Recommended Title

Object-Centric Cues, Not Capacity Alone, Drive Camera-Trap Generalization Across Location and Time

## Narrative Arc

### Act 1: Define The Generalization Problem

Start from the ecological/computer-vision motivation: camera-trap classifiers are deployed under natural domain shifts. A model trained on one set of locations or illumination regimes may fail when the background, viewpoint, animal scale, or day/night appearance changes.

Frame the project as three linked questions:

1. Does increasing CNN capacity improve cross-location and cross-time generalization?
2. Can image interventions that remove, randomize, or enhance nuisance information reduce the generalization gap?
3. If those interventions are weak, what visual evidence explains the remaining failures?

Insert:

- **Table 1:** dataset/split statistics by class and split.
- **Supplementary Figure S1:** class imbalance and failure modes.

Purpose of insertion:

Show that the problem is not only aggregate OOD accuracy; it is also rare classes, day/night asymmetry, and target-split difficulty.

### Act 2: Establish The Baseline And Capacity Result

Introduce the ResNet family as the controlled capacity axis. Keep this section sober: capacity improves some OOD metrics, but the normalized gap remains.

Core claim:

Capacity helps representation learning but does not solve camera-trap domain shift.

Insert:

- **Figure 1:** `figure1_capacity_generalization.pdf`

Text bridge:

> We first ask whether the generalization gap can be closed by scaling a standard CNN backbone. Although larger ResNets improve several target settings, the gap remains substantial, motivating interventions that target the visual sources of shift rather than capacity alone.

### Act 3: Show What The Interventions Visually Do

Before giving intervention results, insert a small visual methods figure. This helps reviewers understand that the interventions are concrete image manipulations rather than abstract labels.

Recommended main-text sample figure:

- Use selected contact sheets from `paper_sample_gallery_retry_20260509_155537/contact_sheets/`:
  - `01_original.jpg`
  - `06_combined.jpg`
  - `10_foreground_only.jpg`
  - `11_background_only.jpg`
  - `12_object_crop.jpg`

Do **not** use the current `03_bbox_blur_proxy.jpg` row in the paper figure. In the retry gallery script, that row was implemented as a foreground-only proxy with a small blur, so it is visually redundant with `10_foreground_only.jpg` and does not faithfully illustrate the formal bbox-blur intervention.

If a bbox-blur row is needed, regenerate it using the true formula:

```text
x' = M * x + (1 - M) * GaussianBlur(x)
```

where `M` is the padded bbox foreground mask. This preserves the animal and blurs the real background; it should not replace the background with gray.

Call this:

- **Figure 2:** visual intervention and diagnostic examples.

Suggested caption:

> **Figure 2. Visual examples of nuisance interventions and object-centric diagnostics.** Five randomly sampled CCT20 training images with available bounding boxes are shown under representative transformations. The perturbation rows illustrate training-time attempts to diversify non-object cues. The foreground-only, background-only, and object-crop rows illustrate diagnostic views used to test whether object localization and background context explain generalization failures. If included, the bbox-blur row should be regenerated with true background blurring rather than the current proxy.

Important caveat:

For `brightness_aligned_proxy`, write:

> These samples are visual proxies for the intervention family. Quantitative results are computed from the corresponding training pipelines, not from this gallery script.

Full gallery placement:

- **Supplementary Figure S2:** `panel_all_stages.jpg`

### Act 4: Report The Negative/Weak Intervention Results

Now show that visually plausible nuisance interventions did not reliably fix the problem.

Core claim:

Removing or perturbing nuisance cues is not enough; it may remove useful ecological context or create train-test mismatch.

Insert:

- **Figure 3:** `figure2_train_time_intervention_deltas.pdf`

Text bridge:

> Motivated by the visual differences in Figure 2, we next evaluate whether suppressing or diversifying nuisance information improves transfer. The effects are weak or negative across most settings, suggesting that the failure is not solved by simple background or brightness manipulation.

Writing stance:

Be honest and analytical. Do not hide that these results are not strong. Their value is that they rule out an overly simple nuisance-removal story.

### Act 5: Separate Visibility From Robustness

Discuss gamma/CLAHE/detail enhancement as a second intervention family. The result is nuanced: test-only changes are often harmful, while train-test consistent enhancement has small gains.

Core claim:

Visual enhancement alone is not a universal solution; distribution consistency matters.

Insert:

- **Figure 4:** `figure3_visibility_detail_effects.pdf`

Text bridge:

> A second hypothesis is that day/night transfer fails because low-light images obscure fine details. Visibility enhancement only weakly supports this: post-hoc test-time enhancement can hurt, while train-test consistent enhancement yields small average improvements.

### Act 6: Present The Strong Result: Object-Centric Diagnostic

This is the turning point. Object crop strongly improves OOD accuracy, macro-F1, and normalized gap across the three main settings.

Core claim:

The central bottleneck is object localization/context entanglement, not just model capacity, brightness, or class weighting.

Insert:

- **Figure 5:** `figure4_object_centric_diagnostic.pdf`

Key numbers to write:

- Cross-location: OOD accuracy `0.559 -> 0.745`, `+18.6 pp`; macro-F1 `0.447 -> 0.621`.
- Day-to-night: OOD accuracy `0.522 -> 0.713`, `+19.0 pp`; macro-F1 `0.444 -> 0.549`.
- Night-to-day: OOD accuracy `0.479 -> 0.588`, `+10.9 pp`; macro-F1 `0.447 -> 0.537`.

Text bridge:

> In contrast to the weak effects of global interventions, object-centric crops produce large improvements across all three domain shifts. This indicates that robust transfer improves when the classifier receives a stable view of the animal rather than the full camera-trap frame.

Important wording:

Always call object crop an **oracle localization diagnostic** unless you are explicitly proposing test-time bounding-box access.

### Act 7: Use Per-Class Results To Explain The Mechanism

Follow the aggregate object-crop result with class-level evidence. This prevents the paper from sounding like it only reports one large bar chart.

Core claim:

Object crops help many visually confusable/context-sensitive classes, but not all failures are localization failures.

Insert:

- **Figure 6:** `figure5_object_crop_per_class_delta.pdf`

Text bridge:

> The class-level analysis reveals where object-centric evidence helps most. Classes such as rabbit, bird, bobcat, opossum, cat, and coyote improve substantially in several target splits. Rodent remains near failure, suggesting a separate small-object, data-scarcity, or label-ambiguity issue.

### Act 8: Treat Rare-Class/Loss Follow-Ups As Secondary Evidence

Class weighting, balanced sampling, and focal loss should not be the central story. They are useful because they show that the object-crop gain is not simply explained by loss reweighting.

Insert:

- **Supplementary Figure S3:** `figureS2_class_balance_loss_followups.pdf`

Text bridge:

> Additional loss and sampling variants produce mixed effects and do not match the object-crop gains, further supporting the interpretation that the dominant bottleneck is visual support/localization rather than class weighting alone.

## Final Figure Order

### Main Text

1. **Figure 1:** Capacity and generalization gap.
2. **Figure 2:** Visual examples of interventions and object-centric diagnostics.
3. **Figure 3:** Train-time intervention deltas.
4. **Figure 4:** Visibility/detail enhancement results.
5. **Figure 5:** Object-centric diagnostic gains.
6. **Figure 6:** Per-class object-crop effects.

### Supplementary

1. **Supplementary Figure S1:** Class imbalance and failure modes.
2. **Supplementary Figure S2:** Full sample gallery, `panel_all_stages.jpg`.
3. **Supplementary Figure S3:** Class-balance/loss follow-ups.

## Revised Section Outline

### Abstract

Open with the deployment problem, not the interventions. End with the object-centric conclusion.

Suggested abstract result sentence:

> Across three domain shifts, increasing ResNet capacity improves performance but leaves persistent generalization gaps, while most nuisance and visibility interventions have weak or unstable effects. In contrast, oracle object crops improve OOD accuracy by 10.9-19.0 percentage points, indicating that object localization and context entanglement are major barriers to robust camera-trap recognition.

### Introduction

End the Introduction with the contribution list:

1. A controlled ResNet capacity study on cross-location and cross-time CCT20 splits.
2. A systematic evaluation of nuisance-reduction, perturbation, and visibility/detail interventions.
3. An object-centric diagnostic showing that oracle localization gives the largest and most consistent gains.
4. A per-class analysis showing which failures are localization-sensitive and which remain unresolved.

### Dataset And Experimental Setup

Introduce splits and metrics before models. Put class imbalance here, with Supplementary Figure S1.

### Methods

Order methods in the same order as the story:

1. ResNet capacity baselines.
2. Train-time nuisance reduction and perturbation.
3. Visibility/detail preprocessing.
4. Rare-class loss/sampling follow-ups.
5. Object-centric diagnostic ablations.

Insert Figure 2 at the end of Methods.

### Results

Order results from broad to diagnostic:

1. Capacity results.
2. Nuisance/perturbation results.
3. Visibility/detail results.
4. Object-centric diagnostic results.
5. Per-class analysis.

### Discussion

Lead with the interpretation:

> The strongest result is not that one preprocessing trick wins, but that the visual support of the classifier matters. When the animal is localized and consistently framed, transfer improves sharply; when only background remains, performance drops.

Then explain why earlier interventions may have failed:

- Background/context can be both nuisance and useful ecological signal.
- Random perturbations may not match real domain shifts.
- Test-time enhancement can create distribution mismatch.
- Rare-class losses cannot recover missing visual evidence.

### Limitations

Put the object-crop caveat first:

- Object crop is an oracle diagnostic using bounding boxes.
- Object-centric results currently need more seeds for a stronger final claim.
- Proxy sample gallery is illustrative; quantitative claims come from experiment artifacts.
- Rare classes like rodent remain unresolved.

### Conclusion

End with a clear forward-looking claim:

> Camera-trap generalization should be approached as an object-aware robustness problem. Scaling CNNs and applying global image transformations are insufficient; future methods should incorporate localization, object-context disentanglement, or multi-view object/context fusion.

## Short Chinese Summary For Your Own Writing

这篇 paper 的叙述不要写成“我试了很多 intervention，但是结果不好”。更好的写法是：

1. 先证明问题真实存在：cross-location 和 day/night 都有 generalization gap。
2. 再证明 capacity 不是万能药：ResNet 变大有帮助，但 gap 仍在。
3. 再证明简单干预不够：背景、亮度、扰动、增强都不是稳定解。
4. 然后拿出关键诊断：object crop 大幅提升，background-only 很差。
5. 最后给出 insight：问题核心是 object localization 和 context entanglement，而不是单纯亮度、背景、模型大小或 class imbalance。
