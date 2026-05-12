# Question-Driven Intervention Taxonomy

This taxonomy groups the 11 non-original visual conditions by the research question they answer, rather than by the image operation they use.

## Central Question

Why do camera-trap classifiers fail to generalize across location and time?

The interventions are organized as competing explanations:

1. **Is the gap caused by low-level appearance shift?**
2. **Is the gap caused by background/context shortcuts?**
3. **Is the gap caused by insufficient invariance to nuisance variation?**
4. **Is the gap caused by weak object visibility/detail?**
5. **Is the gap caused by missing object localization?**

## Question-Driven Groups

| Research question | Experiments | What they test | Expected interpretation |
|---|---|---|---|
| **Q1. Is day/night or location shift mainly a low-level appearance mismatch?** | `02_brightness_aligned_proxy`, `07_gamma`, `08_clahe`, `09_gamma_clahe` | Whether matching brightness or enhancing contrast/detail reduces cross-time or cross-location gap | If strong: appearance normalization is enough. If weak: the failure is not just brightness/detail. |
| **Q2. Is the model overusing background or camera-location context?** | `03_bbox_blur_proxy`, `05_background_perturbation`, `11_background_only` | Whether suppressing, perturbing, or isolating background changes transfer | If background-only works well: background shortcuts are strong. If background-only fails but object crop works: background alone is not sufficient. |
| **Q3. Can training-time nuisance diversification make the model invariant?** | `04_photometric_randomization`, `05_background_perturbation`, `06_combined` | Whether stochastic variation in color, illumination, and background improves robustness | If strong: augmentation-induced invariance helps. If weak: random perturbations do not match the true shift or remove useful cues. |
| **Q4. Is object evidence sufficient when context is removed?** | `10_foreground_only`, `11_background_only` | Whether foreground object pixels or background pixels carry the transferable class evidence | Foreground > background suggests the object matters most, but full cropping/framing may still be needed. |
| **Q5. Is the key bottleneck object localization and framing?** | `12_object_crop`, plus comparison to `10_foreground_only` and `11_background_only` | Whether an oracle object-centered view improves transfer more than masking background | Strong object-crop gains indicate localization/context entanglement is the main bottleneck. |

## Mapping From Gallery Stage To Question

| Gallery stage | Paper name | Primary question | Secondary question |
|---|---|---|---|
| `02_brightness_aligned_proxy` | Brightness alignment / histogram matching | Q1: low-level appearance mismatch | Q4 if day/night affects object visibility |
| `03_bbox_blur_proxy` | BBox background blur | Q2: background/context shortcuts | Q3 if treated as training-time nuisance suppression |
| `04_photometric_randomization` | Photometric randomization | Q3: nuisance invariance | Q1: appearance shift |
| `05_background_perturbation` | Background perturbation | Q2: background/context shortcuts | Q3: nuisance invariance |
| `06_combined` | Combined perturbation | Q3: nuisance invariance | Q1/Q2 jointly |
| `07_gamma` | Gamma correction | Q1: low-level appearance mismatch | Q4: object visibility |
| `08_clahe` | CLAHE | Q1: low-level appearance mismatch | Q4: object detail |
| `09_gamma_clahe` | Gamma + CLAHE | Q1: low-level appearance mismatch | Q4: object visibility/detail |
| `10_foreground_only` | Foreground-only diagnostic | Q4: object evidence sufficiency | Q5: localization/framing |
| `11_background_only` | Background-only diagnostic | Q2: background/context shortcuts | Q4 as a negative control |
| `12_object_crop` | Object-crop diagnostic | Q5: localization/framing | Q4: object evidence |

## Recommended Paper Framing

### Question 1: Is the problem mostly appearance mismatch?

Use:

- Brightness alignment
- Gamma
- CLAHE
- Gamma + CLAHE

Paper claim:

> Appearance normalization and visibility enhancement test whether cross-time generalization fails because night images differ in brightness and contrast. The weak and scope-dependent results suggest that low-level appearance mismatch is not the whole explanation.

### Question 2: Is the model exploiting background shortcuts?

Use:

- BBox background blur
- Background perturbation
- Background-only

Paper claim:

> Background manipulations test whether the model relies on camera-location context. Background-only performance is poor, so background alone does not explain the classifier. However, weak effects from background suppression/perturbation suggest context is entangled with useful object evidence rather than being a removable nuisance.

### Question 3: Can nuisance diversification solve the gap?

Use:

- Photometric randomization
- Background perturbation
- Combined perturbation

Paper claim:

> Training-time diversification asks whether robustness can be induced by making nuisance factors unstable. The mixed results suggest that generic stochastic perturbations do not reliably approximate the real cross-location or cross-time shift.

### Question 4: Are object pixels sufficient?

Use:

- Foreground-only
- Background-only

Paper claim:

> Foreground-only and background-only views separate object evidence from context. Foreground-only generally carries more transferable signal than background-only, but it does not match the object-crop gains, implying that object scale and framing matter beyond merely deleting background.

### Question 5: Is object localization the central bottleneck?

Use:

- Object crop
- Compare against full image, foreground-only, and background-only

Paper claim:

> Object crop is the decisive diagnostic. Because it improves OOD accuracy and macro-F1 most consistently, the main bottleneck appears to be object localization and context entanglement rather than capacity, brightness, background alone, or class imbalance alone.

## Best Results Narrative

The strongest question-driven story is:

1. **Capacity question:** larger ResNets help but leave a gap.
2. **Appearance question:** brightness/detail changes are not enough.
3. **Background question:** background alone is not sufficient, and suppressing it does not reliably fix transfer.
4. **Nuisance-invariance question:** random perturbations are mixed.
5. **Localization question:** object crops give the clearest gains.

## One-Paragraph Paper Version

We organize the interventions as tests of competing explanations for camera-trap domain shift. Brightness alignment and visibility enhancement test whether cross-time transfer is mainly a low-level appearance problem. Background blur, background perturbation, and background-only views test whether the model exploits camera-location context. Photometric and combined perturbations test whether nuisance diversification induces invariance. Foreground-only and background-only diagnostics separate object evidence from context. Finally, object crops test whether oracle localization and stable object framing resolve the transfer bottleneck. The resulting pattern supports the localization explanation most strongly: global appearance and nuisance interventions are weak or mixed, background-only views perform poorly, and object crops produce the largest and most consistent OOD gains.
