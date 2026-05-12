# Final Paper Figure Captions

## Figure 1. Capacity and generalization gap across domain shifts
Out-of-domain accuracy and normalized generalization gap for the ResNet family under cross-location and cross-time camera-trap splits. Larger backbones improve several settings, but model capacity alone does not eliminate the domain gap.

## Figure 2. Training-set nuisance interventions mostly fail to improve transfer
Change in out-of-domain accuracy and normalized gap after train-time nuisance reduction/intervention variants. Most cells are neutral or negative, indicating that simple removal or randomization of nuisance cues is not sufficient to produce robust camera-trap generalization.

## Figure 3. Visibility/detail enhancement has limited and scope-dependent benefits
Test-time only visibility enhancement is generally harmful, while train-test consistent enhancement yields small average gains. This supports the view that representation consistency matters more than post-hoc image enhancement.

## Figure 4. Object-centric localization produces the strongest diagnostic gain
BBox-derived object crops substantially improve OOD accuracy and macro-F1 and reduce normalized gaps across cross-location, day-to-night, and night-to-day transfer. This should be interpreted as an oracle localization diagnostic rather than a deployable baseline unless bounding boxes are available at test time.

## Figure 5. Class-level effects of object cropping
Per-class F1 changes on the hardest target split show that object crops mainly help visually confusable or context-sensitive classes such as rabbit, bird, bobcat, opossum, cat, and coyote. Rodent remains near failure, suggesting a distinct small-object/data-scarcity problem.

## Supplementary Figure S1. Class imbalance and failure modes
Training support and hardest-split F1 reveal consistent rare-class failures, especially rodent, and clarify why aggregate accuracy can hide class-level weaknesses.

## Supplementary Figure S2. Rare-class/loss follow-ups
No class weights, class-balanced sampling, and focal loss produce mixed seed-42 effects and do not match the object-crop diagnostic gains. These runs are useful secondary evidence, but should not be the central claim without more seeds.
