# Research Assessment Figures and Key Numbers

This file was generated from the shared-drive result CSVs, but all outputs were written locally.

## Figure Files

- `figures/fig1_capacity_generalization.png` and `figures/fig1_capacity_generalization.pdf`
- `figures/fig2_train_intervention_deltas.png` and `figures/fig2_train_intervention_deltas.pdf`
- `figures/fig3_visibility_detail_effects.png` and `figures/fig3_visibility_detail_effects.pdf`
- `figures/fig4_class_imbalance_and_failures.png` and `figures/fig4_class_imbalance_and_failures.pdf`

## Capacity: Best Scenario Winners

- cross_location: resnet101 OOD accuracy=0.612, normalized gap=0.176.
- day_to_night: resnet101 OOD accuracy=0.577, normalized gap=0.185.
- night_to_day: resnet50 OOD accuracy=0.466, normalized gap=0.313.

## Train-Time Intervention Summary

- nuisance reduction: mean Delta OOD=-0.056, positive cells=3/24.
- distribution diversification: mean Delta OOD=-0.004, positive cells=16/36.
- visibility/detail, train+test consistent only: mean Delta OOD=+0.006, positive cells=22/36.
- visibility/detail, test-only only: mean Delta OOD=-0.037, positive cells=1/36.

## Consistent Class-Level Failures

- cross_location: rodent F1=0.000, bird F1=0.201, cat F1=0.310, rabbit F1=0.323.
- day_to_night: rodent F1=0.000, bird F1=0.056, dog F1=0.067, rabbit F1=0.081.
- night_to_day: rodent F1=0.000, cat F1=0.062, bird F1=0.098, skunk F1=0.246.
- rodent has only 128 training images and reaches F1=0.000 in every hardest target split.
- squirrel has 432 day images and 0 night images, so time-shift splits are intrinsically asymmetric.

## Representative Confusion Clusters

- cross_location: raccoon->coyote (155), coyote->cat (153), bobcat->cat (151).
- day_to_night: raccoon->opossum (326), raccoon->coyote (230), opossum->raccoon (226).
- night_to_day: cat->rabbit (167), coyote->rabbit (87), coyote->cat (78).