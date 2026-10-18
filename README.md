# MNIST Data Augmentation Ablation Study

This project investigates how different augmentation strategies affect
MNIST CNN classification.

## Research Question

Does data augmentation improve MNIST classification, and does adding
more augmentation always produce better validation accuracy?

## Controlled Variables

The following variables remain fixed:

- CNN architecture
- Number of epochs
- Random seed
- Batch size
- Optimizer
- Learning rate
- Dataset
- Validation/test split

## Experimental Configurations

### Experiment 1

Baseline:
No augmentation

### Experiment 2

Rotation:
RandomRotation(15)

### Experiment 3

Rotation + Shift:
RandomRotation(15)
RandomAffine(translate=10%)

### Experiment 4

Rotation + Shift + Noise:
RandomRotation(15)
RandomAffine(translate=10%)
Gaussian noise

The project also supports:

- custom rotated test images
- shifted test images
- combined rotated/shifted images
- training-time measurement
- MixUp
- model checkpoint saving
- validation curves

Run:
python experiment.py

Then:
python plot_results.py

Output:
Results are saved to:
results/results.csv

Models:
results/models/

Charts:
results/comparison.png

## Conclusion

The ablation study was designed to isolate the effect of increasingly
strong data augmentation strategies on MNIST digit classification.
All four experiments used the same CNN architecture, optimizer, learning
rate, batch size, number of epochs, dataset split, and random seed.
Therefore, the primary experimental variable was the augmentation strategy.
The baseline model provides a reference point because it is trained only
on the original MNIST images. Rotation augmentation tests whether small
orientation changes improve the model's ability to recognize digits.
Rotation combined with translation additionally exposes the model to
digits appearing at slightly different spatial positions. This is
relevant because handwritten digits are not always perfectly centered.
The final configuration adds Gaussian noise on top of rotation and
translation. Although noise can improve robustness when it resembles
realistic input variation, excessive noise may remove useful information
from the small 28x28 MNIST images. Therefore, the effect of noise should
be evaluated from the measured validation accuracy rather than assumed
before the experiment.
The results should be interpreted by comparing both validation accuracy
and training time. If accuracy improves from one configuration to another,
the additional augmentation may provide useful variability. If accuracy
stops improving or decreases, this may indicate diminishing returns or
over-augmentation. The custom rotated and shifted test set provides a
second perspective by measuring robustness to controlled transformations.
Overall, the experiment demonstrates why augmentation should be treated
as an empirical hypothesis rather than automatically adding every
available transformation.
