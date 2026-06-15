# CIFAR-10 Image Classification — ANN vs CNN

## Objective

Build, train, and compare ANN and CNN models on the CIFAR-10 dataset, analyzing how architecture choices and training strategies (deeper layers, filter scaling, data augmentation, and early stopping) affect classification performance.

**Dataset:** CIFAR-10 — 50,000 training images and 10,000 test images, each 32×32 pixels with 3 RGB channels, across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Approach

The notebook follows a step-by-step pipeline: loading and visualizing the CIFAR-10 dataset, normalizing pixel values from the 0–255 range to 0.0–1.0, and flattening images to 1-D vectors for the ANN (since Dense layers require flat input) while keeping the 3-D shape for the CNN to preserve spatial information.

Two baseline models are built first — an ANN that treats images as flat vectors, and a CNN that uses Conv2D layers to extract local spatial features such as edges, textures, and shapes. From there, five tasks extend the baselines:

1. **Deeper ANN** — increased Dense layers from 512 → 256 → 10 to 512 → 256 → 128 → 64 → 10 for richer feature learning.
2. **Filter scaling in CNN** — progressive filter growth (32 → 64 → 128) across convolutional blocks, allowing the network to learn low-level features (edges) in early layers and high-level features (shapes) in deeper layers.
3. **Extended training** — both the deeper ANN and the filter-scaled CNN are retrained for 20 epochs to evaluate the effect of longer training.
4. **EarlyStopping** — a CNN variant trained for up to 30 epochs with EarlyStopping (patience=5, restore_best_weights=True, min_delta=0.001) to automatically halt training once validation loss stops improving and roll back to the best epoch.
5. **Data augmentation** — a CNN trained with RandomFlip, RandomRotation (±36°), and RandomZoom (10%) to synthetically expand the training data and reduce overfitting.

## Results

| Model | Test Accuracy | Task Covered |
|---|---|---|
| ANN (Baseline, 10 epochs) | 42.73% | Baseline ANN |
| CNN (Baseline, 10 epochs) | 73.91% | Baseline CNN |
| CNN (Augmented, 10 epochs) | 63.24% | Data Augmentation |
| ANN (Deep layout, 20 epochs) | 42.97% | More layers + 20 epochs |
| CNN (32→64→128 filters, 20 epochs) | 73.79% | Filter scaling + 20 epochs |
| CNN (EarlyStopping, ≤30 epochs) | 73.13% | EarlyStopping |

## Key Findings

**CNN significantly outperforms ANN** on image classification — the gap is roughly 30 percentage points across all comparable settings. This comes down to four structural advantages CNNs have over fully-connected networks on image data:

- **Weight sharing** — the same convolutional filter scans the entire image, drastically reducing the number of parameters compared to a fully-connected layer.
- **Translation invariance** — a learned feature (like a cat's ear) is detected regardless of where it appears in the image.
- **Hierarchical features** — early convolutional layers learn simple patterns like edges, while deeper layers combine these into complex shapes and object parts.
- **Spatial context preserved** — neighboring pixels remain spatially related throughout the network, whereas flattening for an ANN destroys this structure entirely.

**Adding more layers/epochs to the ANN barely moved accuracy** (42.73% → 42.97%), confirming that the bottleneck for ANN performance on images is architectural, not a matter of capacity or training time.

**Data augmentation reduced CNN accuracy in this 10-epoch run** (73.91% → 63.24%), likely because the augmented model needs more epochs to converge on the harder, more varied training distribution it's being shown — augmentation typically pays off over longer training schedules.

**Filter scaling and EarlyStopping CNN variants performed close to the CNN baseline** (73.79% and 73.13% vs. 73.91%), suggesting the baseline CNN was already near its capacity for this training budget, and EarlyStopping successfully avoided overfitting without sacrificing meaningful accuracy.

## Techniques Used

- **Batch Normalization** — stabilizes training, allows higher learning rates, and acts as a mild regularizer.
- **Dropout** — prevents co-adaptation of neurons and improves generalization.
- **EarlyStopping** — avoids over-training by monitoring validation loss and restoring the best weights.

## Tech Stack

- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib (visualization)

## Notebook

See [`CIFAR10_ANN_CNN_Assignment.ipynb`](./CIFAR10_ANN_CNN_Assignment.ipynb) for the full implementation, training logs, and visualizations.
