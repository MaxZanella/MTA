# On the Test-Time Zero-Shot Generalization of Vision-Language Models: Do We Really Need Prompt Learning? [accepted at CVPR 2024]

Welcome to the official GitHub repository for our paper accepted at CVPR '24. This work introduces the MeanShift Test-time Augmentation (MTA) method, leveraging Vision-Language models without the necessity for prompt learning.

Our method randomly augments a single image into N augmented views, then alternates between two key steps:

### 1. Computing a Score for Each Augmented View

This step involves calculating a score for each augmented view to assess its relevance and quality (inlierness score).

<p align="center">
  <img src="inlierness.png" alt="Score computation for augmented views" width="500" height="150">
  <br>
  <em>Figure 1: Score computation for augmented views.</em>
</p>

### 2. Seeking for the Mode

Based on the scores computed in the previous step, we seek the mode of the data (MeanShift).

<p align="center">
  <img src="mode.png" alt="Seeking the mode, weighted by score" width="500" height="180">
  <br>
  <em>Figure 2: Seeking the mode, weighted by score.</em>
</p>


## Dataset Preparation

We follow TPT preprocessing. This ensures that your dataset is formatted appropriately. You can find their repository [here](https://github.com/azshue/TPT).

## Quick Start Guide

### Running MTA

Execute MTA on the ImageNet dataset with a random seed of 1 and 'a photo of a' prompt by entering the following command:

```bash
python main.py --data /path/to/your/data --mta --testsets I --seed 1
```

Or the 15 datasets at once:

```bash
python main.py --data /path/to/your/data --mta --testsets I/A/R/V/K/DTD/Flower102/Food101/Cars/SUN397/Aircraft/Pets/Caltech101/UCF101/eurosat --seed 1
```

## Acknowledgement

We express our gratitude to the TPT authors for their open-source contribution. You can find their repository [here](https://github.com/azshue/TPT). 

