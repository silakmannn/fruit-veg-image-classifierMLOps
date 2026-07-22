# Dataset Plan

## Dataset Version

Initial dataset version: `v1-local-10-class`

## Class List

1. apple
2. banana
3. orange
4. grape
5. strawberry
6. tomato
7. potato
8. carrot
9. cucumber
10. bell_pepper

## Split Strategy

Use three dataset splits:

- `train`: images used to train the model
- `val`: images used to tune and compare models during development
- `test`: images used only for final evaluation

Recommended split ratio:

- 70% train
- 15% validation
- 15% test

## Image Quality Checklist

Before training, check that:

- Images are readable common formats such as `.jpg`, `.jpeg`, or `.png`.
- Each image mainly contains one fruit or vegetable.
- Class folders use the exact names from `configs/classes.txt`.
- Each class has a similar number of images.
- Very blurry, duplicate, or unrelated images are removed.

## Notes

For the first version, keep the dataset simple. Prefer clear images with one object per image. We can make the dataset harder later after the baseline model works.
