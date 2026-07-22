# Dataset

This project expects a 10-class fruit and vegetable image dataset.

## Classes

Use these exact class folder names:

```text
apple
banana
orange
grape
strawberry
tomato
potato
carrot
cucumber
bell_pepper
```

## Folder Structure

Place images into this structure:

```text
data/raw/
|-- train/
|   |-- apple/
|   |-- banana/
|   |-- orange/
|   |-- grape/
|   |-- strawberry/
|   |-- tomato/
|   |-- potato/
|   |-- carrot/
|   |-- cucumber/
|   `-- bell_pepper/
|-- val/
|   |-- apple/
|   |-- banana/
|   |-- orange/
|   |-- grape/
|   |-- strawberry/
|   |-- tomato/
|   |-- potato/
|   |-- carrot/
|   |-- cucumber/
|   `-- bell_pepper/
`-- test/
    |-- apple/
    |-- banana/
    |-- orange/
    |-- grape/
    |-- strawberry/
    |-- tomato/
    |-- potato/
    |-- carrot/
    |-- cucumber/
    `-- bell_pepper/
```

## Beginner Target

A good starting target is:

- 100-300 training images per class
- 20-50 validation images per class
- 20-50 test images per class

The model can still be tested with fewer images while learning, but balanced classes will make results easier to understand.

## Git Rule

Image files are intentionally ignored by Git. The folder structure is tracked with `.gitkeep` files, but the dataset itself should stay local.
