# Dataset Source

## Source

This project uses the Fruits-360 100x100 dataset as the first local image source.

- Official GitHub organization: https://github.com/fruits-360
- Dataset repository: https://github.com/fruits-360/fruits-360-100x100
- License: CC BY-SA 4.0

## Local Source Mapping

The first project dataset maps one Fruits-360 source folder to each project class:

| Project class | Fruits-360 source folder |
| --- | --- |
| apple | Apple Golden 1 |
| banana | Banana 1 |
| orange | Orange 1 |
| grape | Grape Blue 1 |
| strawberry | Strawberry 1 |
| tomato | Tomato 1 |
| potato | Potato White 1 |
| carrot | Carrot 1 |
| cucumber | Cucumber 1 |
| bell_pepper | Pepper Red 1 |

## Local Split

For the beginner baseline, the local dataset uses:

- 150-200 training images per class
- 50 validation images per class
- 50 test images per class

Carrot and cucumber have fewer available images in the selected Fruits-360 source folders, so their training counts are slightly lower than the other classes.

Raw images are stored locally under `data/raw` and are intentionally ignored by Git.
