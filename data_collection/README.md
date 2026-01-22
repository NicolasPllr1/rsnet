# Hand-Signed Digit Data Pipeline

3 Python scripts to:

1. Collect original data from the webcam using OpenCV (`collect.py`).
2. Augment this original data through translations, rotations and scalings
   (`augment.py`).
3. Package this data into a data ingestible by the rust codebase (`package.py`).

## Napkin Maths - How Much Data Do We Need?

The orignial MNIST dataset is (train, test) split is `(60_000, 1_000)` ( images
with a single color channel (grayscale).

**Note that for each original image we capture from the webcame, we will augment
it.**

A single source image will yield `3^4=81` augmented images -included itself), as
we apply all the posible cartesian product between:

- 3 scalings
- 3 horizontal translations
- 3 vertical translations
- 3 rotations

**To get our `61_000` images, we need `61_000 / 81 ~= 750` images.**

These should be uniformy spread across the 10 classes (10 digits). **For each
digit, we need to collect `75` images.**

To have a quality dataset, images should come with a diverse background,
lightings and general conditions. Let's say we want to capture images from 5
different settings.

**For each `5` settings, and then for each `10` digits, we need to collect `15`
images.**

## Usage

1. Run the `collect.py` script several times to collect data for all digits

```bash
uv run collect.py
```

2. Use the `build_dataset.sh` script to augment and package the data

```bash
./build_dataset.sh <data-dir> <output-dataset-dir>
```

## Pytorch training

You can train a model using the simple and hackable `train.py` script.

I used it to quickly check that a model _could_ train on the data I collected,
sweeping the optimizer from vanilla SGB, to SGD with momentum and finally using
Adam.

My Rust implementation was using SGB and didn't seem to train on my dataset. It
turns out the pytorch version is having issues too. This is symptom of SGD used
in a deeper CNN and more comple dataset compared to what's typically sufficient
to solve MNIST.
