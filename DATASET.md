# Dataset

The models are trained on `skin-ds-processed`, a 14-class image set derived from
the public Kaggle dataset [`ahmedxc4/skin-ds`](https://www.kaggle.com/datasets/ahmedxc4/skin-ds).
It is not tracked in git. Fetch and regenerate it as described below.

## Composition

The 14 classes span two clinically distinct groups. One is pigmented and
keratinocytic lesions typically imaged with dermoscopy, several of them from the
ISIC archive (melanoma, melanocytic nevi, basal and squamous cell carcinoma,
actinic keratoses, benign keratosis-like lesions, dermatofibroma, vascular
lesions). The other is infectious and rash conditions imaged as clinical photos
(monkeypox, cowpox, chickenpox, measles, hand-foot-and-mouth disease), plus a
healthy-skin class. Mixing the two groups in one label space is a property of the
source dataset.

## Per-class counts

The set holds 32,982 images: 29,322 train and 3,660 validation.

| Class | Train | Val |
|-------|-------|-----|
| Melanocytic nevi | 10,300 | 1,287 |
| Melanoma | 3,617 | 452 |
| Monkeypox | 3,408 | 426 |
| Basal cell carcinoma | 2,658 | 332 |
| Benign keratosis-like lesions | 2,099 | 262 |
| HFMD | 1,932 | 241 |
| Healthy | 1,368 | 171 |
| Chickenpox | 900 | 112 |
| Cowpox | 792 | 99 |
| Actinic keratoses | 693 | 86 |
| Measles | 660 | 82 |
| Squamous cell carcinoma | 502 | 62 |
| Vascular lesions | 202 | 25 |
| Dermatofibroma | 191 | 23 |

The distribution is long-tailed: melanocytic nevi has roughly 54x the training
images of dermatofibroma. Every training run uses a `WeightedRandomSampler` to
counter this, and Custom CNN V2 adds focal loss. Validation is left at its
natural distribution, so per-class accuracy is more informative than overall
accuracy.

## Layout

Standard `torchvision.datasets.ImageFolder` tree, one directory per class:

```
skin-ds-processed/
├── train/
│   ├── Melanocytic nevi/
│   ├── Melanoma/
│   └── ... (14 classes)
└── val/
    ├── Melanocytic nevi/
    └── ...
```

## Preprocessing

`analysis/hair_removal.ipynb` produces `skin-ds-processed` from the raw Kaggle download.
Hair occludes lesions and adds high-frequency texture that the models can
exploit, so each image is passed through a hair-removal step before training:

1. Convert to grayscale and apply a BLACKHAT morphological transform with a
   rectangular kernel, which highlights thin dark structures (hairs) against the
   skin.
2. Threshold the BLACKHAT response into a binary hair mask.
3. Inpaint the masked regions with `cv2.INPAINT_TELEA`, filling hair pixels from
   surrounding skin.

The cleaned images are written to the output tree with the original class
structure preserved.

Models were evaluated with and without this step, and explicit hair removal did
not improve accuracy. It is kept for reproducibility.

## Reproducing

```python
import kagglehub
path = kagglehub.dataset_download("ahmedxc4/skin-ds")
```

Point the `RAW_DATA_DIR` in `analysis/hair_removal.ipynb` at the download and run it once
to generate `skin-ds-processed`. The notebook was written for Kaggle, so its
`/kaggle/...` paths must be repointed to run elsewhere.

## Licensing

`ahmedxc4/skin-ds` aggregates images from multiple sources, including the ISIC
archive, each under its own terms. Check the upstream licenses before
redistributing any images. This repository ships neither the raw nor the
processed images. It references the Kaggle source and the preprocessing code so
the dataset can be rebuilt.
