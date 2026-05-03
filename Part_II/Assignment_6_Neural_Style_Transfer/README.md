# Assignment 6 – Neural Style Transfer

## Overview
Creates **artistic images** by blending the *visual content* of one photograph
with the *painterly style* of another, using a **pre-trained VGG19** deep neural
network (Gatys et al., 2015).

The generated image is initialised from the content image and iteratively
optimised to minimise a weighted combination of:
- **Content loss** – preserves the content structure (layer `block4_conv2`)
- **Style loss** – matches Gram matrices across 5 VGG19 style layers
- **Total variation loss** – smoothness regulariser

## Requirements
```
tensorflow>=2.0
numpy
Pillow
matplotlib   (optional – for display)
```
Install with:
```bash
pip install tensorflow numpy Pillow matplotlib
```

## Usage
```bash
# Demo with synthetic images (no images required)
python neural_style_transfer.py

# Your own images
python neural_style_transfer.py --content photo.jpg --style painting.jpg

# Custom output path and more iterations for higher quality
python neural_style_transfer.py \
    --content photo.jpg \
    --style   painting.jpg \
    --output  art.jpg \
    --iterations 500

# All options
python neural_style_transfer.py --help
```

## Parameters
| Argument | Default | Description |
|---|---|---|
| `--content` | synthetic demo | Path to content image |
| `--style` | synthetic demo | Path to style image |
| `--output` | stylised_output.jpg | Output file path |
| `--iterations` | 100 | Optimisation steps (more = higher quality) |
| `--content_weight` | 1e4 | Weight for content loss |
| `--style_weight` | 1e-2 | Weight for style loss |
| `--tv_weight` | 30.0 | Weight for total variation smoothness |
| `--lr` | 0.02 | Adam learning rate |
| `--seed` | 42 | Random seed |
| `--max_dim` | 400 | Max image dimension (pixels) |

## Expected Output
```
============================================================
Neural Style Transfer – Creating Art with Deep Learning
============================================================
  Content image : synthetic demo (coloured circles)
  Style image   : synthetic demo (colour grid)
  Content shape : (256, 256, 3)
  ...
  Loading VGG19 weights (downloads ~80 MB on first run) ...
  Optimising for 100 iterations ...
  Iter    1/100  total=...  content=...  style=...
  ...
  Iter  100/100  total=...  content=...  style=...
  Saved output image: stylised_output.jpg

Style transfer complete!
```

## Notes
- VGG19 weights (~80 MB) are downloaded automatically from Keras on first run.
- For best results use 300–1000 iterations.
- Processing time: ~1–5 min on CPU (GPU recommended for faster results).
- This assignment is related to **Part I Assignment 8** (Neural Style Transfer).
  The implementation is standalone but uses the same VGG19-based approach.
