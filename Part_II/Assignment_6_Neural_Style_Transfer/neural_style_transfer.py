"""
Assignment 6 – Neural Style Transfer (NST)
===========================================
Creates art by blending the *content* of one image with the *style* of another
using a pre-trained VGG19 convolutional neural network.

The algorithm minimises a weighted combination of:
  - Content loss  – how similar the output is to the content image
  - Style loss    – how similar the output is to the style image (via Gram matrices)
  - Total variation loss – smoothness regulariser

Usage
-----
    python neural_style_transfer.py                        # demo with synthetic images
    python neural_style_transfer.py --content photo.jpg --style painting.jpg
    python neural_style_transfer.py --content photo.jpg --style painting.jpg \
        --output art.jpg --iterations 300
    python neural_style_transfer.py --help

Requirements
------------
    pip install tensorflow numpy Pillow matplotlib
"""

import argparse
import os
import sys
import numpy as np


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _import_tf():
    """Import TensorFlow with a helpful error message if not installed."""
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        print("[ERROR] TensorFlow is not installed.")
        print("  Install with:  pip install tensorflow")
        sys.exit(1)


def load_image(path: str, max_dim: int = 512) -> np.ndarray:
    """Load and resize an image so the longest dimension <= max_dim."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_dim / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img, dtype=np.float32)


def save_image(array: np.ndarray, path: str):
    """Save a float32 [0,255] array as an image file."""
    from PIL import Image
    array = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)
    print(f"  Saved output image: {path}")


def make_synthetic_content(h: int = 256, w: int = 256, seed: int = 42) -> np.ndarray:
    """Create a simple synthetic content image (coloured circles)."""
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w, 3), dtype=np.float32)
    for _ in range(20):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        r = rng.randint(10, 60)
        color = rng.randint(50, 256, 3).astype(np.float32)
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        img[mask] = color
    return img


def make_synthetic_style(h: int = 256, w: int = 256, seed: int = 99) -> np.ndarray:
    """Create a simple synthetic style image (colour grid)."""
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w, 3), dtype=np.float32)
    tile = 32
    for i in range(0, h, tile):
        for j in range(0, w, tile):
            color = rng.randint(30, 230, 3).astype(np.float32)
            img[i:i + tile, j:j + tile] = color
    return img


# ---------------------------------------------------------------------------
# VGG19 model setup
# ---------------------------------------------------------------------------

CONTENT_LAYERS = ["block4_conv2"]
STYLE_LAYERS = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]


def build_feature_model(tf):
    """Build a model that outputs VGG19 content and style layer activations."""
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False
    layer_names = CONTENT_LAYERS + STYLE_LAYERS
    outputs = {layer.name: layer.output
               for layer in vgg.layers if layer.name in layer_names}
    return tf.keras.Model(inputs=vgg.input, outputs=outputs)


def vgg_preprocess(img: np.ndarray, tf):
    """Add batch dimension and apply VGG19 preprocessing."""
    t = tf.cast(img, tf.float32)[tf.newaxis, :]
    return tf.keras.applications.vgg19.preprocess_input(t)


def vgg_deprocess(tensor, tf) -> np.ndarray:
    """Reverse VGG19 preprocessing to recover a displayable RGB image."""
    img = tensor.numpy()[0].copy()
    img[:, :, 0] += 103.939  # B mean
    img[:, :, 1] += 116.779  # G mean
    img[:, :, 2] += 123.68   # R mean
    img = img[:, :, ::-1]    # BGR -> RGB
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def gram_matrix(feature_map, tf):
    """Compute normalised Gram matrix for a (1, H, W, C) feature map."""
    result = tf.linalg.einsum("bijc,bijd->bcd", feature_map, feature_map)
    n_loc = tf.cast(
        tf.shape(feature_map)[1] * tf.shape(feature_map)[2], tf.float32
    )
    return result / n_loc


def compute_loss_and_grads(model, generated_var,
                            content_targets, style_targets,
                            content_weight, style_weight, tv_weight, tf):
    """Compute total loss and gradients w.r.t. generated image."""
    with tf.GradientTape() as tape:
        outputs = model(generated_var, training=False)

        # Content loss
        c_loss = content_weight * tf.reduce_mean(
            tf.square(outputs[CONTENT_LAYERS[0]] - content_targets[CONTENT_LAYERS[0]])
        )

        # Style loss (averaged across style layers)
        s_loss = 0.0
        for layer in STYLE_LAYERS:
            gram_gen = gram_matrix(outputs[layer], tf)
            s_loss += tf.reduce_mean(tf.square(style_targets[layer] - gram_gen))
        s_loss = style_weight * s_loss / len(STYLE_LAYERS)

        # Total variation loss
        x_deltas = generated_var[:, :, 1:, :] - generated_var[:, :, :-1, :]
        y_deltas = generated_var[:, 1:, :, :] - generated_var[:, :-1, :, :]
        tv_loss = tv_weight * (
            tf.reduce_mean(tf.abs(x_deltas)) + tf.reduce_mean(tf.abs(y_deltas))
        )

        total_loss = c_loss + s_loss + tv_loss

    grads = tape.gradient(total_loss, generated_var)
    return total_loss, c_loss, s_loss, grads


# ---------------------------------------------------------------------------
# NST pipeline
# ---------------------------------------------------------------------------

def run_nst(content_img: np.ndarray, style_img: np.ndarray,
            iterations: int = 100, content_weight: float = 1e4,
            style_weight: float = 1e-2, tv_weight: float = 30.0,
            learning_rate: float = 0.02, seed: int = 42) -> np.ndarray:
    """Run Neural Style Transfer and return the stylised RGB image array."""
    tf = _import_tf()
    tf.random.set_seed(seed)

    print("  Loading VGG19 weights (downloads ~80 MB on first run) ...")
    model = build_feature_model(tf)

    # Pre-compute targets
    content_tensor = vgg_preprocess(content_img, tf)
    style_tensor = vgg_preprocess(style_img, tf)

    content_outputs = model(content_tensor, training=False)
    style_outputs = model(style_tensor, training=False)

    content_targets = {k: content_outputs[k] for k in CONTENT_LAYERS}
    style_targets = {k: gram_matrix(style_outputs[k], tf) for k in STYLE_LAYERS}

    # Initialise generated image from content image with small noise
    rng = np.random.RandomState(seed)
    noise = rng.randn(*content_img.shape).astype(np.float32) * 0.5
    generated = tf.Variable(vgg_preprocess(content_img + noise, tf))

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, beta_1=0.99, epsilon=1e-1
    )

    print(f"  Optimising for {iterations} iterations ...")
    for i in range(1, iterations + 1):
        total_loss, c_loss, s_loss, grads = compute_loss_and_grads(
            model, generated, content_targets, style_targets,
            content_weight, style_weight, tv_weight, tf
        )
        optimizer.apply_gradients([(grads, generated)])
        generated.assign(tf.clip_by_value(generated, -128.0, 128.0))

        if i % 20 == 0 or i == 1:
            print(f"  Iter {i:>4}/{iterations}  "
                  f"total={float(total_loss):.2f}  "
                  f"content={float(c_loss):.2f}  "
                  f"style={float(s_loss):.4f}")

    return vgg_deprocess(generated, tf)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Neural Style Transfer – create art with deep learning."
    )
    parser.add_argument("--content", default=None,
                        help="Path to content image (default: synthetic demo)")
    parser.add_argument("--style", default=None,
                        help="Path to style image (default: synthetic demo)")
    parser.add_argument("--output", default="stylised_output.jpg",
                        help="Output image path (default: stylised_output.jpg)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Optimisation iterations (default: 100)")
    parser.add_argument("--content_weight", type=float, default=1e4,
                        help="Content loss weight (default: 1e4)")
    parser.add_argument("--style_weight", type=float, default=1e-2,
                        help="Style loss weight (default: 1e-2)")
    parser.add_argument("--tv_weight", type=float, default=30.0,
                        help="Total variation loss weight (default: 30)")
    parser.add_argument("--lr", type=float, default=0.02,
                        help="Learning rate (default: 0.02)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--max_dim", type=int, default=400,
                        help="Max image dimension in pixels (default: 400)")
    args = parser.parse_args()

    print("=" * 60)
    print("Neural Style Transfer – Creating Art with Deep Learning")
    print("=" * 60)

    if args.content:
        print(f"\n  Content image : {args.content}")
        content_img = load_image(args.content, max_dim=args.max_dim)
    else:
        print("\n  Content image : synthetic demo (coloured circles)")
        content_img = make_synthetic_content(256, 256, seed=args.seed)

    if args.style:
        print(f"  Style image   : {args.style}")
        style_img = load_image(args.style, max_dim=args.max_dim)
    else:
        print("  Style image   : synthetic demo (colour grid)")
        style_img = make_synthetic_style(256, 256, seed=99)

    print(f"  Content shape : {content_img.shape}")
    print(f"  Style shape   : {style_img.shape}")
    print(f"  Iterations    : {args.iterations}")
    print(f"  Seed          : {args.seed}")
    print()

    output = run_nst(
        content_img=content_img,
        style_img=style_img,
        iterations=args.iterations,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        learning_rate=args.lr,
        seed=args.seed,
    )

    save_image(output, args.output)
    print("\nStyle transfer complete!")


if __name__ == "__main__":
    main()
