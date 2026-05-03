"""
Practical 8: Neural Style Transfer (NST)
============================================================
Combines the *content* of one image with the *artistic style* of another
using a pre-trained VGG19 deep convolutional network as a feature extractor.

Algorithm:
    1. Extract content features from a deep VGG19 layer (block5_conv2).
    2. Extract style features (Gram matrices) from five shallower layers.
    3. Initialise the generated image as a copy of the content image.
    4. Iteratively minimise:
           total_loss = alpha * content_loss + beta * style_loss
       via Adam gradient descent on the pixel values of the output image.

Usage:
    # Demo with auto-generated synthetic images:
    python 7_Neural_Style_Transfer.py

    # Use your own images:
    python 7_Neural_Style_Transfer.py --content content.jpg --style style.jpg

    # Full options:
    python 7_Neural_Style_Transfer.py \\
        --content content.jpg --style style.jpg \\
        --output  result.png  --iterations 300 --image-size 256

Note:
    First run downloads VGG19 weights (~550 MB). Internet access required
    for that initial download; subsequent runs use the cached weights.

Dependencies: tensorflow>=2.8, Pillow, numpy, matplotlib
"""

import argparse
import os

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic sample-image generator (no external images required)
# ---------------------------------------------------------------------------

def _generate_content_image(size: int = 128,
                             path: str = "sample_content.png") -> str:
    """Create a simple geometric-shapes image as content sample.

    Args:
        size: Image width and height in pixels.
        path: File path to save the generated image.

    Returns:
        Path to the saved image.
    """
    if os.path.exists(path):
        return path
    from PIL import Image, ImageDraw
    img  = Image.new("RGB", (size, size), color=(230, 230, 230))
    draw = ImageDraw.Draw(img)
    # Steel-blue rectangle
    draw.rectangle(
        [size // 5, size // 5, 4 * size // 5, 4 * size // 5],
        fill=(70, 130, 180),
    )
    # White filled ellipse
    draw.ellipse(
        [size // 3, size // 3, 2 * size // 3, 2 * size // 3],
        fill=(255, 255, 255),
    )
    # Dark triangle approximation
    pts = [
        (size // 2,     size // 8),
        (7 * size // 8, 7 * size // 8),
        (size // 8,     7 * size // 8),
    ]
    draw.polygon(pts, fill=(40, 40, 90))
    img.save(path)
    print(f"  Generated synthetic content image -> {path}")
    return path


def _generate_style_image(size: int = 128,
                           path: str = "sample_style.png") -> str:
    """Create a colourful striped pattern image as style sample.

    Args:
        size: Image width and height in pixels.
        path: File path to save the generated image.

    Returns:
        Path to the saved image.
    """
    if os.path.exists(path):
        return path
    from PIL import Image
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    colors = [
        (255, 87,  34),   # deep orange
        (33,  150, 243),  # blue
        (76,  175,  80),  # green
        (255, 193,   7),  # amber
        (156,  39, 176),  # purple
    ]
    stripe_w = size // len(colors)
    for i, color in enumerate(colors):
        arr[:, i * stripe_w: (i + 1) * stripe_w] = color
    # Overlay a diagonal chequerboard for texture
    for r in range(size):
        for c in range(size):
            if (r + c) % 20 < 5:
                ci = (r // max(stripe_w, 1)) % len(colors)
                arr[r, c] = np.array(colors[ci]) // 2
    Image.fromarray(arr).save(path)
    print(f"  Generated synthetic style image   -> {path}")
    return path


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def load_image(path: str, max_dim: int = 256) -> np.ndarray:
    """Load and resize an image, returning a float32 (1, H, W, 3) array.

    Args:
        path:    Path to the image file.
        max_dim: Maximum dimension (height or width) after resizing.

    Returns:
        Float32 numpy array with shape (1, H, W, 3), pixel values [0, 255].
    """
    from PIL import Image
    img = Image.open(path).convert("RGB")
    scale    = max_dim / max(img.size)
    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
    img      = img.resize(new_size, Image.LANCZOS)
    return np.array(img, dtype=np.float32)[np.newaxis]   # add batch dim


def save_image(array: np.ndarray, path: str):
    """Save a float32 (1, H, W, 3) array as an image file.

    Args:
        array: Pixel array with values in [0, 255].
        path:  Output file path.
    """
    from PIL import Image
    img_arr = np.clip(array[0], 0, 255).astype(np.uint8)
    Image.fromarray(img_arr).save(path)
    print(f"  Stylised image saved -> {path}")


# ---------------------------------------------------------------------------
# VGG19 feature extractor
# ---------------------------------------------------------------------------

CONTENT_LAYERS = ["block5_conv2"]
STYLE_LAYERS   = [
    "block1_conv1", "block2_conv1",
    "block3_conv1", "block4_conv1", "block5_conv1",
]


def build_extractor(tf):
    """Build a Keras model returning intermediate VGG19 feature maps.

    Args:
        tf: The tensorflow module.

    Returns:
        (model, layer_names) where layer_names is STYLE_LAYERS + CONTENT_LAYERS.
    """
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False
    layer_names = STYLE_LAYERS + CONTENT_LAYERS
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model   = tf.keras.Model(inputs=vgg.input, outputs=outputs)
    return model, layer_names


# ---------------------------------------------------------------------------
# Preprocessing / deprocessing (VGG19 expects BGR + ImageNet mean subtracted)
# ---------------------------------------------------------------------------

def vgg_preprocess(image: "tf.Variable") -> "tf.Tensor":
    """Apply VGG19 preprocessing: convert RGB -> BGR and subtract mean.

    Args:
        image: Float32 tensor of shape (1, H, W, 3) with values in [0, 255].

    Returns:
        Preprocessed tensor suitable for VGG19.
    """
    import tensorflow as tf
    # Flip RGB to BGR
    bgr  = image[..., ::-1]
    # Subtract ImageNet per-channel mean (BGR order)
    mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
    return bgr - mean


def vgg_deprocess(array: np.ndarray) -> np.ndarray:
    """Reverse VGG19 preprocessing to obtain a displayable RGB array.

    Args:
        array: Float32 (1, H, W, 3) preprocessed array.

    Returns:
        uint8 (1, H, W, 3) RGB array with values in [0, 255].
    """
    img = array.copy()
    img[:, :, :, 0] += 103.939   # B
    img[:, :, :, 1] += 116.779   # G
    img[:, :, :, 2] += 123.68    # R
    img = img[:, :, :, ::-1]     # BGR -> RGB
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def gram_matrix(tf, feature_map: "tf.Tensor") -> "tf.Tensor":
    """Compute the Gram matrix of a feature map (style representation).

    Args:
        tf:          The tensorflow module.
        feature_map: Tensor of shape (batch, H, W, C).

    Returns:
        Gram matrix of shape (batch, C, C), normalised by H*W*C.
    """
    shape = tf.shape(feature_map)
    b, H, W, C = shape[0], shape[1], shape[2], shape[3]
    F = tf.reshape(feature_map, [b, H * W, C])
    return tf.matmul(F, F, transpose_a=True) / tf.cast(H * W * C, tf.float32)


# ---------------------------------------------------------------------------
# Style transfer optimisation
# ---------------------------------------------------------------------------

def style_transfer(
    content_path: str,
    style_path: str,
    output_path: str = "stylised_output.png",
    iterations: int = 200,
    content_weight: float = 1e4,
    style_weight: float = 1e-2,
    image_size: int = 256,
):
    """Run neural style transfer and save the result.

    Args:
        content_path:   Path to the content image.
        style_path:     Path to the style image.
        output_path:    Where to save the stylised output.
        iterations:     Number of optimisation steps.
        content_weight: Weight (alpha) for the content loss term.
        style_weight:   Weight (beta) for the style loss term.
        image_size:     Maximum dimension used when loading images.
    """
    import tensorflow as tf

    print(f"\n  Content image : {content_path}")
    print(f"  Style image   : {style_path}")
    print(f"  Output        : {output_path}")
    print(f"  Iterations    : {iterations}")
    print(f"  Image size    : {image_size}px (max dim)")

    # Load images
    content_arr = load_image(content_path, max_dim=image_size)
    style_arr   = load_image(style_path,   max_dim=image_size)

    # Preprocess for VGG19
    content_pp = vgg_preprocess(tf.constant(content_arr))
    style_pp   = vgg_preprocess(tf.constant(style_arr))

    # Build feature extractor
    model, layer_names = build_extractor(tf)

    # Pre-compute target feature maps
    content_features = {
        name: out
        for name, out in zip(layer_names, model(content_pp))
    }
    style_features = {
        name: out
        for name, out in zip(layer_names, model(style_pp))
    }

    # Pre-compute style Gram matrices (constant during optimisation)
    style_grams = {
        layer: gram_matrix(tf, style_features[layer])
        for layer in STYLE_LAYERS
    }

    # Initialise generated image as the preprocessed content image
    generated = tf.Variable(content_pp.numpy(), dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5.0)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            gen_features = {
                name: out
                for name, out in zip(layer_names, model(generated))
            }

            # Content loss
            c_loss = tf.add_n([
                tf.reduce_mean(
                    (gen_features[layer] - content_features[layer]) ** 2
                )
                for layer in CONTENT_LAYERS
            ]) / len(CONTENT_LAYERS)

            # Style loss
            s_loss = tf.add_n([
                tf.reduce_mean(
                    (gram_matrix(tf, gen_features[layer]) - style_grams[layer]) ** 2
                )
                for layer in STYLE_LAYERS
            ]) / len(STYLE_LAYERS)

            total = content_weight * c_loss + style_weight * s_loss

        grad = tape.gradient(total, generated)
        optimizer.apply_gradients([(grad, generated)])
        return total, c_loss, s_loss

    print("\n  Starting optimisation...\n")
    for i in range(iterations):
        total, c_loss, s_loss = train_step()
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Step {i + 1:4d}/{iterations} | "
                  f"Total={float(total):.2f}  "
                  f"Content={float(c_loss):.4f}  "
                  f"Style={float(s_loss):.6f}")

    # Deprocess and save
    result = vgg_deprocess(generated.numpy())
    save_image(result, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Parse CLI arguments and run neural style transfer."""
    parser = argparse.ArgumentParser(
        description="Neural Style Transfer using VGG19 (TensorFlow/Keras)"
    )
    parser.add_argument("--content",    default=None,
                        help="Path to content image (default: auto-generated)")
    parser.add_argument("--style",      default=None,
                        help="Path to style image   (default: auto-generated)")
    parser.add_argument("--output",     default="stylised_output.png",
                        help="Output image path (default: stylised_output.png)")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Optimisation steps (default: 200)")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Max image dimension in pixels (default: 256)")
    args = parser.parse_args()

    print("--- Neural Style Transfer ---")

    # Use synthetic demo images if no paths are provided
    img_size     = args.image_size
    content_path = args.content or _generate_content_image(size=img_size)
    style_path   = args.style   or _generate_style_image(size=img_size)

    style_transfer(
        content_path  = content_path,
        style_path    = style_path,
        output_path   = args.output,
        iterations    = args.iterations,
        image_size    = img_size,
    )


if __name__ == "__main__":
    main()
