# ============================================================
#   Neural Style Transfer using Deep Learning
#   Uses TensorFlow Hub's Pre-trained Magenta Model
# ============================================================
# HOW TO RUN:
#   1. Install required libraries:
#      pip install tensorflow tensorflow-hub pillow numpy matplotlib
#
#   2. Place your images in the same folder:
#      - content.jpg  → your photo (the image to stylize)
#      - style.jpg    → the painting style you want to apply
#
#   3. Run the program:
#      python A7.py
#
#   OUTPUT: stylized_output.jpg (saved in the same folder)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
try:
    import pkg_resources
except ImportError:
    import packaging.version
    class _MockPkgResources:
        @staticmethod
        def parse_version(v):
            return packaging.version.parse(v)
    sys.modules['pkg_resources'] = _MockPkgResources()

import tensorflow_hub as hub
from PIL import Image
import os

print("=" * 55)
print("   Neural Style Transfer - Deep Learning Practical")
print("=" * 55)


# ── STEP 1: Load and preprocess an image ─────────────────────

def load_image(image_path, max_size=512):
    """
    Load an image from disk and resize it to max_size (keeps aspect ratio).
    Returns a tensor of shape [1, H, W, 3] with values in [0, 1].
    """
    print(f"  Loading: {image_path}")
    img = Image.open(image_path).convert("RGB")

    # Resize so the largest dimension = max_size
    img.thumbnail((max_size, max_size))

    # Convert to numpy array, normalize to [0, 1]
    img_array = np.array(img) / 255.0

    # Add batch dimension: shape becomes [1, H, W, 3]
    img_tensor = tf.constant(img_array[np.newaxis, ...], dtype=tf.float32)

    return img_tensor


# ── STEP 2: Convert tensor back to a viewable image ──────────

def tensor_to_image(tensor):
    """Convert a float32 tensor [1, H, W, 3] → PIL Image."""
    img_array = tensor.numpy().squeeze()          # remove batch dim
    img_array = np.clip(img_array, 0, 1)          # clamp to [0,1]
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)


# ── STEP 3: Check that image files exist ─────────────────────

content_path = "content_image.jpg"
style_path   = "style_image.jpg"

if not os.path.exists(content_path):
    print(f"\n  [ERROR] '{content_path}' not found!")
    print("  Please place your content image (content_image.jpg) in this folder.")
    exit()

if not os.path.exists(style_path):
    print(f"\n  [ERROR] '{style_path}' not found!")
    print("  Please place your style image (style_image.jpg) in this folder.")
    exit()


# ── STEP 4: Load images ──────────────────────────────────────

content_image = load_image(content_path, max_size=512)
style_image   = load_image(style_path,   max_size=256)

print(f"\n  Content image shape : {content_image.shape}")
print(f"  Style image shape   : {style_image.shape}")


# ── STEP 5: Load the Pre-trained Style Transfer Model ────────
# Using Google Magenta's Arbitrary Image Stylization model from TF Hub.
# This model applies any artistic style to any photo in seconds.

print("\n  Loading pre-trained Neural Style Transfer model...")
print("  (This may take a moment on first run — model is downloaded)")

model_url = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
model = hub.load(model_url)

print("  Model loaded successfully!")


# ── STEP 6: Apply Style Transfer ─────────────────────────────
# The model takes (content_image, style_image) and outputs stylized image.

print("\n  Applying style transfer... please wait...")
stylized = model(tf.constant(content_image), tf.constant(style_image))[0]
print("  Style transfer complete!")


# ── STEP 7: Save the output image ────────────────────────────

output_path = "stylized_output.jpg"
result_image = tensor_to_image(stylized)
result_image.save(output_path)
print(f"\n  Output saved as: {output_path}")


# ── STEP 8: Display all 3 images side by side ────────────────

print("\n  Displaying results...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Neural Style Transfer — Deep Learning Practical", fontsize=14)

# Content Image
axes[0].imshow(content_image.numpy().squeeze())
axes[0].set_title("Content Image\n(Original Photo)")
axes[0].axis("off")

# Style Image
axes[1].imshow(style_image.numpy().squeeze())
axes[1].set_title("Style Image\n(Painting / Art)")
axes[1].axis("off")

# Stylized Output
axes[2].imshow(np.array(result_image))
axes[2].set_title("Stylized Output\n(Neural Style Transfer)")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("comparison.jpg", dpi=150, bbox_inches='tight')
plt.show()

print("\n  Comparison image saved as: comparison.jpg")
print("=" * 55)
print("  DONE! Check stylized_output.jpg for the result.")
print("=" * 55)
