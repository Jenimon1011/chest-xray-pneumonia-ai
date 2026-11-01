"""Grad-CAM helpers: show WHERE in the X-ray the model was looking.

Grad-CAM takes the last convolutional layer of the network, measures how much
each feature map pushed the "pneumonia" score up, and turns that into a heatmap.
"""
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib

# Name of the last convolutional activation layer in EfficientNetB0
LAST_CONV_LAYER = "top_activation"


def make_gradcam_heatmap(img_batch, model, last_conv_layer_name=LAST_CONV_LAYER):
    """Return a 2D heatmap (values 0..1) for the first image in img_batch.

    img_batch: float32 array of shape (1, H, W, 3), pixel values 0..255
    """
    grad_model = tf.keras.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_batch)
        score = preds[:, 0]  # probability of "Pneumonia"

    grads = tape.gradient(score, conv_output)            # (1, h, w, channels)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))      # importance of each channel
    heatmap = tf.reduce_sum(conv_output[0] * weights, axis=-1)  # (h, w)
    heatmap = tf.maximum(heatmap, 0)                     # keep positive evidence only
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)  # scale to 0..1
    return heatmap.numpy()


def overlay_heatmap(gray_img, heatmap, alpha=0.4):
    """Blend a heatmap over a grayscale image. Returns a uint8 RGB array."""
    h, w = gray_img.shape[:2]
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.BILINEAR)
    ) / 255.0
    colored = matplotlib.colormaps["jet"](heatmap_resized)[..., :3] * 255  # RGB
    base = np.stack([gray_img] * 3, axis=-1).astype("float32")
    blended = (1 - alpha) * base + alpha * colored
    return np.clip(blended, 0, 255).astype("uint8")
