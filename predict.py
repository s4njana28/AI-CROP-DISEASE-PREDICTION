"""
src/predict.py
"""
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import json
import cv2

# ---------------------------
# Paths
# ---------------------------
MODEL_PATH  = os.path.join("model", "crop_disease_model.h5")
LABELS_PATH = os.path.join("model", "class_labels.json")

# ---------------------------
# Load Model & Labels once
# ---------------------------
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    class_labels = json.load(f)

# ---------------------------
# Predict Function
# ---------------------------
def predict_disease(image: Image.Image):

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = str(np.argmax(prediction))
    confidence = np.max(prediction) * 100
    predicted_class = class_labels[predicted_index]

    return predicted_class, confidence, int(predicted_index)

# ---------------------------
# Grad-CAM Function
# ---------------------------
def generate_gradcam(image: Image.Image, predicted_index: int):

    # Preprocess image
    img = image.resize((224, 224))
    img_array = tf.cast(np.array(img) / 255.0, tf.float32)
    img_array = tf.expand_dims(img_array, axis=0)

    # Find last conv layer name in MobileNetV2
    last_conv_layer_name = None
    for layer in model.layers[0].layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name

    # Build grad model using layer names
    base_model = model.layers[0]  # MobileNetV2
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    # Create two separate models
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )

    # Get gradients
    with tf.GradientTape() as tape:
        conv_outputs, base_predictions = grad_model(img_array)
        final_output = model.layers[1](conv_outputs)  # GlobalAveragePooling
        for layer in model.layers[2:]:               # Remaining layers
            final_output = layer(final_output)
        loss = final_output[:, predicted_index]

    grads = tape.gradient(loss, conv_outputs)

    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight conv outputs
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize and colorize
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )

    # Overlay on original
    original = np.array(image.resize((224, 224)))
    overlayed = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

    return Image.fromarray(overlayed)
