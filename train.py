"""
src/train.py
Run from project root: python src/train.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ── Paths ──────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\sanjana\Desktop\AI_crop_disease\dataset\train"
MODEL_DIR    = r"C:\Users\sanjana\Desktop\AI_crop_disease\model"
RESULTS_DIR  = r"C:\Users\sanjana\Desktop\AI_crop_disease\results"
MODEL_PATH   = os.path.join(MODEL_DIR,   "crop_disease_model.h5")
LABELS_PATH  = os.path.join(MODEL_DIR,   "class_labels.json")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 10

# ── Data generators ────────────────────────────────────────────
print("\n📂 Loading dataset from:")
print("  ", DATASET_PATH)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

NUM_CLASSES = len(train_gen.class_indices)
print(f"\n✅ Classes found  : {NUM_CLASSES}")
print(f"   Train samples  : {train_gen.samples}")
print(f"   Val   samples  : {val_gen.samples}")

# ── Save class labels ──────────────────────────────────────────
labels = {str(v): k for k, v in train_gen.class_indices.items()}
with open(LABELS_PATH, "w") as f:
    json.dump(labels, f, indent=2)

print(f"\n✅ Class labels saved → {LABELS_PATH}")
print("\n   Index : Class Name")
print("   " + "-"*45)
for k, v in sorted(labels.items(), key=lambda x: int(x[0])):
    print(f"   {int(k):>3}   : {v}")

# ── Build model ────────────────────────────────────────────────
print("\n🧠 Building MobileNetV2 model...")

base = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ── Callbacks ──────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        MODEL_PATH,
        save_best_only=True,
        monitor="val_accuracy",
        verbose=1
    ),
    EarlyStopping(
        patience=5,
        restore_best_weights=True,
        monitor="val_accuracy",
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# ── Train ──────────────────────────────────────────────────────
print("\n🚀 Training started...\n")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ── Results ────────────────────────────────────────────────────
best = max(history.history["val_accuracy"]) * 100
print("\n" + "="*50)
print("✅ Training Completed Successfully!")
print(f"   Best Val Accuracy : {best:.2f}%")
print(f"   Model saved       : {MODEL_PATH}")
print(f"   Labels saved      : {LABELS_PATH}")
print("="*50)

# ── Save training plot ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("AI Crop Disease Detection — Training History", fontsize=14, fontweight="bold")

axes[0].plot(history.history["accuracy"],     label="Train",      color="#2ecc71", linewidth=2)
axes[0].plot(history.history["val_accuracy"], label="Validation", color="#e74c3c", linewidth=2)
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_ylim([0, 1])

axes[1].plot(history.history["loss"],     label="Train",      color="#2ecc71", linewidth=2)
axes[1].plot(history.history["val_loss"], label="Validation", color="#e74c3c", linewidth=2)
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "training_history.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"   Training plot     : {plot_path}\n")