"""
train_model.py
==============
Trains a Convolutional Neural Network (CNN) on the MNIST dataset
and saves the trained model to disk for use by the GUI application.

Usage:
    python train_model.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Configuration ─────────────────────────────────────────────────────────────
EPOCHS       = 15
BATCH_SIZE   = 64
MODEL_PATH   = "mnist_cnn_model.keras"
RESULTS_DIR  = "results"


def load_and_preprocess():
    """Load MNIST, normalise to [0,1], and reshape for CNN.

    Tries keras.datasets.mnist.load_data() first (requires internet on first
    run).  Falls back to a local ``mnist_data.npz`` file placed in the same
    directory so the script works in offline / restricted-network environments.
    """
    local_npz = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "mnist_data.npz")

    try:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        print("MNIST loaded via keras datasets.")
    except Exception as exc:
        if not os.path.exists(local_npz):
            raise FileNotFoundError(
                f"Could not download MNIST ({exc}) and the local fallback "
                f"'{local_npz}' was not found.\n"
                "Please place mnist_data.npz in the project directory."
            ) from exc
        print(f"Network download unavailable ({exc}).")
        print(f"Loading MNIST from local file: {local_npz}")
        data    = np.load(local_npz)
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test  = data["x_test"]
        y_test  = data["y_test"]

    # Reshape: (N, 28, 28) → (N, 28, 28, 1)  (grayscale channel)
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test,  -1)

    print(f"Training samples : {x_train.shape[0]}")
    print(f"Test samples     : {x_test.shape[0]}")
    print(f"Image shape      : {x_train.shape[1:]}")
    return (x_train, y_train), (x_test, y_test)


def build_model(input_shape=(28, 28, 1), num_classes=10):
    """
    CNN Architecture
    ────────────────
    Block 1 : Conv2D(32) → BatchNorm → ReLU → Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout
    Block 2 : Conv2D(64) → BatchNorm → ReLU → Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout
    Head    : Flatten → Dense(128) → BatchNorm → ReLU → Dropout → Dense(10, softmax)
    """
    inputs = keras.Input(shape=input_shape)

    # ── Block 1 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # ── Block 2 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # ── Fully-connected head ──────────────────────────────────────────────────
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="MNIST_CNN")
    return model


def plot_history(history, save_dir):
    """Save accuracy and loss curves."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curves saved → {path}")


def save_results(test_loss, test_acc, save_dir):
    """Persist evaluation metrics to a text file."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "evaluation_results.txt")
    with open(path, "w") as f:
        f.write("MNIST CNN Evaluation Results\n")
        f.write("=" * 35 + "\n")
        f.write(f"Test Accuracy : {test_acc * 100:.2f}%\n")
        f.write(f"Test Loss     : {test_loss:.4f}\n")
    print(f"Results saved → {path}")


def main():
    print("\n" + "=" * 50)
    print("  MNIST CNN Training")
    print("=" * 50 + "\n")

    # 1. Data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess()

    # 2. Model
    model = build_model()
    model.summary()

    # 3. Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 4. Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH, save_best_only=True,
            monitor="val_accuracy", verbose=1
        ),
    ]

    # 5. Train
    print("\nTraining …\n")
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    # 6. Evaluate
    print("\nEvaluating on test set …")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy : {test_acc * 100:.2f}%")
    print(f"Test Loss     : {test_loss:.4f}\n")

    # 7. Persist
    plot_history(history, RESULTS_DIR)
    save_results(test_loss, test_acc, RESULTS_DIR)

    print(f"Model saved → {MODEL_PATH}")
    print("\nTraining complete!\n")


if __name__ == "__main__":
    main()
