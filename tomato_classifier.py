"""
Sistem Klasifikasi Penyakit Daun Tomat - DenseNet121
Fixed, Clean & Optimized Training Code (Updated for Keras 3)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
keras.utils.set_random_seed(42)
tf.random.set_seed(42)
np.random.seed(42)

# ============================================
# CONFIGURATION
# ============================================
IMG_SIZE = 224
BATCH_SIZE = 32
INITIAL_LR = 0.001
FINE_TUNE_LR = 0.00001
OOD_THRESHOLD = 0.7  # Confidence threshold for OOD detection

# ============================================
# UTILITY FUNCTIONS
# ============================================
def check_dataset_path(dataset_path):
    """Validate dataset directory structure"""
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    # Check for val directory too
    val_path = os.path.join(dataset_path, 'val')

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train directory not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test directory not found: {test_path}")

    has_val = os.path.exists(val_path)
    print(f"✓ Dataset validation passed: {dataset_path}")
    if has_val:
        print(f"✓ Validation directory found: {val_path}")
    else:
        print("ℹ Validation directory not found, will split from train.")

    return True, has_val

# ============================================
# DATASETS
# ============================================
def create_datasets(dataset_path='.'):
    """Create train, val, test datasets"""

    # Validate dataset first
    _, has_val = check_dataset_path(dataset_path)

    # Common parameters
    load_params = dict(
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    print(f"\n{'='*60}")
    print(f"LOADING DATASETS")
    print(f"{'='*60}")

    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')

    if has_val:
        val_dir = os.path.join(dataset_path, 'val')

        train_ds = keras.utils.image_dataset_from_directory(
            train_dir,
            shuffle=True,
            **load_params
        )

        val_ds = keras.utils.image_dataset_from_directory(
            val_dir,
            shuffle=False, # Validation doesn't need shuffle usually, but for metrics it is fine
            **load_params
        )
    else:
        # Split train into train/val
        train_ds = keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=0.1,
            subset="training",
            shuffle=True,
            **load_params
        )

        val_ds = keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=0.1,
            subset="validation",
            shuffle=False,
            **load_params
        )

    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        shuffle=False, # Important for evaluation to match labels
        **load_params
    )

    class_names = train_ds.class_names
    if len(class_names) == 0:
        raise ValueError("No classes found in training directory!")
    print(f"Classes detected ({len(class_names)}): {class_names}")

    # Performance optimization (Prefetching)
    # NOTE: .cache() without argument caches to memory, which can cause OOM for large datasets.
    # For large datasets, consider using .cache(filename) to cache to disk instead.
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

# ============================================
# MODEL BUILDING
# ============================================
def build_model(num_classes: int) -> keras.Model:
    """Build DenseNet121 with custom head and augmentation"""

    # Data Augmentation (Part of the model)
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.11), # ~40 degrees
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2), # Brightness
        # RandomShear is not standard in keras.layers yet without KerasCV, skipping or approximating
        # layers.RandomTranslation(0.2, 0.2) # Optional replacement for shift/shear
    ], name="data_augmentation")

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input')

    # Apply augmentation only during training
    x = data_augmentation(inputs)

    # Clip values to 0-255 range after augmentation (e.g., RandomBrightness can produce out-of-range values)
    # preprocess_input expects values in 0-255 range
    x = tf.clip_by_value(x, 0, 255)

    # Preprocess input (DenseNet expects specific scaling)
    # keras.applications.densenet.preprocess_input handles 0-255 inputs
    x = keras.applications.densenet.preprocess_input(x)

    base_model = keras.applications.DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3), # This is ignored if tensor is passed but good for init
    )
    base_model.trainable = False  # Freeze for feature extraction

    # Connect base model
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dense(256, activation='relu', name='dense_256')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = keras.Model(inputs, outputs, name='TomatoDiseaseClassifier')

    print(f"\n{'='*60}")
    print(f"MODEL ARCHITECTURE")
    print(f"{'='*60}")
    print(f"Base Model: DenseNet121 (ImageNet weights)")
    print(f"Input Shape: {IMG_SIZE}x{IMG_SIZE}x3")
    print(f"Number of Classes: {num_classes}")
    print(f"Total Layers: {len(model.layers)}")
    print(f"{'='*60}\n")

    return model

# ============================================
# TRAINING
# ============================================
def train_model(model: keras.Model, train_ds, val_ds, epochs: int = 20, lr: float = 0.001, phase: str = 'feature_extraction'):
    """Train model with callbacks"""

    checkpoint_path = f'best_model_{phase}.keras'

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print(f"\n{'='*60}")
    print(f"{phase.upper().replace('_', ' ')} - TRAINING STARTED")
    print(f"{'='*60}")
    print(f"Learning Rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n✓ {phase.upper()} completed!")
    print(f"Best model saved to: {checkpoint_path}\n")

    return history

# ============================================
# EVALUATION & VISUALIZATION
# ============================================
def evaluate_model(model: keras.Model, test_ds, class_names: list):
    """Comprehensive model evaluation"""

    print(f"\n{'='*60}")
    print("MODEL EVALUATION ON TEST SET")
    print(f"{'='*60}\n")

    # Predictions
    y_pred = model.predict(test_ds, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Extract true labels from dataset
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_true = np.argmax(y_true, axis=1) # Convert one-hot to index

    # Metrics
    accuracy = accuracy_score(y_true, y_pred_classes)
    print(f"\n{'='*60}")
    print(f"TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}\n")

    # Classification Report
    print("CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)

    # Per-class accuracy
    print("\nPER-CLASS ACCURACY:")
    print(f"{'='*60}")
    for i, class_name in enumerate(class_names):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"{class_name:30s}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    print(f"{'='*60}\n")

    return y_true, y_pred, y_pred_classes, cm

def plot_training_history(history, phase='training'):
    """Plot training curves"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2, marker='s')
    axes[0].set_title(f'Model Accuracy - {phase.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2, marker='o')
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2, marker='s')
    axes[1].set_title(f'Model Loss - {phase.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'training_history_{phase}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix heatmap"""

    plt.figure(figsize=(12, 10))

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'},
        square=True,
        linewidths=0.5
    )

    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    filename = 'confusion_matrix.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")

def plot_per_class_metrics(y_true, y_pred_classes, class_names):
    """Plot per-class precision, recall, F1-score"""

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred_classes, average=None
    )

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(x - width, precision, width, label='Precision', alpha=0.8, edgecolor='black')
    ax.bar(x, recall, width, label='Recall', alpha=0.8, edgecolor='black')
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8, edgecolor='black')

    ax.set_xlabel('Disease Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    filename = 'per_class_metrics.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")

def plot_prediction_distribution(y_pred):
    """Plot confidence distribution"""

    max_probs = np.max(y_pred, axis=1)

    plt.figure(figsize=(10, 6))
    plt.hist(max_probs, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(0.9, color='red', linestyle='--', linewidth=2, label='High Confidence (0.9)')
    plt.axvline(OOD_THRESHOLD, color='orange', linestyle='--', linewidth=2,
                label=f'OOD Threshold ({OOD_THRESHOLD})')

    # Statistics
    mean_conf = np.mean(max_probs)
    median_conf = np.median(max_probs)
    below_threshold = np.sum(max_probs < OOD_THRESHOLD)

    plt.axvline(mean_conf, color='green', linestyle=':', linewidth=2,
                label=f'Mean ({mean_conf:.3f})')

    plt.xlabel('Prediction Confidence', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add text box with statistics
    textstr = f'Mean: {mean_conf:.3f}\nMedian: {median_conf:.3f}\nBelow threshold: {below_threshold}/{len(max_probs)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    filename = 'confidence_distribution.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")

def plot_model_architecture(model, filename='model_architecture.png'):
    """Visualize model architecture"""
    try:
        keras.utils.plot_model(
            model,
            to_file=filename,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=150
        )
        print(f"✓ Model architecture saved: {filename}")
    except Exception as e:
        print(f"⚠ Could not save model architecture: {e}")

# ============================================
# OOD DETECTION
# ============================================
def analyze_ood_performance(y_pred):
    """Analyze OOD detection performance"""

    max_probs = np.max(y_pred, axis=1)

    print(f"\n{'='*60}")
    print("OUT-OF-DISTRIBUTION DETECTION ANALYSIS")
    print(f"{'='*60}")
    print(f"Threshold: {OOD_THRESHOLD}")
    print(f"Total predictions: {len(max_probs)}")
    print(f"High confidence (≥{OOD_THRESHOLD}): {np.sum(max_probs >= OOD_THRESHOLD)} ({np.sum(max_probs >= OOD_THRESHOLD)/len(max_probs)*100:.1f}%)")
    print(f"Low confidence (<{OOD_THRESHOLD}): {np.sum(max_probs < OOD_THRESHOLD)} ({np.sum(max_probs < OOD_THRESHOLD)/len(max_probs)*100:.1f}%)")
    print(f"Mean confidence: {np.mean(max_probs):.4f}")
    print(f"Median confidence: {np.median(max_probs):.4f}")
    print(f"Min confidence: {np.min(max_probs):.4f}")
    print(f"Max confidence: {np.max(max_probs):.4f}")
    print(f"{'='*60}\n")

    return OOD_THRESHOLD

# ============================================
# CONVERT TO TFLITE
# ============================================
def convert_to_tflite(model_path, output_path='tomato_disease_model.tflite'):
    """Convert best Keras model to TFLite with quantization"""

    print(f"\n{'='*60}")
    print("CONVERTING MODEL TO TFLITE")
    print(f"{'='*60}")

    # Load the best model
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)

    # Convert
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    print("Converting... (this may take a while)")
    try:
        tflite_model = converter.convert()

        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        # Size comparison
        original_size = os.path.getsize(model_path) / (1024**2)
        tflite_size = os.path.getsize(output_path) / (1024**2)

        print(f"\n{'='*60}")
        print("CONVERSION SUMMARY")
        print(f"{'='*60}")
        print(f"Original model size: {original_size:.2f} MB")
        print(f"TFLite model size: {tflite_size:.2f} MB")
        print(f"Compression ratio: {(1 - tflite_size/original_size)*100:.1f}%")
        print(f"Saved to: {output_path}")
        print(f"{'='*60}\n")
        return output_path
    except Exception as e:
        print(f"⚠ TFLite conversion failed: {e}")
        return None

# ============================================
# MAIN TRAINING PIPELINE
# ============================================
def main(dataset_path='.', dry_run=False):
    """Main training pipeline with all steps"""

    print("\n" + "="*60)
    print("TOMATO DISEASE DETECTION - TRAINING PIPELINE")
    print("DenseNet121 with Transfer Learning & Fine-Tuning")
    print("Keras 3 Implementation")
    print("="*60 + "\n")

    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU Available: {len(gpus) > 0} ({gpus})")

    # Create output directory
    os.makedirs('outputs', exist_ok=True)

    # 1. Load Data
    print("STEP 1: Loading Dataset...")
    train_ds, val_ds, test_ds, class_names = create_datasets(dataset_path)

    # 2. Build Model
    print("\nSTEP 2: Building Model...")
    model = build_model(num_classes=len(class_names))
    model.summary()

    # Optional: Save architecture diagram
    plot_model_architecture(model, 'outputs/model_architecture.png')

    epochs = 1 if dry_run else 20

    # 3. Phase 1: Feature Extraction
    print("\nSTEP 3: Phase 1 - Feature Extraction Training...")
    history_fe = train_model(
        model, train_ds, val_ds,
        epochs=epochs,
        lr=INITIAL_LR,
        phase='feature_extraction'
    )
    plot_training_history(history_fe, 'feature_extraction')

    # 4. Phase 2: Fine-Tuning
    print("\nSTEP 4: Phase 2 - Fine-Tuning...")

    # Unfreeze the base model
    # We find the DenseNet121 layer by name to avoid matching the augmentation Sequential layer
    base_model = None
    for layer in model.layers:
        if layer.name == 'densenet121':
            base_model = layer
            break

    if base_model:
        base_model.trainable = True

        # Freeze all layers except last 100
        total_layers = len(base_model.layers)
        layers_to_unfreeze = min(100, total_layers)

        print(f"Total DenseNet121 layers: {total_layers}")
        print(f"Unfreezing last {layers_to_unfreeze} layers")

        for layer in base_model.layers[:-layers_to_unfreeze]:
            layer.trainable = False
    else:
        print("⚠ Could not find base model for fine-tuning configuration. Fine-tuning all trainable layers.")


    # Count trainable parameters
    # In Keras 3, model.trainable_weights is a list of variables
    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_count = sum([tf.size(w).numpy() for w in model.non_trainable_weights])

    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Non-trainable parameters: {non_trainable_count:,}")

    epochs_ft = 1 if dry_run else 50

    # Fine-tune
    history_ft = train_model(
        model, train_ds, val_ds,
        epochs=epochs_ft,
        lr=FINE_TUNE_LR,
        phase='fine_tuning'
    )
    plot_training_history(history_ft, 'fine_tuning')

    # 5. Load Best Model for Evaluation
    print("\nSTEP 5: Loading Best Model for Evaluation...")
    best_model_path = 'best_model_fine_tuning.keras'
    if os.path.exists(best_model_path):
        model = keras.models.load_model(best_model_path)
        print(f"✓ Loaded best model from: {best_model_path}")
    else:
        print(f"⚠ Could not find {best_model_path}, using current model.")

    # 6. Evaluation
    print("\nSTEP 6: Evaluating Model on Test Set...")
    y_true, y_pred, y_pred_classes, cm = evaluate_model(model, test_ds, class_names)

    # 7. Visualizations
    print("\nSTEP 7: Creating Visualizations...")
    plot_confusion_matrix(cm, class_names)
    plot_per_class_metrics(y_true, y_pred_classes, class_names)
    plot_prediction_distribution(y_pred)

    # 8. OOD Detection Analysis
    print("\nSTEP 8: Analyzing OOD Detection...")
    ood_threshold = analyze_ood_performance(y_pred)

    # 9. Convert to TFLite
    print("\nSTEP 9: Converting to TFLite...")
    convert_to_tflite(best_model_path, 'tomato_disease_model.tflite')

    # 10. Save final model (also in .h5 for compatibility)
    print("\nSTEP 10: Saving Final Models...")
    model.save('final_model.keras')
    try:
        model.save('final_model.h5')  # Legacy format for compatibility
        print("✓ Saved: final_model.h5 (legacy format)")
    except Exception as e:
        print(f"ℹ Could not save .h5 format (might not be supported by custom layers in Keras 3): {e}")

    print("✓ Saved: final_model.keras")

    # Summary
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated Files:")
    print("  Models:")
    print("    - best_model_feature_extraction.keras")
    print("    - best_model_fine_tuning.keras")
    print("    - final_model.keras (recommended)")
    print("    - tomato_disease_model.tflite (mobile deployment)")
    print("\n  Visualizations:")
    print("    - training_history_feature_extraction.png")
    print("    - training_history_fine_tuning.png")
    print("    - confusion_matrix.png")
    print("    - per_class_metrics.png")
    print("    - confidence_distribution.png")
    print("    - outputs/model_architecture.png")
    print("\n  Configuration:")
    print(f"    - OOD Threshold: {ood_threshold}")
    print(f"    - Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"    - Number of Classes: {len(class_names)}")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Allow custom dataset path
    import sys

    # Determine default path
    default_path = '/kaggle/input/tomato-dataset'
    if not os.path.exists(default_path) and os.path.exists('train'):
        default_path = '.'

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    # Handle flags
    dry_run = '--dry-run' in sys.argv
    if dataset_path == '--dry-run':
        dataset_path = default_path

    main(dataset_path, dry_run=dry_run)
