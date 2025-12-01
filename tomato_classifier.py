"""
Sistem Klasifikasi Penyakit Daun Tomat - DenseNet121
Complete Implementation with Thesis-Ready Metrics and Visualizations
Optimized for Kaggle GPU (Tesla P100/T4) - 100% Compatible

Features:
- Comprehensive metrics: ROC-AUC, PR-AUC, Cohen's Kappa, MCC, Balanced Accuracy, etc.
- Complete visualizations: ROC curves, PR curves, confusion matrices, radar charts, etc.
- Optimized training: AdamW, label smoothing, class weights, mixed precision
- Multi-dataset evaluation: Train, Validation, Test with comparison
- Kaggle-compatible with graceful fallbacks
"""

import os
import sys
import time
import csv
import functools
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef, log_loss,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    top_k_accuracy_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
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
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 1e-4

# Maximum number of classes to display in calibration curve (for readability)
MAX_CALIBRATION_CLASSES = 5

# Whether to call plt.show() after saving plots (useful for Kaggle notebooks)
SHOW_PLOTS = True

# Dataset loading configuration
SHUFFLE_BUFFER_SIZE = 10000  # Max buffer size for shuffling (limits memory usage)
TRAIN_VAL_SPLIT_RATIO = 0.9  # Ratio for train/validation split when no val dir exists


def get_default_dataset_path() -> str:
    """Get the default dataset path, prioritizing Kaggle input paths.
    
    Returns:
        Default dataset path string.
    """
    # Kaggle-specific paths (prioritized)
    kaggle_paths = [
        '/kaggle/input/tomato-dataset',
        '/kaggle/input/tomato-leaf-disease-dataset',
        '/kaggle/input/tomato',
    ]
    
    for path in kaggle_paths:
        if os.path.exists(path):
            return path
    
    # Local development paths
    if os.path.exists('train'):
        return '.'
    
    # Default fallback
    return '/kaggle/input/tomato-dataset'


# Default configuration for notebook environments
# Users can modify these values before calling run()
DEFAULT_CONFIG = {
    'dataset_path': None,  # Will be auto-detected if None
    'epochs': 20,
    'epochs_ft': 50,
    'batch_size': 32,
    'dry_run': False,
    'use_mixed_precision': True,
}


def run(
    dataset_path: Optional[str] = None,
    epochs: int = 20,
    epochs_ft: int = 50,
    batch_size: int = 32,
    dry_run: bool = False,
    use_mixed_precision: bool = True
) -> None:
    """Run the tomato disease classification training pipeline.
    
    This is a user-friendly wrapper function designed for notebook environments
    (Kaggle, Colab, Jupyter). Users can call this function with keyword arguments
    instead of using command-line arguments.
    
    Example usage in a notebook:
        from tomato_classifier import run
        
        # Run with default settings
        run()
        
        # Run with custom settings
        run(epochs=10, epochs_ft=20, dry_run=True)
        
        # Run with custom dataset path
        run(dataset_path='/kaggle/input/my-dataset')
    
    Args:
        dataset_path: Path to dataset directory. If None, auto-detects
            from common Kaggle paths or current directory.
        epochs: Number of epochs for feature extraction phase (default: 20).
        epochs_ft: Number of epochs for fine-tuning phase (default: 50).
        batch_size: Training batch size (default: 32).
        dry_run: If True, run with 1 epoch for quick testing (default: False).
        use_mixed_precision: Whether to use mixed precision training (default: True).
    """
    # Auto-detect dataset path if not provided
    if dataset_path is None:
        dataset_path = get_default_dataset_path()
    
    print(f"Dataset path: {dataset_path}")
    
    main(
        dataset_path=dataset_path,
        dry_run=dry_run,
        epochs=epochs,
        epochs_ft=epochs_ft,
        batch_size=batch_size,
        use_mixed_precision=use_mixed_precision
    )

# Colorblind-friendly palette (extended for more classes)
COLORS = ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311',
          '#EE3377', '#BBBBBB', '#AA4499', '#999933', '#44AA99',
          '#117733', '#882255', '#332288', '#88CCEE', '#DDCC77']


def get_color(index: int) -> str:
    """Get color from palette using modulo to avoid index errors.
    
    Args:
        index: Color index
        
    Returns:
        Color hex string
    """
    return COLORS[index % len(COLORS)]

# ============================================
# DATA CLASSES FOR METRICS
# ============================================
@dataclass
class DatasetMetrics:
    """Container for all metrics computed on a dataset."""
    name: str
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    precision_weighted: float = 0.0
    recall_weighted: float = 0.0
    f1_weighted: float = 0.0
    specificity_macro: float = 0.0
    roc_auc_macro: float = 0.0
    roc_auc_micro: float = 0.0
    pr_auc_macro: float = 0.0
    cohen_kappa: float = 0.0
    mcc: float = 0.0
    log_loss_value: float = 0.0
    top1_accuracy: float = 0.0
    top3_accuracy: float = 0.0
    top5_accuracy: float = 0.0
    per_class_precision: np.ndarray = field(default_factory=lambda: np.array([]))
    per_class_recall: np.ndarray = field(default_factory=lambda: np.array([]))
    per_class_f1: np.ndarray = field(default_factory=lambda: np.array([]))
    per_class_specificity: np.ndarray = field(default_factory=lambda: np.array([]))
    per_class_roc_auc: np.ndarray = field(default_factory=lambda: np.array([]))
    per_class_pr_auc: np.ndarray = field(default_factory=lambda: np.array([]))
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    y_true: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred_proba: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class TrainingHistory:
    """Container for training history with LR tracking."""
    accuracy: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    phase: str = ""


# ============================================
# UTILITY FUNCTIONS
# ============================================
def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "2h 15m 30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_gpu_memory_info() -> str:
    """Get GPU memory usage information.
    
    Returns:
        String with GPU memory info or N/A if not available
    """
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Try to get memory info (only works with certain TF versions)
            # Use first available GPU
            gpu_device = gpus[0].name.replace('/physical_device:', '')
            info = tf.config.experimental.get_memory_info(gpu_device)
            current_mb = info['current'] / (1024**2)
            peak_mb = info['peak'] / (1024**2)
            return f"Current: {current_mb:.0f}MB, Peak: {peak_mb:.0f}MB"
    except Exception:
        pass
    return "N/A"


def setup_mixed_precision() -> bool:
    """Enable mixed precision training if available.
    
    Returns:
        True if mixed precision was enabled, False otherwise
    """
    try:
        keras.mixed_precision.set_global_policy('mixed_float16')
        print("✓ Mixed precision enabled (float16)")
        return True
    except Exception as e:
        print(f"ℹ Mixed precision not available, using float32: {e}")
        return False


def create_output_directories(base_path: str = 'outputs') -> Dict[str, str]:
    """Create output directory structure.
    
    Args:
        base_path: Base output directory
        
    Returns:
        Dictionary with paths to each subdirectory
    """
    dirs = {
        'base': base_path,
        'models': os.path.join(base_path, 'models'),
        'visualizations': os.path.join(base_path, 'visualizations'),
        'reports': os.path.join(base_path, 'reports'),
        'logs': os.path.join(base_path, 'logs', 'tensorboard')
    }
    
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    
    return dirs


def check_dataset_path(dataset_path: str) -> Tuple[bool, bool]:
    """Validate dataset directory structure.
    
    Args:
        dataset_path: Path to the dataset root directory
        
    Returns:
        Tuple of (validation_passed, has_val_directory)
        
    Raises:
        FileNotFoundError: If required directories are missing
    """
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
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

# Valid image extensions supported by TensorFlow
VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')


def has_valid_image_header(filepath: str) -> bool:
    """Check if file has valid image header (magic bytes).
    
    This is a fast preliminary check that validates magic bytes before
    attempting full TensorFlow decoding.
    
    Supported formats:
    - JPEG (FF D8)
    - PNG (89 50 4E 47 / 89PNG)
    - BMP (42 4D / BM)
    - GIF (47 49 46 38 / GIF8)
    - WebP (52 49 46 46 ... 57 45 42 50 / RIFF...WEBP)
    
    Args:
        filepath: Path to the file to validate
        
    Returns:
        True if file has valid image magic bytes, False otherwise
    """
    try:
        # Check file exists and is not empty
        if not os.path.isfile(filepath):
            return False
        if os.path.getsize(filepath) == 0:
            return False
        
        with open(filepath, 'rb') as f:
            header = f.read(12)
            if not header:
                return False
            # JPEG (FF D8)
            if header.startswith(b'\xff\xd8'):
                return True
            # PNG (89 50 4E 47)
            if header.startswith(b'\x89PNG'):
                return True
            # BMP (42 4D)
            if header.startswith(b'BM'):
                return True
            # GIF (GIF87a or GIF89a - both start with 'GIF8')
            if header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
                return True
            # WebP (RIFF....WEBP)
            if header.startswith(b'RIFF') and header[8:12] == b'WEBP':
                return True
            return False
    except (OSError, IOError):
        return False


def is_valid_image_file(filepath: str) -> Tuple[bool, str]:
    """Validate image file using TensorFlow decoding.
    
    This function performs comprehensive image validation by:
    1. First checking magic bytes (fast preliminary check)
    2. Then attempting to decode the image with TensorFlow
    
    This ensures that images with correct headers but corrupted content
    are detected and excluded during dataset scanning, preventing crashes
    during training.
    
    Args:
        filepath: Path to the file to validate
        
    Returns:
        Tuple of (is_valid, reason) where:
        - is_valid: True if image can be decoded successfully
        - reason: Description of why the image was rejected (empty string if valid)
    """
    # Fast preliminary check using magic bytes
    if not has_valid_image_header(filepath):
        return False, "invalid or missing image header"
    
    # Full TensorFlow-based validation by attempting to decode the image
    try:
        img_bytes = tf.io.read_file(filepath)
        # Use decode_image which handles JPEG, PNG, GIF, BMP
        # expand_animations=False ensures animated GIFs return single frame
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        # Verify we got valid tensor with expected shape (height, width, channels)
        # Use len() for shape validation as img.shape.rank may not be an integer
        if len(img.shape) != 3 or img.shape[-1] != 3:
            return False, "decoded image has unexpected shape"
        return True, ""
    except tf.errors.InvalidArgumentError as e:
        return False, f"TensorFlow decode error: {str(e)[:100]}"
    except Exception as e:
        return False, f"unexpected error: {str(e)[:100]}"


def get_image_files_and_labels(
    data_dir: str,
    class_names: Optional[List[str]] = None
) -> Tuple[List[str], List[int], List[str], Dict[str, int]]:
    """Scan directory for valid images and return file paths with labels.
    
    This function scans the data directory for valid images using TensorFlow-based
    validation. It filters out corrupted or invalid files and reports any
    skipped files with detailed reasons.
    
    Args:
        data_dir: Path to data directory (e.g., 'train', 'val', or 'test')
        class_names: Optional list of class names. If None, will be auto-detected
            from subdirectory names (sorted alphabetically).
            
    Returns:
        Tuple of (file_paths, labels, class_names, class_counts) where:
        - file_paths: List of valid image file paths
        - labels: List of integer labels corresponding to each file
        - class_names: List of class names (sorted alphabetically)
        - class_counts: Dictionary mapping class names to sample counts
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Auto-detect class names if not provided
    if class_names is None:
        class_names = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
        ])
    
    if len(class_names) == 0:
        raise ValueError(f"No class directories found in {data_dir}")
    
    file_paths = []
    labels = []
    class_counts = {name: 0 for name in class_names}
    skipped_files: List[Tuple[str, str]] = []  # (filepath, reason)
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for filename in os.listdir(class_dir):
            # Skip hidden files
            if filename.startswith('.'):
                continue
            
            # Check file extension
            if not filename.lower().endswith(VALID_IMAGE_EXTENSIONS):
                continue
            
            file_path = os.path.join(class_dir, filename)
            
            # Validate using TensorFlow-based decoding
            is_valid, reason = is_valid_image_file(file_path)
            if is_valid:
                file_paths.append(file_path)
                labels.append(class_idx)
                class_counts[class_name] += 1
            else:
                skipped_files.append((file_path, reason))
    
    # Report skipped files with detailed logging
    if skipped_files:
        print(f"  ⚠ Skipped {len(skipped_files)} invalid/corrupted files in {os.path.basename(data_dir)}:")
        # Show up to 5 files with their reasons
        files_to_show = min(5, len(skipped_files))
        for filepath, reason in skipped_files[:files_to_show]:
            print(f"    - {os.path.basename(filepath)}: {reason}")
        if len(skipped_files) > files_to_show:
            print(f"    ... and {len(skipped_files) - files_to_show} more")
    
    return file_paths, labels, class_names, class_counts


def process_path(file_path: tf.Tensor, label: tf.Tensor, num_classes: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Process a single file path to load and preprocess the image.
    
    This function is designed to be used with tf.data.Dataset.map().
    It reads the image file, decodes it, resizes it, and returns it
    along with the one-hot encoded label.
    
    Args:
        file_path: TensorFlow string tensor containing the file path
        label: TensorFlow int tensor containing the class label
        num_classes: Number of classes for one-hot encoding
        
    Returns:
        Tuple of (image, one_hot_label) where:
        - image: Float32 tensor of shape (IMG_SIZE, IMG_SIZE, 3) with values in [0, 255]
        - one_hot_label: Float32 tensor of shape (num_classes,)
    """
    # Read and decode image
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    
    # One-hot encode label
    one_hot_label = tf.one_hot(label, num_classes)
    
    return img, one_hot_label


def create_dataset_from_paths(
    file_paths: List[str],
    labels: List[int],
    num_classes: int,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    seed: int = 42
) -> tf.data.Dataset:
    """Create a tf.data.Dataset from validated file paths and labels.
    
    This function creates an optimized TensorFlow dataset from a list of
    validated image file paths and their corresponding labels. It handles
    batching, caching, prefetching, and optional shuffling.
    
    The dataset pipeline includes ignore_errors() to gracefully skip any
    remaining corrupted images that might have passed initial validation,
    ensuring training continues without crashes.
    
    Args:
        file_paths: List of validated image file paths
        labels: List of integer labels corresponding to each file
        num_classes: Number of classes for one-hot encoding
        batch_size: Batch size for the dataset
        shuffle: Whether to shuffle the dataset (default: True)
        seed: Random seed for shuffling (default: 42)
        
    Returns:
        Optimized tf.data.Dataset with batched (image, label) pairs
    """
    if len(file_paths) == 0:
        raise ValueError("No file paths provided for dataset creation")
    
    if len(file_paths) != len(labels):
        raise ValueError(f"Mismatch between file_paths ({len(file_paths)}) and labels ({len(labels)})")
    
    # Create dataset from file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    # Shuffle before mapping (more efficient)
    # Use a reasonable buffer size to avoid excessive memory usage for large datasets
    if shuffle:
        buffer_size = min(SHUFFLE_BUFFER_SIZE, len(file_paths))
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)
    
    # Map the processing function using functools.partial for better serialization
    process_fn = functools.partial(process_path, num_classes=num_classes)
    dataset = dataset.map(
        process_fn,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Add ignore_errors() to gracefully skip any corrupted images that might
    # have passed initial validation. This ensures training continues without
    # crashing if any edge cases slip through the validation.
    dataset = dataset.ignore_errors(log_warning=True)
    
    # Batch and optimize
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset


def create_datasets(dataset_path: str = '.', batch_size: int = BATCH_SIZE, seed: int = 42) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str], Dict[str, int]]:
    """Create train, validation, and test datasets using robust image loading.
    
    This function uses a robust data loading approach that validates each image
    using TensorFlow-based decoding before loading. This prevents crashes from
    corrupted or invalid image files in the dataset.
    
    The validation process:
    1. Checks magic bytes for quick preliminary filtering
    2. Attempts TensorFlow decoding to ensure image content is valid
    3. Logs detailed information about any skipped files
    4. Dataset pipeline includes ignore_errors() for additional safety
    
    Args:
        dataset_path: Path to the dataset root directory
        batch_size: Batch size for datasets
        seed: Random seed for shuffling and splitting (default: 42)
        
    Returns:
        Tuple of (train_ds, val_ds, test_ds, class_names, class_counts)
    """
    _, has_val = check_dataset_path(dataset_path)

    print(f"\n{'='*60}")
    print("LOADING DATASETS (Robust Mode)")
    print(f"{'='*60}")

    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')

    # Scan and validate training data
    print("\nScanning training directory...")
    train_paths, train_labels, class_names, train_class_counts = get_image_files_and_labels(train_dir)
    
    if len(class_names) == 0:
        raise ValueError("No classes found in training directory!")
    
    if len(train_paths) == 0:
        raise ValueError("No valid images found in training directory!")
    
    num_classes = len(class_names)
    
    # Handle validation data
    if has_val:
        val_dir = os.path.join(dataset_path, 'val')
        print("\nScanning validation directory...")
        val_paths, val_labels, _, _ = get_image_files_and_labels(val_dir, class_names)
        
        # Create training dataset
        train_ds = create_dataset_from_paths(
            train_paths, train_labels, num_classes, 
            batch_size=batch_size, shuffle=True, seed=seed
        )
        
        # Create validation dataset
        val_ds = create_dataset_from_paths(
            val_paths, val_labels, num_classes,
            batch_size=batch_size, shuffle=False
        )
    else:
        # Split training data for validation
        val_split_pct = int((1 - TRAIN_VAL_SPLIT_RATIO) * 100)
        print(f"\n  No validation directory found, splitting from train ({val_split_pct}%)...")
        
        # Shuffle data before splitting to ensure random split
        np.random.seed(seed)
        indices = np.random.permutation(len(train_paths))
        split_idx = int(len(indices) * TRAIN_VAL_SPLIT_RATIO)
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_paths_split = [train_paths[i] for i in train_indices]
        train_labels_split = [train_labels[i] for i in train_indices]
        val_paths_split = [train_paths[i] for i in val_indices]
        val_labels_split = [train_labels[i] for i in val_indices]
        
        # Update train_class_counts to reflect the split
        train_class_counts = {name: 0 for name in class_names}
        for lbl in train_labels_split:
            train_class_counts[class_names[lbl]] += 1
        
        # Create training dataset
        train_ds = create_dataset_from_paths(
            train_paths_split, train_labels_split, num_classes,
            batch_size=batch_size, shuffle=True, seed=seed
        )
        
        # Create validation dataset
        val_ds = create_dataset_from_paths(
            val_paths_split, val_labels_split, num_classes,
            batch_size=batch_size, shuffle=False
        )
    
    # Scan and validate test data
    print("\nScanning test directory...")
    test_paths, test_labels, _, _ = get_image_files_and_labels(test_dir, class_names)
    
    # Create test dataset
    test_ds = create_dataset_from_paths(
        test_paths, test_labels, num_classes,
        batch_size=batch_size, shuffle=False
    )
    
    print(f"\nClasses detected ({len(class_names)}): {class_names}")
    print(f"\nClass distribution (training):")
    for name, count in train_class_counts.items():
        print(f"  {name}: {count} samples")

    return train_ds, val_ds, test_ds, class_names, train_class_counts


def get_class_weights(class_counts: Dict[str, int], class_names: List[str]) -> Dict[int, float]:
    """Compute class weights for imbalanced datasets from validated class counts.
    
    Uses the class counts from validated image scanning to compute balanced
    class weights for training on imbalanced datasets.
    
    Args:
        class_counts: Dictionary mapping class names to validated sample counts
        class_names: List of class names (in order)
        
    Returns:
        Dictionary mapping class indices to weights
    """
    # Build labels list from validated class counts
    labels = []
    for class_idx, class_name in enumerate(class_names):
        count = class_counts.get(class_name, 0)
        labels.extend([class_idx] * count)
    
    if len(labels) == 0:
        return {i: 1.0 for i in range(len(class_names))}
    
    # Compute balanced weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    return dict(enumerate(class_weights))

# ============================================
# MODEL BUILDING
# ============================================
def build_model(num_classes: int) -> keras.Model:
    """Build DenseNet121 with enhanced custom head and augmentation.
    
    Features:
    - Enhanced data augmentation (horizontal/vertical flip, rotation, zoom, etc.)
    - BatchNormalization in custom head
    - Deeper custom head (512 -> 256 -> output)
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    # Enhanced Data Augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomFlip("vertical"),
        layers.RandomRotation(0.15),  # ~54 degrees
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
        layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input')

    # Apply augmentation only during training
    x = data_augmentation(inputs)

    # Clip values to 0-255 range after augmentation using Lambda layer
    x = layers.Lambda(lambda t: keras.ops.clip(t, 0, 255), name='clip_values')(x)

    # Preprocess input (DenseNet expects specific scaling)
    # Wrap in Lambda layer for Keras 3 compatibility
    x = layers.Lambda(
        lambda t: keras.applications.densenet.preprocess_input(t),
        name='preprocess'
    )(x)

    base_model = keras.applications.DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False  # Freeze for feature extraction

    # Connect base model
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Enhanced Custom Head with BatchNormalization
    x = layers.Dense(512, name='dense_512')(x)
    x = layers.BatchNormalization(name='bn_512')(x)
    x = layers.Activation('relu', name='relu_512')(x)
    x = layers.Dropout(0.4, name='dropout_1')(x)
    
    x = layers.Dense(256, name='dense_256')(x)
    x = layers.BatchNormalization(name='bn_256')(x)
    x = layers.Activation('relu', name='relu_256')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='output', dtype='float32')(x)

    model = keras.Model(inputs, outputs, name='TomatoDiseaseClassifier')

    print(f"\n{'='*60}")
    print("MODEL ARCHITECTURE")
    print(f"{'='*60}")
    print(f"Base Model: DenseNet121 (ImageNet weights)")
    print(f"Input Shape: {IMG_SIZE}x{IMG_SIZE}x3")
    print(f"Number of Classes: {num_classes}")
    print(f"Total Layers: {len(model.layers)}")
    print(f"Custom Head: Dense(512)->BN->ReLU->Dropout(0.4)->Dense(256)->BN->ReLU->Dropout(0.3)")
    print(f"Data Augmentation: Flip(H/V), Rotation, Zoom, Brightness, Contrast, Translation")
    print(f"{'='*60}\n")

    return model

# ============================================
# LEARNING RATE CALLBACK
# ============================================
class LearningRateLogger(keras.callbacks.Callback):
    """Callback to log learning rate at each epoch.
    
    Attributes:
        lr_history: List of learning rates per epoch
    """
    
    def __init__(self):
        super().__init__()
        self.lr_history = []
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Record learning rate at end of each epoch."""
        try:
            lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        except Exception:
            lr = float(self.model.optimizer.learning_rate)
        self.lr_history.append(lr)


# ============================================
# TRAINING
# ============================================
def train_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 20,
    lr: float = 0.001,
    phase: str = 'feature_extraction',
    output_dirs: Optional[Dict[str, str]] = None,
    class_weight: Optional[Dict[int, float]] = None
) -> Tuple[TrainingHistory, float]:
    """Train model with optimized settings and callbacks.
    
    Features:
    - AdamW optimizer with weight decay
    - Label smoothing
    - Class weights for imbalanced data
    - Learning rate logging
    - TensorBoard logging
    
    Args:
        model: Keras model to train
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs: Number of training epochs
        lr: Initial learning rate
        phase: Training phase name
        output_dirs: Dictionary of output directories
        class_weight: Optional class weights dictionary
        
    Returns:
        Tuple of (TrainingHistory, training_time_seconds)
    """
    if output_dirs is None:
        output_dirs = create_output_directories()
    
    checkpoint_path = os.path.join(output_dirs['models'], f'best_model_{phase}.keras')

    # Compile with AdamW optimizer and label smoothing
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=WEIGHT_DECAY,
            amsgrad=True
        ),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )

    # Learning rate logger
    lr_logger = LearningRateLogger()

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
        ),
        lr_logger
    ]
    
    # Add TensorBoard callback
    try:
        tb_log_dir = os.path.join(output_dirs['logs'], phase)
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=tb_log_dir,
                histogram_freq=1,
                write_graph=True
            )
        )
        print(f"✓ TensorBoard logging enabled: {tb_log_dir}")
    except Exception as e:
        print(f"ℹ TensorBoard not available: {e}")

    print(f"\n{'='*60}")
    print(f"{phase.upper().replace('_', ' ')} - TRAINING STARTED")
    print(f"{'='*60}")
    print(f"Learning Rate: {lr}")
    print(f"Weight Decay: {WEIGHT_DECAY}")
    print(f"Label Smoothing: {LABEL_SMOOTHING}")
    print(f"Epochs: {epochs}")
    print(f"Checkpoint: {checkpoint_path}")
    if class_weight:
        print(f"Class Weights: Enabled (balanced)")
    print(f"{'='*60}\n")

    start_time = time.time()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    training_time = time.time() - start_time

    print(f"\n✓ {phase.upper()} completed in {format_time(training_time)}")
    print(f"Best model saved to: {checkpoint_path}\n")

    # Create TrainingHistory object
    training_history = TrainingHistory(
        accuracy=history.history.get('accuracy', []),
        val_accuracy=history.history.get('val_accuracy', []),
        loss=history.history.get('loss', []),
        val_loss=history.history.get('val_loss', []),
        lr=lr_logger.lr_history,
        phase=phase
    )

    return training_history, training_time

# ============================================
# COMPREHENSIVE METRICS COMPUTATION
# ============================================
def compute_comprehensive_metrics(
    model: keras.Model,
    dataset: tf.data.Dataset,
    class_names: List[str],
    dataset_name: str = "Dataset"
) -> DatasetMetrics:
    """Compute all thesis-required metrics on a dataset.
    
    Computes:
    - Basic: Accuracy, Balanced Accuracy
    - Precision, Recall, F1 (weighted and per-class)
    - Specificity (per-class and macro)
    - ROC-AUC (per-class, micro, macro)
    - PR-AUC (per-class and macro)
    - Cohen's Kappa, MCC
    - Log Loss
    - Top-K Accuracy (1, 3, 5)
    - Confusion Matrix
    
    Args:
        model: Trained Keras model
        dataset: Dataset to evaluate
        class_names: List of class names
        dataset_name: Name for logging
        
    Returns:
        DatasetMetrics object with all computed metrics
    """
    print(f"\nComputing metrics for {dataset_name}...")
    
    # Get predictions
    y_pred_proba = model.predict(dataset, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Extract true labels
    y_true_onehot = np.concatenate([y for x, y in dataset], axis=0)
    y_true = np.argmax(y_true_onehot, axis=1)
    
    num_classes = len(class_names)
    metrics = DatasetMetrics(name=dataset_name)
    metrics.y_true = y_true
    metrics.y_pred = y_pred
    metrics.y_pred_proba = y_pred_proba
    
    # Basic metrics
    metrics.accuracy = accuracy_score(y_true, y_pred)
    metrics.balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    metrics.top1_accuracy = metrics.accuracy
    
    # Top-K accuracy
    if num_classes >= 3:
        metrics.top3_accuracy = top_k_accuracy_score(y_true, y_pred_proba, k=3)
    else:
        metrics.top3_accuracy = metrics.accuracy
        
    if num_classes >= 5:
        metrics.top5_accuracy = top_k_accuracy_score(y_true, y_pred_proba, k=5)
    else:
        metrics.top5_accuracy = metrics.top3_accuracy
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics.per_class_precision = precision
    metrics.per_class_recall = recall
    metrics.per_class_f1 = f1
    
    # Weighted averages
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics.precision_weighted = precision_w
    metrics.recall_weighted = recall_w
    metrics.f1_weighted = f1_w
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics.confusion_matrix = cm
    
    # Specificity (True Negative Rate) per class
    specificities = []
    for i in range(num_classes):
        # For class i: TN = sum of all entries except row i and column i
        # FP = sum of column i except diagonal
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)
    metrics.per_class_specificity = np.array(specificities)
    metrics.specificity_macro = np.mean(specificities)
    
    # Cohen's Kappa
    metrics.cohen_kappa = cohen_kappa_score(y_true, y_pred)
    
    # Matthews Correlation Coefficient
    metrics.mcc = matthews_corrcoef(y_true, y_pred)
    
    # Log Loss
    metrics.log_loss_value = log_loss(y_true, y_pred_proba)
    
    # ROC-AUC per class and averages
    try:
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        
        # Per-class ROC-AUC
        per_class_roc_auc = []
        for i in range(num_classes):
            if len(np.unique(y_true_bin[:, i])) > 1:
                auc_score = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
            else:
                auc_score = 0.5
            per_class_roc_auc.append(auc_score)
        metrics.per_class_roc_auc = np.array(per_class_roc_auc)
        
        # Macro and Micro ROC-AUC
        if num_classes == 2:
            metrics.roc_auc_macro = roc_auc_score(y_true, y_pred_proba[:, 1])
            metrics.roc_auc_micro = metrics.roc_auc_macro
        else:
            metrics.roc_auc_macro = roc_auc_score(y_true_bin, y_pred_proba, average='macro')
            metrics.roc_auc_micro = roc_auc_score(y_true_bin, y_pred_proba, average='micro')
    except Exception as e:
        print(f"  Warning: Could not compute ROC-AUC: {e}")
        metrics.per_class_roc_auc = np.zeros(num_classes)
        metrics.roc_auc_macro = 0.0
        metrics.roc_auc_micro = 0.0
    
    # PR-AUC per class
    try:
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        per_class_pr_auc = []
        for i in range(num_classes):
            if len(np.unique(y_true_bin[:, i])) > 1:
                ap = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
            else:
                ap = 0.0
            per_class_pr_auc.append(ap)
        metrics.per_class_pr_auc = np.array(per_class_pr_auc)
        metrics.pr_auc_macro = np.mean(per_class_pr_auc)
    except Exception as e:
        print(f"  Warning: Could not compute PR-AUC: {e}")
        metrics.per_class_pr_auc = np.zeros(num_classes)
        metrics.pr_auc_macro = 0.0
    
    print(f"  ✓ {dataset_name}: Accuracy={metrics.accuracy:.4f}, F1={metrics.f1_weighted:.4f}")
    
    return metrics


def get_confusion_details(cm: np.ndarray, class_idx: int) -> Tuple[int, int, int, int]:
    """Extract TP, TN, FP, FN for a specific class from confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_idx: Index of the class
        
    Returns:
        Tuple of (TP, TN, FP, FN)
    """
    tp = cm[class_idx, class_idx]
    fn = np.sum(cm[class_idx, :]) - tp
    fp = np.sum(cm[:, class_idx]) - tp
    tn = np.sum(cm) - tp - fn - fp
    return int(tp), int(tn), int(fp), int(fn)

def plot_training_history(history: TrainingHistory, phase: str = 'training', output_dirs: Optional[Dict[str, str]] = None) -> None:
    """Plot training curves for accuracy and loss.
    
    Args:
        history: TrainingHistory object
        phase: Training phase name
        output_dirs: Dictionary of output directories
    """
    if output_dirs is None:
        output_dirs = {'visualizations': '.'}
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axes[0].plot(history.accuracy, label='Train Accuracy', linewidth=2, marker='o', color=COLORS[0])
    axes[0].plot(history.val_accuracy, label='Val Accuracy', linewidth=2, marker='s', color=COLORS[1])
    axes[0].set_title(f'Model Accuracy - {phase.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.loss, label='Train Loss', linewidth=2, marker='o', color=COLORS[0])
    axes[1].plot(history.val_loss, label='Val Loss', linewidth=2, marker='s', color=COLORS[1])
    axes[1].set_title(f'Model Loss - {phase.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    filename = os.path.join(output_dirs['visualizations'], f'training_history_{phase}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_combined_training_history(
    history_fe: TrainingHistory,
    history_ft: TrainingHistory,
    output_dirs: Optional[Dict[str, str]] = None
) -> None:
    """Plot combined training progress with dual y-axis (Accuracy + Loss).
    
    Args:
        history_fe: Feature extraction training history
        history_ft: Fine-tuning training history
        output_dirs: Dictionary of output directories
    """
    if output_dirs is None:
        output_dirs = {'visualizations': '.'}
    
    # Combine histories
    epochs_fe = len(history_fe.accuracy)
    epochs_ft = len(history_ft.accuracy)
    
    all_train_acc = history_fe.accuracy + history_ft.accuracy
    all_val_acc = history_fe.val_accuracy + history_ft.val_accuracy
    all_train_loss = history_fe.loss + history_ft.loss
    all_val_loss = history_fe.val_loss + history_ft.val_loss
    
    epochs = range(1, len(all_train_acc) + 1)
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Accuracy on left y-axis
    color_acc = COLORS[0]
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', color=color_acc, fontsize=12)
    ax1.plot(epochs, all_train_acc, color=color_acc, linewidth=2, label='Train Accuracy', linestyle='-')
    ax1.plot(epochs, all_val_acc, color=COLORS[1], linewidth=2, label='Val Accuracy', linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim([0, 1.05])
    
    # Loss on right y-axis
    ax2 = ax1.twinx()
    color_loss = COLORS[3]
    ax2.set_ylabel('Loss', color=color_loss, fontsize=12)
    ax2.plot(epochs, all_train_loss, color=color_loss, linewidth=2, label='Train Loss', linestyle='--')
    ax2.plot(epochs, all_val_loss, color=COLORS[4], linewidth=2, label='Val Loss', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color_loss)
    
    # Add phase separator
    ax1.axvline(x=epochs_fe, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax1.text(epochs_fe + 0.5, 0.5, 'Fine-tuning starts', rotation=90, fontsize=10, alpha=0.7)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)
    
    plt.title('Combined Training Progress', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(output_dirs['visualizations'], 'training_combined.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_learning_rate_history(
    history_fe: TrainingHistory,
    history_ft: TrainingHistory,
    output_dirs: Optional[Dict[str, str]] = None
) -> None:
    """Plot learning rate changes over epochs.
    
    Args:
        history_fe: Feature extraction training history
        history_ft: Fine-tuning training history
        output_dirs: Dictionary of output directories
    """
    if output_dirs is None:
        output_dirs = {'visualizations': '.'}
    
    all_lr = history_fe.lr + history_ft.lr
    epochs = range(1, len(all_lr) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, all_lr, linewidth=2, marker='o', markersize=3, color=COLORS[0])
    plt.axvline(x=len(history_fe.lr), color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Fine-tuning starts')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate History', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    filename = os.path.join(output_dirs['visualizations'], 'learning_rate_history.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_dirs: Optional[Dict[str, str]] = None,
    normalized: bool = True
) -> None:
    """Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_dirs: Dictionary of output directories
        normalized: Whether to normalize the confusion matrix
    """
    if output_dirs is None:
        output_dirs = {'visualizations': '.'}
    
    plt.figure(figsize=(12, 10))
    
    if normalized:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        suffix = 'normalized'
        cbar_label = 'Percentage'
    else:
        cm_display = cm
        fmt = 'd'
        suffix = 'counts'
        cbar_label = 'Count'
    
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': cbar_label},
        square=True,
        linewidths=0.5
    )
    
    title = f'Confusion Matrix ({"Normalized" if normalized else "Counts"})'
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    filename = os.path.join(output_dirs['visualizations'], f'confusion_matrix_{suffix}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_roc_curves(
    metrics: DatasetMetrics,
    class_names: List[str],
    output_dirs: Optional[Dict[str, str]] = None
) -> None:
    """Plot ROC curves per class with AUC values in legend.
    
    Args:
        metrics: DatasetMetrics object containing y_true and y_pred_proba
        class_names: List of class names
        output_dirs: Dictionary of output directories
    """
    if output_dirs is None:
        output_dirs = {'visualizations': '.'}
    
    num_classes = len(class_names)
    y_true_bin = label_binarize(metrics.y_true, classes=range(num_classes))
    
    plt.figure(figsize=(12, 10))
    
    for i in range(num_classes):
        if len(np.unique(y_true_bin[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], metrics.y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=get_color(i), linewidth=2,
                    label=f'{class_names[i]} (AUC={roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {metrics.name} Set\n(Macro AUC = {metrics.roc_auc_macro:.4f})', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(output_dirs['visualizations'], 'roc_curves.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_precision_recall_curves(
    metrics: DatasetMetrics,
    class_names: List[str],
    output_dirs: Optional[Dict[str, str]] = None
) -> None:
    """Plot Precision-Recall curves per class with AP values.
    
    Args:
        metrics: DatasetMetrics object
        class_names: List of class names
        output_dirs: Dictionary of output directories
    """
    if output_dirs is None:
        output_dirs = {'visualizations': '.'}
    
    num_classes = len(class_names)
    y_true_bin = label_binarize(metrics.y_true, classes=range(num_classes))
    
    plt.figure(figsize=(12, 10))
    
    for i in range(num_classes):
        if len(np.unique(y_true_bin[:, i])) > 1:
            precision_curve, recall_curve, _ = precision_recall_curve(
                y_true_bin[:, i], metrics.y_pred_proba[:, i]
            )
            ap = metrics.per_class_pr_auc[i]
            plt.plot(recall_curve, precision_curve, color=get_color(i), linewidth=2,
                    label=f'{class_names[i]} (AP={ap:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curves - {metrics.name} Set\n(Macro AP = {metrics.pr_auc_macro:.4f})',
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(output_dirs['visualizations'], 'precision_recall_curves.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"✓ Saved: {filename}")

def plot_per_class_metrics_bar(
    metrics: DatasetMetrics,
    class_names: List[str],
    output_dirs: Optional[Dict[str, str]] = None
) -> None:
    """Plot per-class precision, recall, F1-score as bar chart.
    
    Args:
        metrics: DatasetMetrics object
        class_names: List of class names
        output_dirs: Dictionary of output directories
    """
    if output_dirs is None:
        output_dirs = {'visualizations': '.'}

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(x - width, metrics.per_class_precision, width, label='Precision', 
           alpha=0.8, edgecolor='black', color=COLORS[0])
    ax.bar(x, metrics.per_class_recall, width, label='Recall', 
           alpha=0.8, edgecolor='black', color=COLORS[1])
    ax.bar(x + width, metrics.per_class_f1, width, label='F1-Score', 
           alpha=0.8, edgecolor='black', color=COLORS[2])

    ax.set_xlabel('Disease Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Class Performance Metrics - {metrics.name} Set', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    filename = os.path.join(output_dirs['visualizations'], 'per_class_metrics_bar.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_per_class_metrics_heatmap(
    metrics: DatasetMetrics,
    class_names: List[str],
    output_dirs: Optional[Dict[str, str]] = None
) -> None:
    """Plot per-class metrics as a heatmap.
    
    Args:
        metrics: DatasetMetrics object
        class_names: List of class names
        output_dirs: Dictionary of output directories
    """
    if output_dirs is None:
        output_dirs = {'visualizations': '.'}
    
    # Create metrics matrix
    metric_names = ['Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC-AUC', 'PR-AUC']
    data = np.array([
        metrics.per_class_precision,
        metrics.per_class_recall,
        metrics.per_class_f1,
        metrics.per_class_specificity,
        metrics.per_class_roc_auc,
        metrics.per_class_pr_auc
    ])
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        xticklabels=class_names,
        yticklabels=metric_names,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={'label': 'Score'}
    )
    
    plt.title(f'Per-Class Metrics Heatmap - {metrics.name} Set', fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Metric', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    filename = os.path.join(output_dirs['visualizations'], 'per_class_metrics_heatmap.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_dataset_comparison_bar(
    train_metrics: DatasetMetrics,
    val_metrics: DatasetMetrics,
    test_metrics: DatasetMetrics,
    output_dirs: Optional[Dict[str, str]] = None
) -> None:
    """Plot bar chart comparing metrics across Train/Val/Test sets.
    
    Args:
        train_metrics: Metrics for training set
        val_metrics: Metrics for validation set
        test_metrics: Metrics for test set
        output_dirs: Dictionary of output directories
    """
    if output_dirs is None:
        output_dirs = {'visualizations': '.'}
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    train_values = [train_metrics.accuracy, train_metrics.precision_weighted, 
                    train_metrics.recall_weighted, train_metrics.f1_weighted]
    val_values = [val_metrics.accuracy, val_metrics.precision_weighted,
                  val_metrics.recall_weighted, val_metrics.f1_weighted]
    test_values = [test_metrics.accuracy, test_metrics.precision_weighted,
                   test_metrics.recall_weighted, test_metrics.f1_weighted]
    
    x = np.arange(len(metrics_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, train_values, width, label='Train', color=COLORS[0], edgecolor='black')
    bars2 = ax.bar(x, val_values, width, label='Validation', color=COLORS[1], edgecolor='black')
    bars3 = ax.bar(x + width, test_values, width, label='Test', color=COLORS[2], edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Comparison - Key Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.15])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(output_dirs['visualizations'], 'dataset_comparison_bar.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_dataset_comparison_radar(
    train_metrics: DatasetMetrics,
    val_metrics: DatasetMetrics,
    test_metrics: DatasetMetrics,
    output_dirs: Optional[Dict[str, str]] = None
) -> None:
    """Plot radar/spider chart for multi-dimensional metrics comparison.
    
    Args:
        train_metrics: Metrics for training set
        val_metrics: Metrics for validation set
        test_metrics: Metrics for test set
        output_dirs: Dictionary of output directories
    """
    if output_dirs is None:
        output_dirs = {'visualizations': '.'}
    
    categories = ['Accuracy', 'Balanced\nAccuracy', 'Precision', 'Recall', 
                  'F1-Score', 'Specificity', 'ROC-AUC', "Cohen's\nKappa"]
    
    train_values = [train_metrics.accuracy, train_metrics.balanced_accuracy,
                    train_metrics.precision_weighted, train_metrics.recall_weighted,
                    train_metrics.f1_weighted, train_metrics.specificity_macro,
                    train_metrics.roc_auc_macro, train_metrics.cohen_kappa]
    
    val_values = [val_metrics.accuracy, val_metrics.balanced_accuracy,
                  val_metrics.precision_weighted, val_metrics.recall_weighted,
                  val_metrics.f1_weighted, val_metrics.specificity_macro,
                  val_metrics.roc_auc_macro, val_metrics.cohen_kappa]
    
    test_values = [test_metrics.accuracy, test_metrics.balanced_accuracy,
                   test_metrics.precision_weighted, test_metrics.recall_weighted,
                   test_metrics.f1_weighted, test_metrics.specificity_macro,
                   test_metrics.roc_auc_macro, test_metrics.cohen_kappa]
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop
    
    # Close the plots
    train_values += train_values[:1]
    val_values += val_values[:1]
    test_values += test_values[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Draw the datasets
    ax.plot(angles, train_values, 'o-', linewidth=2, label='Train', color=COLORS[0])
    ax.fill(angles, train_values, alpha=0.15, color=COLORS[0])
    
    ax.plot(angles, val_values, 'o-', linewidth=2, label='Validation', color=COLORS[1])
    ax.fill(angles, val_values, alpha=0.15, color=COLORS[1])
    
    ax.plot(angles, test_values, 'o-', linewidth=2, label='Test', color=COLORS[2])
    ax.fill(angles, test_values, alpha=0.15, color=COLORS[2])
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    
    # Set y-axis range
    ax.set_ylim(0, 1.0)
    
    plt.title('Multi-Dimensional Metrics Comparison', fontsize=14, fontweight='bold', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
    
    plt.tight_layout()
    filename = os.path.join(output_dirs['visualizations'], 'dataset_comparison_radar.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_class_distribution(
    class_counts: Dict[str, int],
    output_dirs: Optional[Dict[str, str]] = None
) -> None:
    """Plot class distribution bar chart.
    
    Args:
        class_counts: Dictionary mapping class names to sample counts
        output_dirs: Dictionary of output directories
    """
    if output_dirs is None:
        output_dirs = {'visualizations': '.'}
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    # Use get_color for safe color access
    bar_colors = [get_color(i) for i in range(len(classes))]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(classes)), counts, color=bar_colors, edgecolor='black')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
    plt.title('Class Distribution in Training Set', fontsize=14, fontweight='bold')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add total count
    plt.text(0.02, 0.98, f'Total: {sum(counts)} samples',
             transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filename = os.path.join(output_dirs['visualizations'], 'class_distribution.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_confidence_distribution(
    y_pred_proba: np.ndarray,
    output_dirs: Optional[Dict[str, str]] = None
) -> None:
    """Plot prediction confidence distribution.
    
    Args:
        y_pred_proba: Prediction probabilities
        output_dirs: Dictionary of output directories
    """
    if output_dirs is None:
        output_dirs = {'visualizations': '.'}

    max_probs = np.max(y_pred_proba, axis=1)

    plt.figure(figsize=(10, 6))
    plt.hist(max_probs, bins=50, edgecolor='black', alpha=0.7, color=COLORS[0])
    plt.axvline(0.9, color='red', linestyle='--', linewidth=2, label='High Confidence (0.9)')
    plt.axvline(OOD_THRESHOLD, color='orange', linestyle='--', linewidth=2,
                label=f'OOD Threshold ({OOD_THRESHOLD})')

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

    textstr = f'Mean: {mean_conf:.3f}\nMedian: {median_conf:.3f}\nBelow threshold: {below_threshold}/{len(max_probs)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    filename = os.path.join(output_dirs['visualizations'], 'confidence_distribution.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_calibration_curve(
    metrics: DatasetMetrics,
    class_names: List[str],
    output_dirs: Optional[Dict[str, str]] = None
) -> None:
    """Plot calibration curve (reliability diagram).
    
    Args:
        metrics: DatasetMetrics object
        class_names: List of class names
        output_dirs: Dictionary of output directories
    """
    if output_dirs is None:
        output_dirs = {'visualizations': '.'}
    
    plt.figure(figsize=(10, 8))
    
    # Plot perfectly calibrated line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    
    num_classes = len(class_names)
    y_true_bin = label_binarize(metrics.y_true, classes=range(num_classes))
    
    # Limit to MAX_CALIBRATION_CLASSES for readability
    for i in range(min(MAX_CALIBRATION_CLASSES, num_classes)):
        try:
            if len(np.unique(y_true_bin[:, i])) > 1:
                prob_true, prob_pred = calibration_curve(
                    y_true_bin[:, i], 
                    metrics.y_pred_proba[:, i], 
                    n_bins=10,
                    strategy='uniform'
                )
                plt.plot(prob_pred, prob_true, 's-', color=get_color(i),
                        label=f'{class_names[i]}', linewidth=2)
        except Exception:
            continue
    
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Curve (Reliability Diagram)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(output_dirs['visualizations'], 'calibration_curve.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"✓ Saved: {filename}")

def plot_model_architecture(model: keras.Model, output_dirs: Optional[Dict[str, str]] = None) -> None:
    """Visualize model architecture.
    
    Args:
        model: Keras model
        output_dirs: Dictionary of output directories
    """
    if output_dirs is None:
        output_dirs = {'visualizations': '.'}
    
    filename = os.path.join(output_dirs['visualizations'], 'model_architecture.png')
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
def analyze_ood_performance(y_pred_proba: np.ndarray) -> float:
    """Analyze OOD detection performance.
    
    Args:
        y_pred_proba: Prediction probabilities
        
    Returns:
        OOD threshold value
    """
    max_probs = np.max(y_pred_proba, axis=1)

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
def convert_to_tflite(
    model_path: str,
    output_dirs: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """Convert best Keras model to TFLite with quantization.
    
    Args:
        model_path: Path to Keras model file
        output_dirs: Dictionary of output directories
        
    Returns:
        Path to TFLite model or None if conversion failed
    """
    if output_dirs is None:
        output_dirs = {'models': '.'}
    
    output_path = os.path.join(output_dirs['models'], 'tomato_disease_model.tflite')

    print(f"\n{'='*60}")
    print("CONVERTING MODEL TO TFLITE")
    print(f"{'='*60}")

    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    print("Converting... (this may take a while)")
    try:
        tflite_model = converter.convert()

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

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
# REPORTS AND CONSOLE OUTPUT
# ============================================
def print_comprehensive_metrics_summary(
    train_metrics: DatasetMetrics,
    val_metrics: DatasetMetrics,
    test_metrics: DatasetMetrics,
    total_training_time: float,
    best_epoch_fe: int,
    best_epoch_ft: int,
    final_lr: float,
    total_params: int,
    trainable_params: int
) -> None:
    """Print comprehensive metrics summary in thesis format.
    
    Args:
        train_metrics: Training set metrics
        val_metrics: Validation set metrics
        test_metrics: Test set metrics
        total_training_time: Total training time in seconds
        best_epoch_fe: Best epoch for feature extraction
        best_epoch_ft: Best epoch for fine-tuning
        final_lr: Final learning rate
        total_params: Total model parameters
        trainable_params: Trainable model parameters
    """
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE METRICS SUMMARY FOR THESIS")
    print("="*60)
    
    print("\n📌 OVERALL METRICS")
    print("-"*60)
    print(f"{'Metric':<24}| {'Train':<10}| {'Validation':<10}| {'Test':<10}")
    print("-"*60)
    
    metrics_rows = [
        ("Accuracy", train_metrics.accuracy, val_metrics.accuracy, test_metrics.accuracy),
        ("Balanced Accuracy", train_metrics.balanced_accuracy, val_metrics.balanced_accuracy, test_metrics.balanced_accuracy),
        ("Precision (Weighted)", train_metrics.precision_weighted, val_metrics.precision_weighted, test_metrics.precision_weighted),
        ("Recall (Weighted)", train_metrics.recall_weighted, val_metrics.recall_weighted, test_metrics.recall_weighted),
        ("F1-Score (Weighted)", train_metrics.f1_weighted, val_metrics.f1_weighted, test_metrics.f1_weighted),
        ("Specificity (Macro)", train_metrics.specificity_macro, val_metrics.specificity_macro, test_metrics.specificity_macro),
        ("ROC-AUC (Macro)", train_metrics.roc_auc_macro, val_metrics.roc_auc_macro, test_metrics.roc_auc_macro),
        ("PR-AUC (Macro)", train_metrics.pr_auc_macro, val_metrics.pr_auc_macro, test_metrics.pr_auc_macro),
        ("Cohen's Kappa", train_metrics.cohen_kappa, val_metrics.cohen_kappa, test_metrics.cohen_kappa),
        ("MCC", train_metrics.mcc, val_metrics.mcc, test_metrics.mcc),
        ("Log Loss", train_metrics.log_loss_value, val_metrics.log_loss_value, test_metrics.log_loss_value),
        ("Top-1 Accuracy", train_metrics.top1_accuracy, val_metrics.top1_accuracy, test_metrics.top1_accuracy),
        ("Top-3 Accuracy", train_metrics.top3_accuracy, val_metrics.top3_accuracy, test_metrics.top3_accuracy),
        ("Top-5 Accuracy", train_metrics.top5_accuracy, val_metrics.top5_accuracy, test_metrics.top5_accuracy),
    ]
    
    for name, train_val, val_val, test_val in metrics_rows:
        print(f"{name:<24}| {train_val:<10.4f}| {val_val:<10.4f}| {test_val:<10.4f}")
    
    print("="*60)
    
    print("\n📌 TRAINING SUMMARY")
    print("-"*60)
    print(f"Total Training Time     : {format_time(total_training_time)}")
    print(f"Best Epoch (FE)         : {best_epoch_fe}")
    print(f"Best Epoch (FT)         : {best_epoch_ft}")
    print(f"Final Learning Rate     : {final_lr:.2e}")
    print(f"Total Parameters        : {total_params:,}")
    print(f"Trainable Parameters    : {trainable_params:,}")
    print("="*60)


def save_metrics_to_csv(
    train_metrics: DatasetMetrics,
    val_metrics: DatasetMetrics,
    test_metrics: DatasetMetrics,
    output_dirs: Dict[str, str]
) -> None:
    """Save metrics summary to CSV file.
    
    Args:
        train_metrics: Training set metrics
        val_metrics: Validation set metrics
        test_metrics: Test set metrics
        output_dirs: Dictionary of output directories
    """
    filepath = os.path.join(output_dirs['reports'], 'metrics_summary.csv')
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Train', 'Validation', 'Test'])
        
        metrics_rows = [
            ('Accuracy', train_metrics.accuracy, val_metrics.accuracy, test_metrics.accuracy),
            ('Balanced Accuracy', train_metrics.balanced_accuracy, val_metrics.balanced_accuracy, test_metrics.balanced_accuracy),
            ('Precision (Weighted)', train_metrics.precision_weighted, val_metrics.precision_weighted, test_metrics.precision_weighted),
            ('Recall (Weighted)', train_metrics.recall_weighted, val_metrics.recall_weighted, test_metrics.recall_weighted),
            ('F1-Score (Weighted)', train_metrics.f1_weighted, val_metrics.f1_weighted, test_metrics.f1_weighted),
            ('Specificity (Macro)', train_metrics.specificity_macro, val_metrics.specificity_macro, test_metrics.specificity_macro),
            ('ROC-AUC (Macro)', train_metrics.roc_auc_macro, val_metrics.roc_auc_macro, test_metrics.roc_auc_macro),
            ('ROC-AUC (Micro)', train_metrics.roc_auc_micro, val_metrics.roc_auc_micro, test_metrics.roc_auc_micro),
            ('PR-AUC (Macro)', train_metrics.pr_auc_macro, val_metrics.pr_auc_macro, test_metrics.pr_auc_macro),
            ('Cohen Kappa', train_metrics.cohen_kappa, val_metrics.cohen_kappa, test_metrics.cohen_kappa),
            ('MCC', train_metrics.mcc, val_metrics.mcc, test_metrics.mcc),
            ('Log Loss', train_metrics.log_loss_value, val_metrics.log_loss_value, test_metrics.log_loss_value),
            ('Top-1 Accuracy', train_metrics.top1_accuracy, val_metrics.top1_accuracy, test_metrics.top1_accuracy),
            ('Top-3 Accuracy', train_metrics.top3_accuracy, val_metrics.top3_accuracy, test_metrics.top3_accuracy),
            ('Top-5 Accuracy', train_metrics.top5_accuracy, val_metrics.top5_accuracy, test_metrics.top5_accuracy),
        ]
        
        for row in metrics_rows:
            writer.writerow(row)
    
    print(f"✓ Saved: {filepath}")


def save_per_class_metrics_csv(
    metrics: DatasetMetrics,
    class_names: List[str],
    output_dirs: Dict[str, str]
) -> None:
    """Save per-class metrics to CSV file.
    
    Args:
        metrics: DatasetMetrics object
        class_names: List of class names
        output_dirs: Dictionary of output directories
    """
    filepath = os.path.join(output_dirs['reports'], 'per_class_metrics.csv')
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Specificity', 
                        'ROC-AUC', 'PR-AUC', 'TP', 'TN', 'FP', 'FN'])
        
        for i, class_name in enumerate(class_names):
            tp, tn, fp, fn = get_confusion_details(metrics.confusion_matrix, i)
            writer.writerow([
                class_name,
                f"{metrics.per_class_precision[i]:.4f}",
                f"{metrics.per_class_recall[i]:.4f}",
                f"{metrics.per_class_f1[i]:.4f}",
                f"{metrics.per_class_specificity[i]:.4f}",
                f"{metrics.per_class_roc_auc[i]:.4f}",
                f"{metrics.per_class_pr_auc[i]:.4f}",
                tp, tn, fp, fn
            ])
    
    print(f"✓ Saved: {filepath}")


def save_training_history_csv(
    history_fe: TrainingHistory,
    history_ft: TrainingHistory,
    output_dirs: Dict[str, str]
) -> None:
    """Save training history to CSV file.
    
    Args:
        history_fe: Feature extraction history
        history_ft: Fine-tuning history
        output_dirs: Dictionary of output directories
    """
    filepath = os.path.join(output_dirs['reports'], 'training_history.csv')
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Phase', 'Epoch', 'Train_Accuracy', 'Val_Accuracy', 
                        'Train_Loss', 'Val_Loss', 'Learning_Rate'])
        
        # Feature extraction
        for i in range(len(history_fe.accuracy)):
            lr = history_fe.lr[i] if i < len(history_fe.lr) else 0
            writer.writerow([
                'Feature_Extraction', i + 1,
                f"{history_fe.accuracy[i]:.4f}",
                f"{history_fe.val_accuracy[i]:.4f}",
                f"{history_fe.loss[i]:.4f}",
                f"{history_fe.val_loss[i]:.4f}",
                f"{lr:.2e}"
            ])
        
        # Fine-tuning
        epoch_offset = len(history_fe.accuracy)
        for i in range(len(history_ft.accuracy)):
            lr = history_ft.lr[i] if i < len(history_ft.lr) else 0
            writer.writerow([
                'Fine_Tuning', epoch_offset + i + 1,
                f"{history_ft.accuracy[i]:.4f}",
                f"{history_ft.val_accuracy[i]:.4f}",
                f"{history_ft.loss[i]:.4f}",
                f"{history_ft.val_loss[i]:.4f}",
                f"{lr:.2e}"
            ])
    
    print(f"✓ Saved: {filepath}")


def save_classification_report_txt(
    metrics: DatasetMetrics,
    class_names: List[str],
    output_dirs: Dict[str, str]
) -> None:
    """Save classification report to text file.
    
    Args:
        metrics: DatasetMetrics object
        class_names: List of class names
        output_dirs: Dictionary of output directories
    """
    filepath = os.path.join(output_dirs['reports'], 'classification_report.txt')
    
    report = classification_report(
        metrics.y_true, 
        metrics.y_pred, 
        target_names=class_names, 
        digits=4
    )
    
    with open(filepath, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CLASSIFICATION REPORT - TEST SET\n")
        f.write("="*60 + "\n\n")
        f.write(report)
        f.write("\n\n")
        f.write("="*60 + "\n")
        f.write("ADDITIONAL METRICS\n")
        f.write("="*60 + "\n")
        f.write(f"Cohen's Kappa: {metrics.cohen_kappa:.4f}\n")
        f.write(f"Matthews Correlation Coefficient (MCC): {metrics.mcc:.4f}\n")
        f.write(f"Balanced Accuracy: {metrics.balanced_accuracy:.4f}\n")
        f.write(f"ROC-AUC (Macro): {metrics.roc_auc_macro:.4f}\n")
        f.write(f"PR-AUC (Macro): {metrics.pr_auc_macro:.4f}\n")
        f.write(f"Log Loss: {metrics.log_loss_value:.4f}\n")
    
    print(f"✓ Saved: {filepath}")


def save_model_summary_txt(
    model: keras.Model,
    output_dirs: Dict[str, str]
) -> None:
    """Save model summary to text file.
    
    Args:
        model: Keras model
        output_dirs: Dictionary of output directories
    """
    filepath = os.path.join(output_dirs['reports'], 'model_summary.txt')
    
    # Capture model summary
    string_buffer = []
    model.summary(print_fn=lambda x: string_buffer.append(x))
    
    with open(filepath, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write("\n".join(string_buffer))
    
    print(f"✓ Saved: {filepath}")


# ============================================
# MAIN TRAINING PIPELINE
# ============================================
def main(
    dataset_path: str = '.',
    dry_run: bool = False,
    epochs: int = 20,
    epochs_ft: int = 50,
    batch_size: int = 32,
    use_mixed_precision: bool = True
) -> None:
    """Main training pipeline with comprehensive metrics and visualizations.
    
    This pipeline includes:
    - AdamW optimizer with weight decay
    - Label smoothing
    - Class weights for imbalanced data
    - Mixed precision training (optional)
    - Comprehensive metrics (ROC-AUC, PR-AUC, Cohen's Kappa, MCC, etc.)
    - All required visualizations for thesis
    - Multi-dataset evaluation (Train, Val, Test)
    - CSV and TXT reports
    
    Args:
        dataset_path: Path to dataset directory
        dry_run: If True, run with 1 epoch for testing
        epochs: Number of epochs for feature extraction
        epochs_ft: Number of epochs for fine-tuning
        batch_size: Batch size for training
        use_mixed_precision: Whether to use mixed precision training
    """
    global BATCH_SIZE
    BATCH_SIZE = batch_size
    
    total_start_time = time.time()

    print("\n" + "="*60)
    print("🍅 TOMATO DISEASE DETECTION - TRAINING PIPELINE")
    print("DenseNet121 with Transfer Learning & Fine-Tuning")
    print("Complete Implementation for Thesis")
    print("="*60 + "\n")

    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU Available: {len(gpus) > 0}")
    if gpus:
        print(f"  GPUs: {gpus}")
        print(f"  GPU Memory: {get_gpu_memory_info()}")
    
    # Enable mixed precision if requested
    if use_mixed_precision:
        setup_mixed_precision()

    # Create output directory structure
    output_dirs = create_output_directories()
    print(f"✓ Output directories created: {output_dirs['base']}")

    # ========================================
    # STEP 1: Load Data
    # ========================================
    print("\n" + "="*60)
    print("STEP 1: Loading Dataset...")
    print("="*60)
    train_ds, val_ds, test_ds, class_names, class_counts = create_datasets(dataset_path, batch_size)
    
    # Compute class weights for imbalanced data using class counts from create_datasets
    class_weight = get_class_weights(class_counts, class_names)
    print(f"\nClass weights computed:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_weight[i]:.4f}")

    # ========================================
    # STEP 2: Build Model
    # ========================================
    print("\n" + "="*60)
    print("STEP 2: Building Model...")
    print("="*60)
    model = build_model(num_classes=len(class_names))
    model.summary()

    # Save model summary and architecture
    save_model_summary_txt(model, output_dirs)
    plot_model_architecture(model, output_dirs)
    
    # Plot class distribution
    plot_class_distribution(class_counts, output_dirs)

    # Adjust epochs for dry run
    if dry_run:
        epochs = 1
        epochs_ft = 1
        print("\n⚡ DRY RUN MODE: Using 1 epoch per phase")

    # ========================================
    # STEP 3: Feature Extraction Training
    # ========================================
    print("\n" + "="*60)
    print("STEP 3: Phase 1 - Feature Extraction Training...")
    print("="*60)
    history_fe, time_fe = train_model(
        model, train_ds, val_ds,
        epochs=epochs,
        lr=INITIAL_LR,
        phase='feature_extraction',
        output_dirs=output_dirs,
        class_weight=class_weight
    )
    plot_training_history(history_fe, 'feature_extraction', output_dirs)
    best_epoch_fe = len(history_fe.val_accuracy) - history_fe.val_accuracy[::-1].index(max(history_fe.val_accuracy))

    # ========================================
    # STEP 4: Fine-Tuning
    # ========================================
    print("\n" + "="*60)
    print("STEP 4: Phase 2 - Fine-Tuning...")
    print("="*60)

    # Unfreeze the base model
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
        print("⚠ Could not find base model. Fine-tuning all trainable layers.")

    # Count trainable parameters
    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_count = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_params = trainable_count + non_trainable_count

    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Non-trainable parameters: {non_trainable_count:,}")
    print(f"Total parameters: {total_params:,}")

    # Fine-tune
    history_ft, time_ft = train_model(
        model, train_ds, val_ds,
        epochs=epochs_ft,
        lr=FINE_TUNE_LR,
        phase='fine_tuning',
        output_dirs=output_dirs,
        class_weight=class_weight
    )
    plot_training_history(history_ft, 'fine_tuning', output_dirs)
    best_epoch_ft = len(history_fe.val_accuracy) + len(history_ft.val_accuracy) - history_ft.val_accuracy[::-1].index(max(history_ft.val_accuracy))
    
    # Get final learning rate
    final_lr = history_ft.lr[-1] if history_ft.lr else FINE_TUNE_LR

    # Plot combined training history
    plot_combined_training_history(history_fe, history_ft, output_dirs)
    plot_learning_rate_history(history_fe, history_ft, output_dirs)
    
    # Save training history
    save_training_history_csv(history_fe, history_ft, output_dirs)

    # ========================================
    # STEP 5: Load Best Model
    # ========================================
    print("\n" + "="*60)
    print("STEP 5: Loading Best Model for Evaluation...")
    print("="*60)
    best_model_path = os.path.join(output_dirs['models'], 'best_model_fine_tuning.keras')
    if os.path.exists(best_model_path):
        model = keras.models.load_model(best_model_path)
        print(f"✓ Loaded best model from: {best_model_path}")
    else:
        print(f"⚠ Could not find {best_model_path}, using current model.")

    # ========================================
    # STEP 6: Comprehensive Evaluation
    # ========================================
    print("\n" + "="*60)
    print("STEP 6: Computing Comprehensive Metrics...")
    print("="*60)
    
    # Evaluate on all three datasets
    train_metrics = compute_comprehensive_metrics(model, train_ds, class_names, "Train")
    val_metrics = compute_comprehensive_metrics(model, val_ds, class_names, "Validation")
    test_metrics = compute_comprehensive_metrics(model, test_ds, class_names, "Test")

    # ========================================
    # STEP 7: Generate Visualizations
    # ========================================
    print("\n" + "="*60)
    print("STEP 7: Generating Visualizations...")
    print("="*60)
    
    # Confusion matrices (normalized and counts)
    plot_confusion_matrix(test_metrics.confusion_matrix, class_names, output_dirs, normalized=True)
    plot_confusion_matrix(test_metrics.confusion_matrix, class_names, output_dirs, normalized=False)
    
    # ROC and PR curves
    plot_roc_curves(test_metrics, class_names, output_dirs)
    plot_precision_recall_curves(test_metrics, class_names, output_dirs)
    
    # Per-class metrics
    plot_per_class_metrics_bar(test_metrics, class_names, output_dirs)
    plot_per_class_metrics_heatmap(test_metrics, class_names, output_dirs)
    
    # Dataset comparison
    plot_dataset_comparison_bar(train_metrics, val_metrics, test_metrics, output_dirs)
    plot_dataset_comparison_radar(train_metrics, val_metrics, test_metrics, output_dirs)
    
    # Confidence distribution
    plot_confidence_distribution(test_metrics.y_pred_proba, output_dirs)
    
    # Calibration curve
    plot_calibration_curve(test_metrics, class_names, output_dirs)

    # ========================================
    # STEP 8: OOD Detection Analysis
    # ========================================
    print("\n" + "="*60)
    print("STEP 8: Analyzing OOD Detection...")
    print("="*60)
    analyze_ood_performance(test_metrics.y_pred_proba)

    # ========================================
    # STEP 9: Save Reports
    # ========================================
    print("\n" + "="*60)
    print("STEP 9: Saving Reports...")
    print("="*60)
    
    save_metrics_to_csv(train_metrics, val_metrics, test_metrics, output_dirs)
    save_per_class_metrics_csv(test_metrics, class_names, output_dirs)
    save_classification_report_txt(test_metrics, class_names, output_dirs)

    # ========================================
    # STEP 10: Convert to TFLite
    # ========================================
    print("\n" + "="*60)
    print("STEP 10: Converting to TFLite...")
    print("="*60)
    convert_to_tflite(best_model_path, output_dirs)

    # ========================================
    # STEP 11: Save Final Model
    # ========================================
    print("\n" + "="*60)
    print("STEP 11: Saving Final Models...")
    print("="*60)
    
    final_model_path = os.path.join(output_dirs['models'], 'final_model.keras')
    model.save(final_model_path)
    print(f"✓ Saved: {final_model_path}")
    
    try:
        h5_path = os.path.join(output_dirs['models'], 'final_model.h5')
        model.save(h5_path)
        print(f"✓ Saved: {h5_path} (legacy format)")
    except Exception as e:
        print(f"ℹ Could not save .h5 format: {e}")

    # ========================================
    # FINAL SUMMARY
    # ========================================
    total_training_time = time.time() - total_start_time
    
    print_comprehensive_metrics_summary(
        train_metrics, val_metrics, test_metrics,
        total_training_time,
        best_epoch_fe, best_epoch_ft,
        final_lr,
        total_params, trainable_count
    )

    print("\n" + "="*60)
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📁 Output Structure:")
    print(f"  {output_dirs['base']}/")
    print(f"  ├── models/")
    print(f"  │   ├── best_model_feature_extraction.keras")
    print(f"  │   ├── best_model_fine_tuning.keras")
    print(f"  │   ├── final_model.keras")
    print(f"  │   └── tomato_disease_model.tflite")
    print(f"  ├── visualizations/")
    print(f"  │   ├── roc_curves.png")
    print(f"  │   ├── precision_recall_curves.png")
    print(f"  │   ├── confusion_matrix_normalized.png")
    print(f"  │   ├── confusion_matrix_counts.png")
    print(f"  │   ├── dataset_comparison_bar.png")
    print(f"  │   ├── dataset_comparison_radar.png")
    print(f"  │   ├── per_class_metrics_bar.png")
    print(f"  │   ├── per_class_metrics_heatmap.png")
    print(f"  │   ├── training_history_*.png")
    print(f"  │   ├── training_combined.png")
    print(f"  │   ├── learning_rate_history.png")
    print(f"  │   ├── class_distribution.png")
    print(f"  │   ├── confidence_distribution.png")
    print(f"  │   ├── calibration_curve.png")
    print(f"  │   └── model_architecture.png")
    print(f"  ├── reports/")
    print(f"  │   ├── metrics_summary.csv")
    print(f"  │   ├── per_class_metrics.csv")
    print(f"  │   ├── training_history.csv")
    print(f"  │   ├── classification_report.txt")
    print(f"  │   └── model_summary.txt")
    print(f"  └── logs/tensorboard/")
    print(f"\n⏱️  Total Time: {format_time(total_training_time)}")
    print(f"🎯 Test Accuracy: {test_metrics.accuracy:.4f} ({test_metrics.accuracy*100:.2f}%)")
    print(f"📊 Test F1-Score: {test_metrics.f1_weighted:.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Start training directly with default configuration
    # To change parameters, edit DEFAULT_CONFIG above
    run(
        dataset_path=DEFAULT_CONFIG['dataset_path'],
        epochs=DEFAULT_CONFIG['epochs'],
        epochs_ft=DEFAULT_CONFIG['epochs_ft'],
        batch_size=DEFAULT_CONFIG['batch_size'],
        dry_run=DEFAULT_CONFIG['dry_run'],
        use_mixed_precision=DEFAULT_CONFIG['use_mixed_precision'],
    )
