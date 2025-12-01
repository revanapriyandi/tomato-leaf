# Tomato Disease Classifier - README

## 🌱 Overview

A deep learning system for classifying tomato leaf diseases using **DenseNet121** with transfer learning. This project uses Keras 3 and TensorFlow 2.13+ to detect 10 different disease classes from tomato leaf images.

## ✨ Features

✅ **Transfer Learning** - DenseNet121 pretrained on ImageNet  
✅ **Two-Phase Training** - Feature extraction + Fine-tuning  
✅ **Data Augmentation** - RandomFlip, RandomRotation, RandomZoom, RandomBrightness  
✅ **Out-of-Distribution Detection** - Confidence-based filtering  
✅ **Mobile Deployment** - TFLite quantization (float16)  
✅ **Comprehensive Evaluation** - Confusion matrix, per-class metrics, confidence analysis  
✅ **Reproducible** - Fixed random seeds for consistent results

## 🎯 Disease Classes (10)

1. **Bacterial Spot** - Bacterial infection causing dark spots
2. **Early Blight** - Fungal disease with target-like spots
3. **Healthy** - No disease present
4. **Late Blight** - Aggressive fungal disease (Phytophthora infestans)
5. **Leaf Mold** - Fungal disease on leaf undersides
6. **Septoria Leaf Spot** - Fungal disease with circular spots
7. **Spider Mites (Two-spotted)** - Pest damage causing yellowing
8. **Target Spot** - Fungal disease with concentric rings
9. **Tomato Mosaic Virus** - Viral disease causing mottling
10. **Tomato Yellow Leaf Curl Virus** - Viral disease causing leaf curl

## 🔧 Technical Stack

| Component    | Version |
| ------------ | ------- |
| Python       | 3.8+    |
| TensorFlow   | 2.13.0+ |
| Keras        | 3.0.0+  |
| NumPy        | 1.24.0+ |
| Scikit-learn | 1.3.0+  |
| Matplotlib   | 3.7.0+  |
| Seaborn      | 0.12.0+ |

## 📊 Model Architecture

```
Input (224x224x3)
    ↓
Data Augmentation (RandomFlip, Rotation, Zoom, Brightness)
    ↓
Preprocessing (DenseNet preprocess_input)
    ↓
DenseNet121 (ImageNet weights, frozen initially)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, ReLU)
    ↓
Dropout(0.3)
    ↓
Dense(10, Softmax) → Output
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset (Local)

```
dataset/
├── train/   (70% of data)
│   ├── Bacterial_spot/
│   ├── Early_blight/
│   └── ... (10 disease folders)
├── val/     (10% of data)
│   └── ... (10 disease folders)
└── test/    (20% of data)
    └── ... (10 disease folders)
```

### 3. Run Training (Kaggle)

```python
# In Kaggle notebook
%run tomato_classifier.py /kaggle/input/tomato-dataset
```

### 4. Local Training

```bash
python tomato_classifier.py /path/to/dataset
```

### 5. Dry Run (Test - 1 epoch each)

```bash
python tomato_classifier.py /path/to/dataset --dry-run
```

## 📈 Training Pipeline

### Phase 1: Feature Extraction (20 epochs)

- **Objective**: Train custom head with frozen base model
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Checkpoint**: `best_model_feature_extraction.keras`

### Phase 2: Fine-Tuning (50 epochs)

- **Objective**: Fine-tune last 100 DenseNet layers with low learning rate
- **Learning Rate**: 0.00001
- **Unfrozen Layers**: Last 100 layers of DenseNet121
- **Checkpoint**: `best_model_fine_tuning.keras`

### Callbacks

- **EarlyStopping**: Patience=10 on validation accuracy
- **ModelCheckpoint**: Save best model based on val_accuracy
- **ReduceLROnPlateau**: Reduce LR if val_loss plateaus (factor=0.5, patience=5)

## 📁 Project Structure

```
tomato-classifier/
├── tomato_classifier.py         # Main training script
├── inference.py                 # Inference/prediction script
├── requirements.txt             # Dependencies
├── config.yaml                  # Configuration file
├── README.md                    # This file
├── QUICKSTART.md               # Quick start guide
├── CONTRIBUTING.md             # Contribution guidelines
├── ROADMAP.md                  # Future improvements
├── .copilot-instructions.md    # GitHub Copilot instructions
│
├── train/                       # Training dataset (70%)
│   └── [10 disease folders]
├── val/                         # Validation dataset (10%)
│   └── [10 disease folders]
├── test/                        # Test dataset (20%)
│   └── [10 disease folders]
│
└── outputs/
    ├── best_model_feature_extraction.keras
    ├── best_model_fine_tuning.keras
    ├── final_model.keras
    ├── tomato_disease_model.tflite
    ├── training_history_*.png
    ├── confusion_matrix.png
    ├── per_class_metrics.png
    ├── confidence_distribution.png
    └── model_architecture.png
```

## ⚙️ Configuration Parameters

Edit in `tomato_classifier.py`:

```python
IMG_SIZE = 224                  # Input image size
BATCH_SIZE = 32                 # Batch size for training
INITIAL_LR = 0.001              # Feature extraction learning rate
FINE_TUNE_LR = 0.00001          # Fine-tuning learning rate
OOD_THRESHOLD = 0.7             # Confidence threshold for OOD detection
```

Or in `config.yaml` for centralized management.

## 📊 Expected Performance

Based on typical DenseNet121 + fine-tuning:

| Metric             | Typical Value |
| ------------------ | ------------- |
| Test Accuracy      | ~95%+         |
| Per-class F1-score | 0.92-0.98     |
| Average Precision  | ~0.96         |
| Average Recall     | ~0.96         |

_Note: Actual values depend on dataset quality and augmentation parameters_

## 📤 Model Export

### TFLite Conversion

Automatically performed during training:

```bash
tomato_disease_model.tflite  # ~45-50 MB (float16 quantized)
```

**For Mobile Deployment:**

- Android: Use TFLite Support Library
- iOS: Use TensorFlow Lite for iOS
- File size: ~50% of original after quantization

## 🔍 Evaluation Metrics

The training script generates:

1. **Overall Metrics**

   - Accuracy
   - Precision, Recall, F1-score

2. **Per-Class Metrics**

   - Per-disease accuracy
   - Per-disease precision/recall/F1

3. **Visualizations**

   - Training curves (loss & accuracy)
   - Confusion matrix (normalized)
   - Per-class performance bar chart
   - Confidence distribution histogram
   - Model architecture diagram

4. **OOD Detection Analysis**
   - Confidence statistics
   - Below-threshold predictions

## 💻 Usage Examples

### Training from Kaggle

```python
# Kaggle notebook cell
!pip install -q tensorflow keras scikit-learn
%run tomato_classifier.py /kaggle/input/tomato-dataset
```

### Single Image Prediction

```python
from inference import TomatoDiseasePredictor
import keras

# Load model
predictor = TomatoDiseasePredictor('final_model.keras')

# Predict
result = predictor.predict_single('path/to/image.jpg')
predictor.print_result(result)
```

### Batch Prediction

```python
from inference import TomatoDiseasePredictor

predictor = TomatoDiseasePredictor('final_model.keras')
results = predictor.predict_batch('path/to/image/directory')

# Access results
for result in results:
    print(f"{result['image']}: {result['predicted_class']} ({result['confidence']:.2%})")
```

### Using Command Line

```bash
# Single image
python inference.py --model final_model.keras --image path/to/image.jpg

# Directory of images
python inference.py --model final_model.keras --directory path/to/images --output results.json
```

## 🐛 Troubleshooting

### GPU Not Detected

```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")
```

### Out of Memory

Reduce batch size in `tomato_classifier.py`:

```python
BATCH_SIZE = 16  # or 8
```

### Slow Training

- Enable GPU acceleration in Kaggle notebook settings
- Reduce number of epochs for testing
- Use `--dry-run` flag for quick test

### Dataset Not Found

Verify dataset structure:

```bash
python -c "from tomato_classifier import check_dataset_path; check_dataset_path('path/to/dataset')"
```

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Quick setup and first run
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute
- **[ROADMAP.md](ROADMAP.md)** - Future improvements
- **[.copilot-instructions.md](.copilot-instructions.md)** - GitHub Copilot guidelines

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Code style and conventions
- Development setup
- Making changes and testing
- Pull request process

## 🗺️ Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features:

- v1.1: Model ensemble, TFLite optimization
- v1.2: Web/mobile deployment
- v1.3: Explainability, active learning
- v2.0: Vision Transformers, advanced architectures

## 📞 Support

For issues or questions:

1. Check [QUICKSTART.md](QUICKSTART.md)
2. Review troubleshooting section above
3. Open an issue on GitHub

## 🙏 Acknowledgments

- **DenseNet121**: [Huang et al., 2016](https://arxiv.org/abs/1608.06993)
- **Tomato Dataset**: [Kaggle Tomato Disease Dataset](https://www.kaggle.com/datasets)
- **TensorFlow & Keras**: [Google AI](https://www.tensorflow.org)

**Last Updated**: December 2025  
**Status**: Production Ready ✅  
**Framework**: Keras 3 + TensorFlow 2.13+
