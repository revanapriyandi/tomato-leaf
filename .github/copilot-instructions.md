# GitHub Copilot Instructions - Tomato Disease Classifier

## Repository Context

**Project**: Tomato Disease Detection System  
**Framework**: TensorFlow/Keras 3  
**Model**: DenseNet121 Transfer Learning  
**Purpose**: Classify 10 tomato leaf diseases  
**Training Environment**: Kaggle Notebooks

---

## Code Guidelines

### File Structure & Patterns

#### Main Training File: `tomato_classifier.py`

- **Purpose**: Complete training pipeline with all phases
- **Key Pattern**: Two-phase training (feature extraction → fine-tuning)
- **Exports**: Keras models + TFLite
- **Key Functions**:
  ```python
  create_datasets()      # Load train/val/test from Kaggle
  build_model()         # DenseNet121 + custom head
  train_model()         # Single training phase
  evaluate_model()      # Comprehensive metrics
  convert_to_tflite()   # Export to mobile format
  ```

#### Inference File: `inference.py`

- **Purpose**: Single/batch predictions on new images
- **Class**: `TomatoDiseasePredictor`
- **Methods**:
  - `predict_single(image_path)` - One image
  - `predict_batch(image_dir)` - Multiple images
  - `preprocess_image()` - Image preprocessing
- **Output Format**: JSON with confidence scores

### Python Patterns to Follow

**1. Type Hints (Always)**

```python
def create_datasets(dataset_path: str = '.') -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
```

**2. Docstrings (Google Format)**

```python
def train_model(model: keras.Model, train_ds: tf.data.Dataset) -> Dict:
    """
    Train model with callbacks and checkpointing.

    Args:
        model: Compiled Keras model
        train_ds: Training dataset

    Returns:
        Training history object

    Raises:
        ValueError: If dataset is empty
    """
```

**3. Configuration as Constants (Top of file)**

```python
IMG_SIZE = 224
BATCH_SIZE = 32
INITIAL_LR = 0.001
OOD_THRESHOLD = 0.7
```

**4. Error Handling**

```python
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
```

**5. Logging with Print Statements**

```python
print(f"\n{'='*60}")
print("TRAINING STARTED")
print(f"{'='*60}\n")
```

---

## Keras 3 Specific Patterns

### Model Building

```python
# Define using Functional API
inputs = keras.Input(shape=(224, 224, 3))
x = layers.RandomFlip("horizontal")(inputs)
x = keras.applications.densenet.preprocess_input(x)
base_model = keras.applications.DenseNet121(weights='imagenet', include_top=False)
x = base_model(x, training=False)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs, outputs)
```

### Callbacks Pattern

```python
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10),
    keras.callbacks.ModelCheckpoint('best.keras', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5)
]
```

### Compilation & Training

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(train_ds, validation_data=val_ds, callbacks=callbacks)
```

---

## Kaggle Integration Patterns

### Dataset Loading (Kaggle Paths)

```python
dataset_path = '/kaggle/input/tomato-dataset'
train_dir = os.path.join(dataset_path, 'train')

train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    seed=42,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)
```

### GPU Configuration

```python
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU Available: {len(gpus) > 0}")

# For mixed precision (optional)
policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)
```

### Output Saving (Kaggle Working Directory)

```python
model.save('final_model.keras')
plt.savefig('outputs/plot.png', dpi=300, bbox_inches='tight')
```

---

## Common Development Tasks

### Task: Add New Model Architecture

**Pattern**:

```python
def build_model_efficientnet(num_classes: int) -> keras.Model:
    """Build EfficientNetB3 model (alternative to DenseNet)"""
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = keras.applications.efficientnet.preprocess_input(x)

    base_model = keras.applications.EfficientNetB3(
        weights='imagenet',
        include_top=False
    )
    # ... rest of model
    return model
```

### Task: Add Custom Metric

**Pattern**:

```python
class F1Score(keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize variables

    def update_state(self, y_true, y_pred):
        # Update calculation
        pass

    def result(self):
        # Return metric value
        pass
```

### Task: Modify Training Loop

**Pattern**:

```python
def train_model_custom(model, train_ds, val_ds, epochs):
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = keras.losses.CategoricalCrossentropy()

    for epoch in range(epochs):
        for x, y in train_ds:
            with tf.GradientTape() as tape:
                predictions = model(x)
                loss = loss_fn(y, predictions)
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```

### Task: Add Data Augmentation

**Pattern**:

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.15),
    # Add new augmentation:
    layers.RandomTranslation(0.1, 0.1),
])
```

---

## Configuration Management

### Using config.yaml

```python
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

IMG_SIZE = config['dataset']['img_size']
BATCH_SIZE = config['dataset']['batch_size']
LR = config['training']['phase1']['learning_rate']
```

### When to Update config.yaml

- [ ] Changing hyperparameters
- [ ] Adding new configuration options
- [ ] Documenting parameter choices

---

## Testing & Validation

### Quick Test Run

```bash
python tomato_classifier.py /path/to/dataset --dry-run
# Runs 1 epoch each phase for validation
```

### Kaggle Notebook Test

```python
%run tomato_classifier.py /kaggle/input/tomato-dataset --dry-run
```

### Validation Checklist

- [ ] Dataset loads correctly
- [ ] Model compiles without errors
- [ ] Training starts and progresses
- [ ] Evaluation completes
- [ ] Visualizations generate
- [ ] Models save successfully

---

## Performance Optimization

### Data Pipeline

```python
# Always use AUTOTUNE and cache
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

### Model Optimization

```python
# For TFLite export with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
```

### Monitoring Performance

```python
# Check parameter counts
total_params = model.count_params()
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"Total: {total_params}, Trainable: {trainable_params}")
```

---

## Documentation Requirements

### For New Features

- [ ] Add docstring with Args/Returns/Raises
- [ ] Update README.md if user-facing
- [ ] Update ROADMAP.md if future-related
- [ ] Add example usage in docstring

### For Bug Fixes

- [ ] Document root cause in comments
- [ ] Explain fix rationale
- [ ] Link to related issues

### For Changes to tomato_classifier.py

- [ ] Update comments in code
- [ ] Update configuration section if needed
- [ ] Update main() docstring

---

## Common Errors & Fixes

| Error                                | Cause                  | Solution                    |
| ------------------------------------ | ---------------------- | --------------------------- |
| `No such file or directory: 'train'` | Wrong dataset path     | Use correct Kaggle path     |
| `CUDA out of memory`                 | Batch size too large   | Reduce BATCH_SIZE           |
| `Module not found: keras`            | Keras not installed    | `pip install keras>=3.0`    |
| `Model architecture diagram failed`  | Graphviz not installed | Install optional dependency |

---

## Tips for Copilot Assistance

When asking Copilot to help:

1. **Mention "Keras 3"** if discussing model code
2. **Reference "DenseNet121"** for architecture questions
3. **Say "Kaggle"** when context is important
4. **Use "transfer learning"** for fine-tuning discussions
5. **Specify "TFLite"** for mobile export questions

---

## Project Standards Summary

✅ **DO:**

- Use type hints
- Add docstrings
- Include error handling
- Log progress with print statements
- Use callbacks for training
- Test before committing
- Update documentation

❌ **DON'T:**

- Hardcode paths (use arguments)
- Skip docstrings
- Ignore error handling
- Leave debug prints
- Modify random seeds casually
- Mix Keras/TF APIs inconsistently

---

Last Updated: December 2025
