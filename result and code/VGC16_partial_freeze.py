import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns

def plot_training_history(history, model_name):
    """Creates comprehensive training history plots."""
    plt.figure(figsize=(15, 10))

    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Training', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation', marker='o')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Training', marker='o')
    plt.plot(history.history['val_loss'], label='Validation', marker='o')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot precision
    if 'precision' in history.history:
        plt.subplot(2, 2, 3)
        plt.plot(history.history['precision'], label='Training', marker='o')
        plt.plot(history.history['val_precision'], label='Validation', marker='o')
        plt.title('Model Precision over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)

    # Plot recall
    if 'recall' in history.history:
        plt.subplot(2, 2, 4)
        plt.plot(history.history['recall'], label='Training', marker='o')
        plt.plot(history.history['val_recall'], label='Validation', marker='o')
        plt.title('Model Recall over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    plt.close()

def plot_confusion_matrix(conf_matrix, class_names, model_name):
    """Creates and saves confusion matrix visualization."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()

def save_experiment_results(history, test_results, experiment_name, experiment_info, model, test_generator):
    """Saves comprehensive experiment results."""
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_names = list(test_generator.class_indices.keys())

    results = {
        "experiment_info": experiment_info,
        "training_metrics": {
            "best_accuracy": {
                "training": float(max(history.history['accuracy'])),
                "validation": float(max(history.history['val_accuracy'])),
                "test": float(test_results[1])
            },
            "final_loss": {
                "training": float(history.history['loss'][-1]),
                "validation": float(history.history['val_loss'][-1]),
                "test": float(test_results[0])
            },
            "best_precision": {
                "training": float(max(history.history.get('precision', [0]))),
                "validation": float(max(history.history.get('val_precision', [0])))
            },
            "best_recall": {
                "training": float(max(history.history.get('recall', [0]))),
                "validation": float(max(history.history.get('val_recall', [0])))
            },
            "final_epoch": len(history.history['accuracy']),
            "stopped_early": len(history.history['accuracy']) < experiment_info['epochs']
        },
        "per_class_metrics": {
            class_name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i])
            } for i, class_name in enumerate(class_names)
        },
        "confusion_matrix": conf_matrix.tolist(),
    }

    with open(f'{experiment_name}_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    plot_training_history(history, experiment_name)
    plot_confusion_matrix(conf_matrix, class_names, experiment_name)

    print(f"\n=== Results for {experiment_name} ===")
    print("\nAccuracy Metrics:")
    print(f"Best Training Accuracy: {results['training_metrics']['best_accuracy']['training']:.4f}")
    print(f"Best Validation Accuracy: {results['training_metrics']['best_accuracy']['validation']:.4f}")
    print(f"Test Accuracy: {results['training_metrics']['best_accuracy']['test']:.4f}")

    print("\nFinal Loss Values:")
    print(f"Training Loss: {results['training_metrics']['final_loss']['training']:.4f}")
    print(f"Validation Loss: {results['training_metrics']['final_loss']['validation']:.4f}")
    print(f"Test Loss: {results['training_metrics']['final_loss']['test']:.4f}")

    print("\nPer-Class Metrics:")
    for class_name in class_names:
        metrics = results['per_class_metrics'][class_name]
        print(f"\n{class_name}:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")

    print(f"\nTotal Epochs: {results['training_metrics']['final_epoch']}")
    if results['training_metrics']['stopped_early']:
        print("Note: Training stopped early due to early stopping criteria")

    return results

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 16  # Reduced batch size
EPOCHS = 50
LEARNING_RATE = 0.00005  # Reduced learning rate
L2_LAMBDA = 0.001  # Increased L2 regularization

# Directory paths
TRAIN_DIR = "C:\\Users\\SelmaB\\Desktop\\Plant_desease\\Train\\Train"
VALID_DIR = "C:\\Users\\SelmaB\\Desktop\\Plant_desease\\Validation\\Validation"
TEST_DIR = "C:\\Users\\SelmaB\\Desktop\\Plant_desease\\Test\\Test"

# Enhanced data generators with stronger augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    zoom_range=0.2,
    fill_mode='nearest'
)

valid_test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("Loading validation data...")
validation_generator = valid_test_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("Loading test data...")
test_generator = valid_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Create model
print("Creating model...")
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# More conservative layer unfreezing
for layer in base_model.layers:
    layer.trainable = False
# Unfreeze only the last 4 layers
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Create the complete model with increased regularization
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu',
          kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))(x)
x = Dropout(0.6)(x)
x = Dense(256, activation='relu',
          kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))(x)
x = Dropout(0.5)(x)
outputs = Dense(3, activation='softmax',
                kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA))(x)

model = tf.keras.Model(inputs, outputs)

# Compile with additional metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# Enhanced callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    min_delta=0.001
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# Train the model
print("\nStarting training...")
print(f"Training with {train_generator.samples} training images")
print(f"Validating with {validation_generator.samples} validation images")
print(f"Will train for maximum {EPOCHS} epochs")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate
print("\nEvaluating on test set...")
test_results = model.evaluate(test_generator, verbose=1)

# Save results
experiment_info = {
    'type': 'regularized_model',
    'model': 'VGG16',
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'l2_regularization': L2_LAMBDA,
    'architecture': {
        'layers': ['VGG16', 'GAP', 'Dense(512)', 'Dropout(0.6)',
                  'Dense(256)', 'Dropout(0.5)', 'Dense(3)'],
        'regularization': 'L2 + Dropout',
        'unfrozen_layers': 4
    },
    'data_augmentation': {
        'enabled': True,
        'rotation_range': 20,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'horizontal_flip': True,
        'brightness_range': [0.8, 1.2],
        'zoom_range': 0.2
    }
}

results = save_experiment_results(
    history=history,
    test_results=test_results,
    experiment_name='regularized_vgg16',
    experiment_info=experiment_info,
    model=model,
    test_generator=test_generator
)

print("\nExperiment completed! Check the generated files:")
print("1. regularized_vgg16_training_history.png - Training plots")
print("2. regularized_vgg16_results.json - Detailed metrics")