# -*- coding: utf-8 -*-
"""image_classification_vgg16.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam, RMSprop
import wandb
from wandb.integration.keras import WandbCallback

# Mount Google Drive if in Colab
try:
    from google.colab import drive
    mount = '/content/drive/'
    drive.mount(mount)
    drive_root = mount + "/My Drive/Colab Notebooks/purcell/throat"
    print("\nColab: Changing directory to ", drive_root)
    # Change to your drive directory
    # %cd $drive_root
    IN_COLAB = True
except:
    mount = './'
    drive_root = './'
    IN_COLAB = False

# Setup directory structure
base_dir = "./"
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_strep_dir = os.path.join(train_dir, 'pharyngitis')  # Directory with training unhealthy pictures
train_healthy_dir = os.path.join(train_dir, 'no_pharyngitis')  # Directory with training healthy pictures
validation_strep_dir = os.path.join(validation_dir, 'pharyngitis')  # Directory with validation unhealthy pictures
validation_healthy_dir = os.path.join(validation_dir, 'no_pharyngitis')  # Directory with validation healthy pictures

# List files in each directory
train_strep_fnames = os.listdir(train_strep_dir)
train_healthy_fnames = os.listdir(train_healthy_dir)
validation_strep_fnames = os.listdir(validation_strep_dir)
validation_healthy_fnames = os.listdir(validation_healthy_dir)

# Set up data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.,
    rotation_range=40,
    horizontal_flip=True
)

# No augmentation for validation data
test_datagen = ImageDataGenerator(rescale=1.0/255.)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size=20,
    shuffle=True,  # Changed to True for better training
    class_mode='categorical',
    target_size=(256, 256)
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    batch_size=10,
    shuffle=False,
    class_mode='categorical',
    target_size=(256, 256)
)

# Define the model architecture
def getbase_model(inp):
    """Returns a VGG16 model with the given input tensor"""
    vgg = tf.keras.applications.VGG16(
        include_top=False, 
        weights='imagenet', 
        input_tensor=inp,
        input_shape=(256, 256, 3)
    )
    vgg.trainable = False
    return vgg

def myvggmodel():
    """Creates a custom VGG-based model for pharyngitis classification"""
    inp = keras.layers.Input(shape=(256, 256, 3))
    vgg = getbase_model(inp)
    
    # Get the output of the last pooling layer
    x = vgg.get_layer('block5_pool').output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    output = keras.layers.Dense(2, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=inp, outputs=output)
    return model

# Clear session and create model
keras.backend.clear_session()
model = myvggmodel()
model.summary()

# Print layer names and trainable status
for i, layer in enumerate(model.layers, 1):
    print(i, layer.name, "-", layer.trainable)

# Compile the model with appropriate optimizer and loss
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc']
)

# Create GradCAM for visualization
class GradCAM:
    """
    GradCAM implementation for visualizing class activation maps
    Reference: https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
    """

    def __init__(self, model, layerName):
        self.model = model
        self.layerName = layerName
        self.gradModel = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output]
        )

    def compute_heatmap(self, image, classIdx, eps=1e-8):
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = self.gradModel(inputs)
            
            if len(predictions[0]) == 1:
                # Binary Classification
                loss = predictions[0]
            else:
                loss = predictions[:, classIdx]

        grads = tape.gradient(loss, convOutputs)

        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        return (heatmap, output)

# Custom callback for GradCAM visualization
class GRADCamLogger(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, layer_name):
        super(GRADCamLogger, self).__init__()
        self.validation_data = validation_data
        self.layer_name = layer_name

    def on_epoch_end(self, epoch, logs=None):
        images = []
        grad_cam = []

        # Initialize GRADCam Class
        cam = GradCAM(self.model, self.layer_name)

        for image_batch in self.validation_data:
            # Get just one image from the batch
            image = np.expand_dims(image_batch[0], 0)
            pred = self.model.predict(image)
            classIDx = np.argmax(pred[0])
            print("Predicted Label:", classIDx)

            # Compute Heatmap
            heatmap = cam.compute_heatmap(image, classIDx)

            # Process image for visualization
            image = image.reshape(image.shape[1:])
            image = image * 255
            image = image.astype(np.uint8)

            # Overlay heatmap on original image
            heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            (heatmap_colored, output) = cam.overlay_heatmap(heatmap_resized, image, alpha=0.5)

            images.append(image)
            grad_cam.append(output)
            # Just process one image to avoid too many logs
            break

        # Log to wandb if it's initialized
        if wandb.run is not None:
            wandb.log({
                "images": [wandb.Image(img) for img in images],
                "gradcam": [wandb.Image(cam_img) for cam_img in grad_cam]
            })

# Get sample validation data for visualization
sample_images, sample_labels = next(iter(validation_generator))

# Initialize wandb
try:
    wandb.login()  # You'll need your API key
    wandb.init(project="throat_activation_map")
    USE_WANDB = True
except Exception as e:
    print(f"Could not initialize wandb: {e}")
    USE_WANDB = False

# Set up callbacks
callbacks = []

# Add wandb callback if available
if USE_WANDB:
    callbacks.append(WandbCallback(
        input_type="image", 
        validation_data=(sample_images, sample_labels)
    ))
    callbacks.append(GRADCamLogger(
        sample_images, 
        layer_name='block5_conv3'
    ))

# Add model checkpoint callback
checkpoint_filepath = './vgg16_pharyngitis_model.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_acc',
    mode='max',
    verbose=1,
    save_best_only=True
)
callbacks.append(model_checkpoint_callback)

# First training phase
history = model.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=5,
    callbacks=callbacks
)

# Unfreeze the last few layers of VGG16 for fine-tuning
for layer in model.layers:
    if isinstance(layer, keras.models.Model):  # This is our VGG16 base
        for i, sublayer in enumerate(layer.layers):
            # Unfreeze the last conv block (starting from layer 14)
            if i >= 14:
                sublayer.trainable = True
                print(f"Unfreezing: {sublayer.name}")

# Recompile with a lower learning rate for fine-tuning
base_learning_rate = 0.0001
model.compile(
    optimizer=Adam(learning_rate=base_learning_rate),
    loss='categorical_crossentropy',
    metrics=['acc']
)

# Second training phase (fine-tuning)
history_fine_tuning = model.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=5,
    callbacks=callbacks
)

# Plot training history
def plot_training_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'r', label='Training accuracy')
    plt.plot(epochs_range, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'r', label='Training loss')
    plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot the training history
plot_training_history(history)
plot_training_history(history_fine_tuning)

# Load the best model
best_model = keras.models.load_model(checkpoint_filepath)

# Function to visualize model predictions with GradCAM
def visualize_prediction(model, image_path, layer_name='block5_conv3'):
    # Load and preprocess the image
    if isinstance(image_path, str):
        # Load from file path
        image = cv2.imread(image_path)
        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        input_image = image / 255.0  # Normalize to [0,1]
        input_image = np.expand_dims(input_image, axis=0)
    else:
        # Assume it's already a preprocessed image
        input_image = np.expand_dims(image_path, axis=0)
        image = (image_path * 255).astype(np.uint8)
    
    # Make prediction
    preds = model.predict(input_image)
    class_idx = np.argmax(preds[0])
    class_names = ['No Pharyngitis', 'Pharyngitis']
    predicted_class = class_names[class_idx]
    confidence = preds[0][class_idx] * 100
    
    print(f"Prediction: {predicted_class} with {confidence:.2f}% confidence")
    
    # Create GradCAM visualization
    cam = GradCAM(model, layer_name)
    heatmap = cam.compute_heatmap(input_image, class_idx)
    
    # Resize heatmap to match image dimensions
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Overlay heatmap on image
    (heatmap_colored, output) = cam.overlay_heatmap(heatmap, image, alpha=0.5)
    
    # Display results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_colored)
    plt.title('GradCAM Heatmap')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(output)
    plt.title(f'Prediction: {predicted_class}\nConfidence: {confidence:.2f}%')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Test the visualization on a few validation images
for i in range(min(3, len(validation_strep_fnames))):
    img_path = os.path.join(validation_strep_dir, validation_strep_fnames[i])
    print(f"\nVisualizing pharyngitis image {i+1}:")
    visualize_prediction(best_model, img_path)

for i in range(min(3, len(validation_healthy_fnames))):
    img_path = os.path.join(validation_healthy_dir, validation_healthy_fnames[i])
    print(f"\nVisualizing healthy image {i+1}:")
    visualize_prediction(best_model, img_path)

# Function to export model to ONNX format (optional)
def export_to_onnx(model, output_path='model.onnx'):
    try:
        import tf2onnx
        
        # Convert Keras model to ONNX
        onnx_model, _ = tf2onnx.convert.from_keras(model)
        
        # Save the ONNX model
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"Model successfully exported to {output_path}")
        return True
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")
        return False
