import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import io

# Load CIFAR-10 dataset
@st.cache
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Function to plot images
def plot_images():
    fig, axs = plt.subplots(1, 10, figsize=(10, 10))
    for i in range(10):
        axs[i].imshow(x_train[i])
        axs[i].set_title(class_names[y_train[i][0]])
        axs[i].axis('off')
    st.pyplot(fig)

# Function to plot training history
def plot_history(history):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    ax[0].plot(history.history['accuracy'], label='accuracy')
    ax[0].plot(history.history['val_accuracy'], label='val_accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(loc='lower right')

    ax[1].plot(history.history['loss'], label='loss')
    ax[1].plot(history.history['val_loss'], label='val_loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend(loc='upper right')

    st.pyplot(fig)

# Streamlit UI
st.title('CIFAR-10 Classification Dashboard')

st.sidebar.header('Model Configuration')
epochs = st.sidebar.slider('Epochs', min_value=1, max_value=50, value=20)

st.sidebar.text('Click to train the model.')
if st.sidebar.button('Train Model'):
    st.text('Training the model...')

    # Build and compile the CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=epochs, 
                        validation_data=(x_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    st.write(f'Test accuracy: {test_acc}')

    # Plot training history
    plot_history(history)

# Display sample images from the dataset
st.subheader('Sample Images from CIFAR-10 Dataset')
plot_images()

# Show model summary
st.sidebar.header('Model Summary')
if st.sidebar.checkbox('Show Model Summary'):
    model_summary = io.StringIO()
    model.summary(print_fn=lambda x: model_summary.write(x + '\n'))
    st.text(model_summary.getvalue())
