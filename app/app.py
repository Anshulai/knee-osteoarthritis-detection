import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
import keras
from PIL import Image
from tensorflow.keras.preprocessing import image
import os

# Custom SeparableConv2D layer to handle legacy arguments
@keras.saving.register_keras_serializable(package="Custom")
class CustomSeparableConv2D(tf.keras.layers.SeparableConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        kwargs.pop('kernel_initializer', None)
        kwargs.pop('kernel_regularizer', None)
        kwargs.pop('kernel_constraint', None)
        super().__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        return config

# Load model with custom layer
try:
    # Try local development path first
    model_path = os.path.join('..', 'src', 'models', 'model_Xception_ft.hdf5')
    if not os.path.exists(model_path):
        # Try production path
        model_path = os.path.join('models', 'model_Xception_ft.hdf5')
    
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={'SeparableConv2D': CustomSeparableConv2D}
    )
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()
target_size = (224, 224)

# Print model summary to understand the architecture
model.summary()

# Grad-CAM heatmap function
def make_gradcam_heatmap(model, img_array, pred_index=None):
    try:
        # First make a prediction to ensure model is built
        model(img_array)
        
        # Get the Xception layer
        xception_layer = model.get_layer('xception')
        
        # Find the last conv layer in Xception
        last_conv_layer = None
        for layer in xception_layer.layers[::-1]:
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            raise ValueError("Could not find convolutional layer in Xception model")
        
        print(f"Using layer '{last_conv_layer.name}' for GradCAM")
        
        # Create gradient model
        with tf.GradientTape() as tape:
            # Create a model that maps the input image to:
            # 1. The last conv layer output
            # 2. The final model output
            grad_model = tf.keras.models.Model(
                [xception_layer.input],
                [last_conv_layer.output, xception_layer.output]
            )
            
            # Process the image through the base model
            conv_output, xception_output = grad_model(img_array)
            
            # Then through the rest of the model
            final_output = model.layers[-1](
                model.layers[-2](
                    model.layers[-3](xception_output)
                )
            )
            
            if pred_index is None:
                pred_index = tf.argmax(final_output[0])
            class_channel = final_output[:, pred_index]
        
        # Get gradients
        grads = tape.gradient(class_channel, conv_output)

        # Compute importance weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Create the heatmap
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

        return heatmap.numpy()
    except Exception as e:
        print(f"Error in make_gradcam_heatmap: {str(e)}")
        return None

# Function to save and display Grad-CAM overlay
def save_and_display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = image.array_to_img(superimposed_img)

    return superimposed_img

# Streamlit app configuration
try:
    icon = Image.open("img/logo.png")
except:
    icon = None
st.set_page_config(
    page_title="Detection of osteoarthritis",
    page_icon=icon,
)

class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]

# Sidebar
with st.sidebar:
    st.image(icon, use_container_width=True)
    st.subheader("Detection of osteoarthritis")
    st.caption("Made by Anshul Kumar Singh")

    st.subheader(":arrow_up: Upload image")
    uploaded_file = st.file_uploader("Choose x-ray image")

# Body
st.header("Detection of osteoarthritis")

col1, col2 = st.columns(2)
y_pred = None

if uploaded_file is not None:
    with col1:
        st.subheader(":camera: Input")
        st.image(uploaded_file, use_container_width=True)

        img = image.load_img(uploaded_file, target_size=target_size)
        img = image.img_to_array(img)
        img_aux = img.copy()

        if st.button(":arrows_counterclockwise: Predict Arthritis in the Knee"):
            img_array = np.expand_dims(img_aux, axis=0)
            img_array = np.float32(img_array)
            img_array = tf.keras.applications.xception.preprocess_input(img_array)

            with st.spinner("Wait for it..."):
                y_pred = model.predict(img_array)

            y_pred = 100 * y_pred[0]
            probability = np.amax(y_pred)
            number = np.where(y_pred == np.amax(y_pred))
            grade = str(class_names[np.amax(number)])

            st.subheader(":white_check_mark: Prediction")
            st.metric(
                label="Severity Grade:",
                value=f"{class_names[np.amax(number)]} - {probability:.2f}%",
            )

    if y_pred is not None:
        with col2:
            st.subheader(":mag: Explainability")
            heatmap = make_gradcam_heatmap(model, img_array)
            if heatmap is not None:
                image = save_and_display_gradcam(img, heatmap)
                st.image(image, use_container_width=True)
            else:
                st.warning("Could not generate heatmap visualization")

            st.subheader(":bar_chart: Analysis")
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.barh(class_names, y_pred, height=0.55, align="center")
            for i, (c, p) in enumerate(zip(class_names, y_pred)):
                ax.text(p + 2, i - 0.2, f"{p:.2f}%")
            ax.grid(axis="x")
            ax.set_xlim([0, 120])
            ax.set_xticks(range(0, 101, 20))
            fig.tight_layout()
            st.pyplot(fig)
