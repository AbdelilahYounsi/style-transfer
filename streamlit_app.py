import streamlit as st
import torch
import os
from PIL import Image
import io
import time
from pathlib import Path
import sys


from models.style_transfer_net import StyleNet
from torchvision import transforms
import re

# Page configuration
st.set_page_config(
    page_title="Neural Style Transfer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_style_model(model_path, device):
    """Load the style transfer model with caching"""
    try:
        style_model = StyleNet()
        state_dict = torch.load(model_path, map_location=device)        
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model.eval()
        return style_model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def stylize_image(content_image, model, device):
    """Apply style transfer to the content image"""
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    
    content_tensor = content_transform(content_image)
    content_tensor = content_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(content_tensor).cpu()
    
    # Convert output tensor to PIL Image
    output_img = output[0].clone().clamp(0, 255).numpy()
    output_img = output_img.transpose(1, 2, 0).astype("uint8")
    output_img = Image.fromarray(output_img)
    
    return output_img

def get_available_models(models_dir="./trained_models"):
    """Scan for available trained models"""
    if not os.path.exists(models_dir):
        return {}
    
    models = {}
    for file in os.listdir(models_dir):
        if file.endswith(('.pth', '.model')):
            # Extract a readable name from filename
            name = file.replace('.pth', '').replace('.model', '').replace('_', ' ').title()
            models[name] = os.path.join(models_dir, file)
    
    return models

def resize_image(image, max_size=800):
    """Resize image while maintaining aspect ratio"""
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        return image.resize((new_width, new_height), Image.LANCZOS)
    return image

def main():
    # Header
    st.title("🎨 Neural Style Transfer")
    st.markdown("Transform your photos into artistic masterpieces using deep learning!")
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    
    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.info(f"🖥️ Running on: **{device.upper()}**")
    
    # Get available models
    models_dict = get_available_models()
    
    
    # Model selection
    if models_dict:
        selected_model_name = st.sidebar.selectbox(
            "🎭 Select Style Model",
            list(models_dict.keys()),
            help="Choose a pre-trained style to apply"
        )
        selected_model_path = models_dict[selected_model_name]
    else:
        selected_model_path = None
    
    # Image size option
    max_image_size = st.sidebar.slider(
        "📏 Max Image Size (px)",
        min_value=256,
        max_value=1024,
        value=640,
        step=64,
        help="Larger sizes take longer to process"
    )
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Content Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload the photo you want to stylize"
        )
        
        if uploaded_file is not None:
            content_image = Image.open(uploaded_file).convert('RGB')
            
            # Resize if needed
            content_image = resize_image(content_image, max_image_size)
            
            st.image(content_image, caption="Original Image", use_container_width=True)
            
            # Display image info
            st.caption(f"📐 Size: {content_image.size[0]} x {content_image.size[1]} pixels")
    
    with col2:
        st.subheader("🎨 Stylized Output")
        
        if uploaded_file is not None and selected_model_path is not None:
            
            # Stylize button
            if st.button("✨ Apply Style Transfer", type="primary"):
                with st.spinner("🎨 Creating your masterpiece..."):
                    try:
                        # Load model
                        model = load_style_model(selected_model_path, device)
                        
                        if model is not None:
                            # Track processing time
                            start_time = time.time()
                            
                            # Stylize
                            stylized_image = stylize_image(content_image, model, device)
                            
                            processing_time = time.time() - start_time
                            
                            # Store in session state
                            st.session_state['stylized_image'] = stylized_image
                            st.session_state['processing_time'] = processing_time
                            
                    except Exception as e:
                        st.error(f"❌ Error during stylization: {str(e)}")
        
        # Display stylized image if it exists in session state
        if 'stylized_image' in st.session_state:
            stylized_image = st.session_state['stylized_image']
            st.image(stylized_image, caption="Stylized Image", use_container_width=True)
            
            # Display processing time
            if 'processing_time' in st.session_state:
                st.success(f"⏱️ Processed in {st.session_state['processing_time']:.2f} seconds")
            
            # Download button
            buf = io.BytesIO()
            stylized_image.save(buf, format='PNG')
            byte_im = buf.getvalue()
            
            st.download_button(
                label="⬇️ Download Stylized Image",
                data=byte_im,
                file_name=f"stylized_{int(time.time())}.png",
                mime="image/png"
            )
        elif uploaded_file is not None and selected_model_path is not None:
            st.info("👆 Click the button above to apply style transfer!")
        elif selected_model_path is None:
            st.warning("⚠️ Please add trained models to get started")
    
    # Example images section
    st.markdown("---")
    st.subheader("💡 How it works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1️⃣ Upload")
        st.markdown("Choose any photo from your device")
    
    with col2:
        st.markdown("### 2️⃣ Select Style")
        st.markdown("Pick from available artistic styles")
    
    with col3:
        st.markdown("### 3️⃣ Transform")
        st.markdown("Get your stylized masterpiece instantly!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #808495;'>
        Built with PyTorch and Streamlit | Based on Johnson et al. (2016) - Perceptual Losses for Real-Time Style Transfer
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
