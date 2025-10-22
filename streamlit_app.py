import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import numpy as np
import cv2
import time

@st.cache_resource
def load_model():
    """Load model once and cache it"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the checkpoint
    checkpoint = torch.load("model_weights.pth", map_location=device)
    
    # Get number of classes from checkpoint if available
    # Otherwise, change this manually
    num_classes = 61
    
    # Create base model
    model = maskrcnn_resnet50_fpn(weights=None)
    
    # Replace the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    # Load state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        st.write("Loaded model with exact architecture match")
    except Exception as e:
        st.warning(f"Loading with strict=False due to: {str(e)[:100]}")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model.to(device)
    model.eval()
    
    return model, device

def predict_and_visualize(image, model, device, threshold=0.5):
    """Run inference and draw results"""
    
    # Start timing
    start_time = time.time()
    
    # Preprocessing time
    preprocess_start = time.time()
    transform = transforms.ToTensor()
    img_tensor = transform(image).unsqueeze(0).to(device)
    preprocess_time = time.time() - preprocess_start
    
    # Inference time
    inference_start = time.time()
    with torch.no_grad():
        prediction = model(img_tensor)[0]
    inference_time = time.time() - inference_start
    
    # Postprocessing time
    postprocess_start = time.time()
    
    # Convert image to array for drawing
    img_array = np.array(image)
    
    # Filter by threshold
    scores = prediction['scores'].cpu().numpy()
    keep = scores >= threshold
    
    boxes = prediction['boxes'][keep].cpu().numpy().astype(int)
    labels = prediction['labels'][keep].cpu().numpy()
    masks = prediction['masks'][keep].cpu().numpy()
    
    # Draw detections
    for i, (box, label, mask) in enumerate(zip(boxes, labels, masks)):
        # Bounding box
        cv2.rectangle(img_array, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # Label
        text = f"Class {label}: {scores[keep][i]:.2f}"
        cv2.putText(img_array, text, (box[0], box[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Mask overlay
        mask = mask[0] > 0.5
        colored_mask = np.zeros_like(img_array)
        color = np.random.randint(50, 255, 3).tolist()
        colored_mask[mask] = color
        img_array = cv2.addWeighted(img_array, 1, colored_mask, 0.4, 0)
    
    postprocess_time = time.time() - postprocess_start
    
    # Total time
    total_time = time.time() - start_time
    
    # Create timing dictionary
    timing_info = {
        'preprocessing': preprocess_time,
        'inference': inference_time,
        'postprocessing': postprocess_time,
        'total': total_time
    }
    
    return img_array, prediction, keep, timing_info

# Streamlit UI
st.set_page_config(page_title="Mask R-CNN Segmentation", layout="wide")

st.title("Mask R-CNN Instance Segmentation")
st.write("Upload an image to perform instance segmentation using your fine-tuned model")

# Load model
with st.spinner("Loading model..."):
    try:
        model, device = load_model()
        st.success(f"Model loaded successfully on {device}!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Sidebar settings
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
show_masks = st.sidebar.checkbox("Show Masks", value=True)
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load and display original image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        st.caption(f"Image size: {image.size[0]} x {image.size[1]} pixels")
    
    # Run inference
    with st.spinner("Running inference..."):
        try:
            result_img, prediction, keep, timing_info = predict_and_visualize(image, model, device, threshold)
        except Exception as e:
            st.error(f"Error during inference: {str(e)}")
            st.stop()
    
    with col2:
        st.subheader("Detection Results")
        st.image(result_img, use_container_width=True)
    
    # Display timing information
    st.subheader("Processing Time")
    time_col1, time_col2, time_col3, time_col4 = st.columns(4)
    
    with time_col1:
        st.metric("Preprocessing", f"{timing_info['preprocessing']*1000:.1f} ms")
    with time_col2:
        st.metric("Inference", f"{timing_info['inference']*1000:.1f} ms")
    with time_col3:
        st.metric("Postprocessing", f"{timing_info['postprocessing']*1000:.1f} ms")
    with time_col4:
        st.metric("Total Time", f"{timing_info['total']*1000:.1f} ms", 
                  delta=f"{1000/timing_info['total']:.1f} FPS" if timing_info['total'] > 0 else "N/A")
    
    # Timing breakdown chart
    with st.expander("Detailed Timing Breakdown"):
        timing_data = {
            'Stage': ['Preprocessing', 'Inference', 'Postprocessing'],
            'Time (ms)': [
                f"{timing_info['preprocessing']*1000:.2f}",
                f"{timing_info['inference']*1000:.2f}",
                f"{timing_info['postprocessing']*1000:.2f}"
            ],
            'Percentage': [
                f"{(timing_info['preprocessing']/timing_info['total']*100):.1f}%",
                f"{(timing_info['inference']/timing_info['total']*100):.1f}%",
                f"{(timing_info['postprocessing']/timing_info['total']*100):.1f}%"
            ]
        }
        st.table(timing_data)
        
        st.info(f"""
        **Performance Summary:**
        - Total processing time: {timing_info['total']*1000:.2f} ms
        - Frames per second: {1000/timing_info['total']:.2f} FPS
        - Device: {device}
        - Image resolution: {image.size[0]}x{image.size[1]}
        """)
    
    # Display statistics
    st.subheader("Detection Statistics")
    num_detections = int(keep.sum())
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Detections", num_detections)
    
    if num_detections > 0:
        scores = prediction['scores'][keep].cpu().numpy()
        col2.metric("Avg Confidence", f"{scores.mean():.3f}")
        col3.metric("Max Confidence", f"{scores.max():.3f}")
        
        # Details table
        st.subheader("Detection Details")
        labels = prediction['labels'][keep].cpu().numpy()
        boxes = prediction['boxes'][keep].cpu().numpy()
        
        details = []
        for i, (label, score, box) in enumerate(zip(labels, scores, boxes)):
            details.append({
                "ID": i+1,
                "Class": int(label),
                "Confidence": f"{score:.3f}",
                "BBox": f"[{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]"
            })
        
        st.table(details)
    else:
        st.info("No detections found. Try lowering the confidence threshold.")
else:
    st.info("Upload an image to get started!")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("This app uses a fine-tuned Mask R-CNN model trained on the D2S supermarket dataset.")

# Add performance tips in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Performance Tips")
st.sidebar.markdown("""
- **GPU**: Ensure CUDA is available for faster inference
- **Image size**: Smaller images process faster
- **Threshold**: Higher threshold = faster postprocessing
""")
if str(device) == 'cpu':
    st.sidebar.warning("Running on CPU. Consider using GPU for faster inference.")
else:
    st.sidebar.success(f"Using {device} for acceleration")