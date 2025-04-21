import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import io
import os

# Set page config
st.set_page_config(page_title="Apple Detection", page_icon="🍏", layout="wide")

# Title and description
st.title("🍏 Apple Detection with YOLOv8")
st.write("Upload an image or video to detect apples using your trained YOLOv8 model.")

# Sidebar for settings
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

# Function to load the YOLOv8 model
@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO('model.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Function to process image
def process_image(image, confidence):
    if model is None:
        st.error("Model not loaded properly")
        return image
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Run inference
    results = model.predict(source=img_array, conf=confidence)
    
    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw rectangle
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Get confidence score
            conf = box.conf[0]
            
            # Put text (class name and confidence)
            label = f"Apple {conf:.2f}"
            cv2.putText(img_array, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_array

# Function to process video
def process_video(video_path, confidence):
    if model is None:
        st.error("Model not loaded properly")
        return None
    
    # Create temporary output file
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_output.name
    temp_output.close()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video frame by frame
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model.predict(source=frame, conf=confidence)
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Get confidence score
                conf = box.conf[0]
                
                # Put text (class name and confidence)
                label = f"Apple {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Write processed frame
        out.write(frame)
        processed_frames += 1
        progress_bar.progress(processed_frames / total_frames)
    
    # Release everything
    cap.release()
    out.release()
    
    return output_path

# Main content
upload_option = st.radio("Select input type:", ("Image", "Video"))

if upload_option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Process and display detected image
        with st.spinner("Detecting apples..."):
            detected_image = process_image(image, confidence_threshold)
            st.image(detected_image, caption="Detected Apples", use_column_width=True)
            
            # Download button for processed image
            img_pil = Image.fromarray(detected_image)
            buf = io.BytesIO()
            img_pil.save(buf, format="JPEG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Processed Image",
                data=byte_im,
                file_name="detected_apples.jpg",
                mime="image/jpeg"
            )

else:  # Video
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save uploaded video to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        # Display original video
        st.video(uploaded_file)
        
        # Process video
        with st.spinner("Processing video..."):
            output_path = process_video(video_path, confidence_threshold)
            
            if output_path:
                # Display processed video
                st.success("Video processing complete!")
                st.video(output_path)
                
                # Download button for processed video
                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                
                st.download_button(
                    label="Download Processed Video",
                    data=video_bytes,
                    file_name="detected_apples.mp4",
                    mime="video/mp4"
                )
                
                # Clean up temporary files
                os.unlink(video_path)
                os.unlink(output_path)

# Add some info
st.markdown("---")
st.markdown("""
### About this App
- This app uses your trained YOLOv8 model to detect apples in images and videos.
- Adjust the confidence threshold in the sidebar to change detection sensitivity.
- Upload an image or video to see the detection in action.
""")