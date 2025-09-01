import os
import cv2
import numpy as np
import onnxruntime as ort

def test_model_with_image(model_path, image_path):
    try:
        # Load the ONNX model
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        
        # Load and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return
            
        # Model expects input shape: [1, 3, 640, 480] (NCHW format)
        target_height, target_width = 640, 480
        
        # Resize image to match model's expected input dimensions
        resized = cv2.resize(image, (target_width, target_height))
        print(f"Resized image shape: {resized.shape} (height, width, channels)")
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] range
        normalized = rgb.astype(np.float32) / 255.0
        print(f"After normalization: {normalized.shape} (height, width, channels)")
        
        # Convert HWC to CHW format
        chw = np.transpose(normalized, (2, 0, 1))  # HWC to CHW
        print(f"After HWC to CHW: {chw.shape} (channels, height, width)")
        
        # Add batch dimension (NCHW format)
        input_tensor = np.expand_dims(chw, axis=0)
        print(f"After adding batch: {input_tensor.shape} (batch, channels, height, width)")
        
        # Verify final shape matches model's expected input
        expected_shape = session.get_inputs()[0].shape
        actual_shape = input_tensor.shape
        if actual_shape != tuple(expected_shape):
            print(f"Warning: Input shape {actual_shape} doesn't match expected {expected_shape}")
            print("Attempting to reshape to match model's expected input...")
            input_tensor = np.transpose(input_tensor, (0, 1, 3, 2))  # Swap height and width
            print(f"After reshaping: {input_tensor.shape}")
        
        # Run inference
        outputs = session.run(None, {input_name: input_tensor})
        
        # Process outputs
        print("\nInference successful!")
        print(f"Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"Output {i} shape: {output.shape}")
            print(f"Output {i} sample values:\n{output[0][:5]}...")  # Print first 5 values
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Use a sample image (replace with path to your test image)
    test_image_path = "test_image.jpg"
    
    # If test image doesn't exist, create a blank one
    if not os.path.exists(test_image_path):
        print(f"Creating test image at {test_image_path}")
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray image
        cv2.imwrite(test_image_path, test_image)
    
    test_model_with_image("best_face.onnx", test_image_path)
