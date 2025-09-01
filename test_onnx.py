import cv2
import numpy as np
import onnxruntime as ort

def test_onnx_model(model_path):
    try:
        # Load the ONNX model
        print(f"Loading model from {model_path}...")
        session = ort.InferenceSession(model_path)
        print("Model loaded successfully!")
        
        # Print model input/output details
        print("\nModel Inputs:")
        for i, input in enumerate(session.get_inputs()):
            print(f"  Input {i}: {input.name}, Shape: {input.shape}, Type: {input.type}")
        
        print("\nModel Outputs:")
        for i, output in enumerate(session.get_outputs()):
            print(f"  Output {i}: {output.name}, Shape: {output.shape}, Type: {output.type}")
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    test_onnx_model("best_face.onnx")
