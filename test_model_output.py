import numpy as np
import onnxruntime as ort

def test_model_output():
    # Load the ONNX model
    model_path = "best_face.onnx"
    session = ort.InferenceSession(model_path)
    
    # Create a dummy input with the correct shape
    dummy_input = np.random.randn(1, 3, 640, 480).astype(np.float32)
    
    # Run inference
    outputs = session.run(None, {session.get_inputs()[0].name: dummy_input})
    
    # Print output details
    print("Model Output Analysis:")
    print("----------------------")
    
    for i, output in enumerate(outputs):
        print(f"Output {i}:")
        print(f"  Shape: {output.shape}")
        print(f"  Dtype: {output.dtype}")
        print(f"  Min: {np.min(output):.4f}, Max: {np.max(output):.4f}")
        print(f"  Mean: {np.mean(output):.4f}, Std: {np.std(output):.4f}")
        print("  Sample values:", np.round(output.ravel()[:5], 4))
        print()

if __name__ == "__main__":
    test_model_output()
