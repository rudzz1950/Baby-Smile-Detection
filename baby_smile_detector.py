import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import time
import os
import winsound
from datetime import datetime
import argparse

class FaceDetector:
    def __init__(self, model_path='best_face.onnx'):
        """Initialize face detector with ONNX model."""
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_size = (self.input_shape[2], self.input_shape[3])
        
    def detect_faces(self, image):
        """Detect faces in the image."""
        # Preprocess
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), False, False
        )
        
        # Forward pass
        self.session.run(None, {self.input_name: blob})
        
        # Process detections (simplified - adjust based on your model's output)
        detections = self.session.run(None, {self.input_name: blob})[0][0][0]
        
        faces = []
        height, width = image.shape[:2]
        
        for detection in detections:
            confidence = detection[2]
            if confidence > 0.7:  # Confidence threshold
                box = detection[3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX - startX, endY - startY))
                
        return faces

class BabySmileDetector:
    def __init__(self, model_path='best_face.onnx'):
        """Initialize the baby smile detector with ONNX model."""
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_size = (self.input_shape[2], self.input_shape[3])  # (width, height)
        self.face_detector = FaceDetector()
        
    def preprocess(self, image):
        """Preprocess the input image for the model."""
        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize to model's expected input shape (height, width) = (640, 480)
        target_height, target_width = 640, 480
        resized = cv2.resize(image, (target_width, target_height))
        
        # Convert to float32 and normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Change from HWC to CHW format
        chw = np.transpose(normalized, (2, 0, 1))  # HWC to CHW
        
        # Add batch dimension (NCHW format)
        batch = np.expand_dims(chw, axis=0)
        
        return batch
    
    def detect(self, image, confidence_threshold=0.5):
        """
        Detect faces and smiles in the input image.
        
        Args:
            image: Input image (BGR format)
            confidence_threshold: Minimum confidence score to consider a detection valid
            
        Returns:
            tuple: (is_smiling, confidence, bbox)
                - is_smiling: Boolean indicating if a smile was detected
                - confidence: Confidence score of the detection
                - bbox: Bounding box coordinates [x1, y1, x2, y2] or None if no detection
        """
        # Preprocess the image
        input_tensor = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Process detections
        if outputs and len(outputs[0]) > 0:
            # Assuming output format: [batch_id, class_id, confidence, x1, y1, x2, y2]
            detections = outputs[0]
            
            # Filter detections by confidence and get the one with highest score
            valid_detections = [d for d in detections if d[2] >= confidence_threshold]
            if valid_detections:
                # Get detection with highest confidence
                best_detection = max(valid_detections, key=lambda x: x[2])
                
                # Extract values
                _, class_id, confidence, x1, y1, x2, y2 = best_detection
                
                # Convert coordinates to integers
                h, w = image.shape[:2]
                x1 = max(0, int(x1 * w))
                y1 = max(0, int(y1 * h))
                x2 = min(w, int(x2 * w))
                y2 = min(h, int(y2 * h))
                
                # Assuming class_id 0 is 'not smiling' and 1 is 'smiling'
                is_smiling = class_id == 1
                
                return is_smiling, float(confidence), [x1, y1, x2, y2]
        
        return False, 0.0, None

def play_sound(frequency=1000, duration=300):
    """Play a beep sound."""
    try:
        winsound.Beep(frequency, duration)
    except:
        pass  # Sound not available on this system

def save_smiling_face(image, face_rect, confidence, output_dir='smile_captures'):
    """Save the detected smiling face image."""
    os.makedirs(output_dir, exist_ok=True)
    x, y, w, h = face_rect
    face_img = image[y:y+h, x:x+w]
    if face_img.size > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = os.path.join(output_dir, f'smile_{timestamp}_{confidence:.2f}.jpg')
        cv2.imwrite(filename, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
        return filename
    return None

def parse_arguments():
    parser = argparse.ArgumentParser(description='Baby Smile Detector')
    
    # Input/Output options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--input', type=str, help='Path to input image or video file')
    input_group.add_argument('--camera', type=int, default=0, help='Camera device index (default: 0)')
    
    # Model options
    parser.add_argument('--model', type=str, default='best_face.onnx', 
                      help='Path to ONNX model (default: best_face.onnx)')
    parser.add_argument('--threshold', type=float, default=0.6, 
                      help='Confidence threshold for smile detection (0-1, default: 0.6)')
    
    # Output options
    parser.add_argument('--output', type=str, default='smile_captures', 
                      help='Output directory for saved smiles (default: smile_captures)')
    parser.add_argument('--no-save', action='store_true', 
                      help='Disable saving detected smiles')
    parser.add_argument('--output-video', type=str, 
                      help='Save output as video file')
    
    # Display options
    parser.add_argument('--no-display', action='store_true', 
                      help='Run without display (useful for processing videos)')
    parser.add_argument('--show-fps', action='store_true', 
                      help='Show FPS counter')
    parser.add_argument('--show-confidence', action='store_true', 
                      help='Show confidence scores')
    
    # Audio options
    parser.add_argument('--sound', action='store_true', 
                      help='Enable sound alerts')
    parser.add_argument('--volume', type=float, default=1.0, 
                      help='Sound volume (0.0 to 1.0, default: 1.0)')
    
    # Processing options
    parser.add_argument('--skip-frames', type=int, default=0, 
                      help='Number of frames to skip between processing (for faster processing)')
    
    # View saved captures
    parser.add_argument('--view-saved', action='store_true',
                      help='View previously saved smile captures')
    
    return parser.parse_args()

def view_saved_captures(directory='smile_captures'):
    """Display a grid of previously saved smile captures."""
    import glob
    from PIL import Image
    
    # Get all saved images
    image_paths = sorted(glob.glob(os.path.join(directory, '*.jpg')) + 
                        glob.glob(os.path.join(directory, '*.png')))
    
    if not image_paths:
        print(f"No saved captures found in {directory}")
        return
    
    print(f"Found {len(image_paths)} saved captures. Press any key to navigate, 'q' to quit.")
    
    idx = 0
    while True:
        img = cv2.imread(image_paths[idx])
        if img is None:
            print(f"Could not load image: {image_paths[idx]}")
            continue
            
        # Show image with filename
        filename = os.path.basename(image_paths[idx])
        cv2.putText(img, filename, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Saved Capture', img)
        
        # Handle key presses
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC to quit
            break
        elif key == 83 or key == 2:  # Right arrow or 'd' for next
            idx = (idx + 1) % len(image_paths)
        elif key == 81 or key == 3:  # Left arrow or 'a' for previous
            idx = (idx - 1) % len(image_paths)
    
    cv2.destroyAllWindows()

def main():
    args = parse_arguments()
    
    # Handle view saved captures mode
    if args.view_saved:
        view_saved_captures(args.output)
        return
    
    # Initialize detectors
    detector = BabySmileDetector(args.model)
    
    # Initialize video capture
    if args.input:
        # Input is a file
        if not os.path.isfile(args.input):
            print(f"Error: Input file not found: {args.input}")
            return
        cap = cv2.VideoCapture(args.input)
        is_video = args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    else:
        # Input is camera
        cap = cv2.VideoCapture(args.camera)
        is_video = True  # Treat camera as continuous video
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.input if args.input else f'camera {args.camera}'}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output video is specified
    video_writer = None
    if args.output_video and is_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = args.output_video if args.output_video.endswith('.mp4') else f"{args.output_video}.mp4"
        video_writer = cv2.VideoWriter(
            output_path, fourcc, fps if fps > 0 else 30, 
            (frame_width, frame_height)
        )
    
    # Create output directory if saving is enabled
    if not args.no_save and not os.path.exists(args.output):
        os.makedirs(args.output)
    
    try:
        # Try to print with UTF-8 encoding
        print("\n=== Baby Smile Detector ===")
        print(f"Input: {args.input if args.input else f'Camera {args.camera}'}")
        print(f"Model: {args.model}")
        print(f"Threshold: {args.threshold:.2f}")
        print(f"Save captures: {not args.no_save} ({args.output if not args.no_save else 'disabled'}")
        if video_writer:
            print(f"Saving output video to: {output_path}")
        print("\nControls:")
        print("  q - Quit")
        print("  s - Save current frame")
        print("  t - Toggle sound")
        print("  Up/Down arrows - Adjust confidence threshold")
        print("  f - Toggle FPS display")
        print("  c - Toggle confidence display")
        print("\nPress 'q' to quit...")
    except UnicodeEncodeError:
        # Fallback to ASCII-only output
        print("\n=== Baby Smile Detector ===")
        print(f"Input: {args.input if args.input else f'Camera {args.camera}'}")
        print(f"Model: {args.model}")
        print(f"Threshold: {args.threshold:.2f}")
        print(f"Save captures: {not args.no_save} ({args.output if not args.no_save else 'disabled'}")
        if video_writer:
            print(f"Saving output video to: {output_path}")
        print("\nControls:")
        print("  q - Quit")
        print("  s - Save current frame")
        print("  t - Toggle sound")
        print("  Up/Down arrows - Adjust confidence threshold")
        print("  f - Toggle FPS display")
        print("  c - Toggle confidence display")
        print("\nPress 'q' to quit...")
    
    # Initialize variables
    prev_time = time.time()
    frame_count = 0
    fps = 0
    frame_skip_counter = 0
    
    # Settings
    sound_enabled = args.sound
    show_fps = args.show_fps
    show_confidence = args.show_confidence
    last_smile_time = 0
    smile_cooldown = 2.0  # seconds
    
    # For FPS calculation
    frame_times = []
    fps_window = 10  # Number of frames to average FPS over
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        current_time = time.time()
        
        # Calculate FPS
        frame_times.append(current_time)
        if len(frame_times) > fps_window:
            frame_times.pop(0)
        
        if len(frame_times) > 1:
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
        
        # Skip frames if specified (for faster processing)
        frame_skip_counter += 1
        if frame_skip_counter <= args.skip_frames:
            continue
        frame_skip_counter = 0
        
        # Make a copy for display
        display_frame = frame.copy()
        
        # Detect face and smile
        start_time = time.time()
        is_smiling, confidence, bbox = detector.detect(frame, args.threshold)
        inference_time = (time.time() - start_time) * 1000  # in ms
        
        # Draw results if face detected
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            
            # Draw face rectangle
            color = (0, 255, 0) if is_smiling else (0, 0, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw smile status and confidence
            status = "Smiling" if is_smiling else "Not Smiling"
            label = f"{status}: {confidence:.2f}"
            cv2.putText(display_frame, label, (x1, y1 - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Handle smile detection
            if is_smiling and (current_time - last_smile_time) > smile_cooldown:
                if sound_enabled:
                    play_sound()
                last_smile_time = current_time
                
                if args.save:
                    saved_file = save_smiling_face(frame, (x1, y1, x2-x1, y2-y1), confidence, args.output)
                    if saved_file:
                        print(f"Saved smile: {saved_file}")
        else:
            # Draw "No face detected" message
            cv2.putText(display_frame, "No face detected", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display status overlay
        y_offset = 30
        line_height = 25
        
        # FPS counter
        if show_fps:
            cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += line_height
        
        # Threshold and controls
        cv2.putText(display_frame, f"Threshold: {args.threshold:.2f} [↑/↓]", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += line_height
        
        # Sound status
        cv2.putText(display_frame, f"Sound: {'ON' if sound_enabled else 'OFF'} [t]", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += line_height
        
        # Display mode toggles
        cv2.putText(display_frame, f"Show FPS: {'ON' if show_fps else 'OFF'} [f]", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += line_height
        
        cv2.putText(display_frame, f"Show conf: {'ON' if show_confidence else 'OFF'} [c]", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show the frame
        cv2.imshow('Baby Smile Detector', display_frame)
        
        # Write frame to output video if enabled
        if video_writer is not None:
            video_writer.write(cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
        
        # Display the frame if not in no-display mode
        if not args.no_display:
            cv2.imshow('Baby Smile Detector', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                break
            elif key == ord('t'):  # Toggle sound
                sound_enabled = not sound_enabled
                print(f"Sound {'enabled' if sound_enabled else 'disabled'}")
            elif key == ord('s'):  # Save current frame
                if not args.no_save:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = os.path.join(args.output, f'frame_{timestamp}.jpg')
                    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    print(f"Saved frame: {filename}")
            elif key == ord('f'):  # Toggle FPS display
                show_fps = not show_fps
            elif key == ord('c'):  # Toggle confidence display
                show_confidence = not show_confidence
            elif key == 82 or key == 56:  # Up arrow or 8 - increase threshold
                args.threshold = min(0.95, args.threshold + 0.05)
                print(f"Threshold: {args.threshold:.2f}")
            elif key == 84 or key == 50:  # Down arrow or 2 - decrease threshold
                args.threshold = max(0.05, args.threshold - 0.05)
                print(f"Threshold: {args.threshold:.2f}")
        
        # If processing a single image, wait for key press
        if not is_video and not args.no_display:
            cv2.waitKey(0)
            break
    
    # Release resources
    cap.release()
    if video_writer is not None:
        video_writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    
    print("\nProcessing complete!")
    if not args.no_save and os.path.exists(args.output):
        num_captures = len([f for f in os.listdir(args.output) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if num_captures > 0:
            print(f"Saved {num_captures} capture(s) to {os.path.abspath(args.output)}")
            print("Run with --view-saved to view all captured smiles")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("\nDetection ended. Smile captures saved to:", os.path.abspath(args.output))

if __name__ == "__main__":
    main()
