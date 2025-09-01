# Baby Smile Detector 👶😊

A real-time baby smile detection system using ONNX runtime and OpenCV. This application can detect smiles in live webcam feed, images, or pre-recorded videos. Perfect for capturing those precious smiling moments of your baby!

![Baby Smile Detection Demo](https://via.placeholder.com/800x450.png?text=Baby+Smile+Detection+Demo)

## Features

- 🎭 Real-time face and smile detection
- 🎥 Multiple input sources (webcam, image, video)
- 💾 Save detected smiles automatically
- 🎚️ Adjustable confidence threshold
- 🔔 Sound alerts on smile detection
- 🎞️ Save processed videos
- 🖥️ Toggle display options (FPS, confidence scores)
- 🖼️ View saved smile captures

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- Webcam (for live detection)
- Windows/macOS/Linux

### Quick Start
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/baby-smile-detector.git
   cd baby-smile-detector
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   # OR
   source venv/bin/activate  # Linux/Mac
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained model (if not included):
   ```bash
   # Place the ONNX model file in the project directory
   # Rename it to 'best_face.onnx' or specify the path using --model
   ```

## 🎯 Usage Examples

### 🎥 Live Webcam Detection
Detect smiles using your default webcam:
```bash
python baby_smile_detector.py --show-fps --show-confidence --sound
```

### 📷 Process a Single Image
Analyze a photo and save the result:
```bash
python baby_smile_detector.py --input test_image.jpg --output results/
```

### 🎞️ Process a Video File
Process a video and save the output:
```bash
python baby_smile_detector.py --input input_video.mp4 --output-video output.mp4 --no-display
```

### 🖼️ View Saved Smile Captures
Browse through previously detected smiles:
```bash
python baby_smile_detector.py --view-saved
```

### 🏃‍♂️ Run in Background Mode
Process without display (useful for servers):
```bash
python baby_smile_detector.py --no-display --no-save --input video.mp4
```

### Advanced Options
```bash
# Run with custom settings
python baby_smile_detector.py \
    --input input.mp4 \
    --output-video output.mp4 \
    --threshold 0.7 \
    --show-fps \
    --show-confidence \
    --skip-frames 1
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to input image or video file | None (use webcam) |
| `--camera` | Camera device index | 0 |
| `--model` | Path to ONNX model | 'best_face.onnx' |
| `--threshold` | Confidence threshold (0-1) | 0.6 |
| `--output` | Output directory for saved smiles | 'smile_captures' |
| `--no-save` | Disable saving detected smiles | False |
| `--output-video` | Save output as video file | None |
| `--no-display` | Run without display | False |
| `--show-fps` | Show FPS counter | False |
| `--show-confidence` | Show confidence scores | False |
| `--sound` | Enable sound alerts | False |
| `--volume` | Sound volume (0.0-1.0) | 1.0 |
| `--skip-frames` | Skip N frames between processing | 0 |
| `--view-saved` | View previously saved captures | False |

## Controls (When Display is Enabled)

- `q` - Quit the application
- `s` - Save current frame
- `t` - Toggle sound alerts
- `↑`/`↓` - Increase/Decrease confidence threshold
- `f` - Toggle FPS display
- `c` - Toggle confidence display

## Viewing Saved Captures

Use the `--view-saved` option to browse through all saved smile captures. In the viewer:
- `→` or `d` - Next image
- `←` or `a` - Previous image
- `q` or `ESC` - Quit viewer

## 📋 Requirements

The application requires the following Python packages (automatically installed via `requirements.txt`):

- `opencv-python>=4.5.0` - For image/video processing
- `numpy>=1.19.0` - For numerical operations
- `onnxruntime>=1.8.0` - For running the ONNX model
- `Pillow>=8.0.0` - For image processing
- `sounddevice>=0.4.0` - For audio alerts (optional)
- `tqdm>=4.50.0` - For progress bars during video processing

## 🛠️ Installation Troubleshooting

### Common Issues
1. **Webcam not detected**:
   - Ensure no other application is using the webcam
   - Try a different camera index: `--camera 1`

2. **Model loading issues**:
   - Verify the ONNX model is in the correct location
   - Check model compatibility with your ONNX Runtime version

3. **Missing dependencies**:
   ```bash
   # If installation fails, try upgrading pip first
   pip install --upgrade pip
   
   # Then install requirements again
   pip install -r requirements.txt
   ```

## 💡 Tips for Best Results

### 📸 Image/Video Quality
- Use good lighting on the face
- Position the camera at eye level
- Ensure the face is clearly visible
- Ideal distance: 1-2 meters from the camera

### ⚙️ Performance Tuning
- **For better accuracy**:
  - Lower the threshold (e.g., `--threshold 0.5`)
  - Ensure good lighting conditions
  
- **For better performance**:
  - Increase `--skip-frames` value
  - Use `--no-display` for faster processing
  - Lower the resolution of input video

### 🎯 Model Information
- Input size: 640x480 pixels (automatically resized)
- Output: Bounding box coordinates and confidence scores
- Model format: ONNX (Open Neural Network Exchange)

## 🤝 Contributing

Contributions are welcome! Here's how you can help:
1. Report bugs or request features by opening an issue
2. Submit pull requests with improvements
3. Share your trained models
4. Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the OpenCV and ONNX Runtime communities
- Pre-trained models from [Face Detection RetinaFace](https://github.com/biubug6/Pytorch_Retinaface)
- Inspired by real-world baby monitoring applications

## License

This project is licensed under the MIT License - see the LICENSE file for details.



