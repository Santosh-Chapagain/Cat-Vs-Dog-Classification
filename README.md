# Cat vs Dog Classification

A deep learning project that uses Convolutional Neural Networks (CNN) to classify images as either cats or dogs.

## Project Overview

This project implements a binary image classifier using TensorFlow and Keras to distinguish between images of cats and dogs. The model is built using a CNN architecture with multiple convolutional layers, batch normalization, and dropout for regularization.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- OpenCV (cv2)
- Matplotlib
- NumPy

### Installation

```bash
pip install tensorflow keras opencv-python matplotlib numpy
```

## Model Architecture

The CNN model consists of:

- **3 Convolutional Blocks:**
  - Conv2D layer (32, 64, 128 filters respectively)
  - Batch Normalization
  - MaxPooling (2x2)

- **Fully Connected Layers:**
  - Flatten layer
  - Dense layer (128 neurons) with ReLU activation
  - Dropout (0.2)
  - Dense layer (64 neurons) with ReLU activation
  - Dropout (0.2)
  - Output layer (1 neuron) with Sigmoid activation

**Input Shape:** 256x256x3 (RGB images)  
**Loss Function:** Binary Crossentropy  
**Optimizer:** Adam  
**Metrics:** Accuracy

## Usage

### Training the Model

1. Ensure your dataset is organized in the structure shown above
2. Open `main.ipynb` in Jupyter Notebook or VS Code
3. Run all cells sequentially to:
   - Load and preprocess the data
   - Build the CNN model
   - Train for 20 epochs
   - Visualize training/validation accuracy

### Making Predictions

To classify a new image:

```python
import cv2

# Load and preprocess the image
test_img = cv2.imread('path_to_image.png')
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img = cv2.resize(test_img, (256, 256))
test_input = test_img.reshape(1, 256, 256, 3)
test_input = test_input / 255.0

# Make prediction
prediction = model.predict(test_input)[0][0]
if prediction > 0.5:
    print(f"The given image is Dog (confidence: {prediction:.2%})")
else:
    print(f"The given image is Cat (confidence: {(1-prediction):.2%})")
```

## Features

- **Data Augmentation Ready:** Uses Keras image dataset generators
- **Batch Processing:** Processes data in batches of 32
- **Normalization:** All images are normalized (pixel values 0-1)
- **Regularization:** Dropout layers to prevent overfitting
- **Visualization:** Training history plots showing accuracy trends

## Model Performance

The model is trained for 20 epochs with both training and validation datasets. Performance can be visualized through accuracy plots generated during training.

## License

This project is open source and available for educational purposes.

## Author

Santosh Chapagain

## Acknowledgments

- Dataset: Cat and Dog images dataset
- Framework: TensorFlow/Keras
