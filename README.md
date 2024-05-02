# WonkaVision

## Project Overview

The goal of this project was to develop a computer vision model, named "Willy Vision," capable of efficiently detecting various models of chocolates based on shape, texture, and flavor. Deployed on the FG company's production line, this system can identify over 60 different types of chocolates from a catalog of 150, showcasing its ability to recognize a diverse array of chocolate products.

## What is Computer Vision?

Computer vision is a field within artificial intelligence that enables computers to interpret and understand the visual world. By using digital images and videos, along with deep learning models, the technology mimics human vision to identify, classify, and react to elements within visual data. This capability is pivotal in automating tasks that require visual recognition.

# The Magic behind WillyVison

## Bounding Box Prediction

Each grid cell in YOLO predicts multiple bounding boxes. Each bounding box prediction consists of five elements:

- **bx, by**: The center coordinates of the box relative to the bounds of the grid cell.
- **bw, bh**: The width and height of the box relative to the whole image.
- **Confidence score**: The likelihood that the box contains a specific object and how accurate it thinks the box is.

These values are predicted using the following formulas:

- \( bx = \sigma(t_x) + c_x \)
- \( by = \sigma(t_y) + c_y \)
- \( bw = p_w e^{t_w} \)
- \( bh = p_h e^{t_h} \)
- \( \text{Confidence} = \sigma(t_o) \)

Here, \( t_x, t_y, t_w, t_h, t_o \) are the outputs from the model, \( \sigma \) is the sigmoid function that normalizes the output to be between 0 and 1, making it a probability. \( c_x, c_y \) are the top-left coordinates of the grid cell, and \( p_w, p_h \) are the width and height of the anchor box.

## Class Prediction

The class probabilities are calculated for each bounding box using a softmax function, which is a generalized logistic function used for multiclass classification. If there are \( C \) classes, the softmax score for the \( i \)-th class given by a bounding box is calculated as:

\[ P(\text{class}_i | \text{object}) = \frac{e^{s_i}}{\sum_{j=1}^C e^{s_j}} \]

where \( s_i \) are the scores predicted by the model for each class.

## Non-Max Suppression (NMS)

After detecting multiple boxes, YOLO uses non-max suppression to eliminate redundant boxes by keeping only the best ones. This involves two key steps:

1. **Discarding all boxes with confidence less than a certain threshold**:
   - if \( \text{Confidence} < \text{threshold} \), discard the box.

2. **For remaining boxes, sort by confidence and compare using IoU (Intersection over Union)**:
   \[ \text{IoU}(A, B) = \frac{\text{area of overlap between A and B}}{\text{area of union between A and B}} \]


If IoU is greater than a specified threshold, the box with the lower confidence score is discarded. This is based on the idea that boxes with high overlap are likely covering the same object.

## Detail of the Trainings

- Go to the `ModelWeitgh` folder, and you wil have acces to detailed graph and pictures

## Usage

To run the "Willy Vision" detection model on your local machine, follow these steps:

#### Set Up Your Environment:

1. Ensure Python and necessary libraries `(opencv-python-headless, ultralytics)` are installed.
2. Install dependencies: pip install `opencv-python-headless ultralytics`.

#### Run the Detection:

1. Navigate to the project directory.
2. Execute the script: `python3 predict_video.py`.
3. View the Output:

The script displays the video with detected chocolates and saves an output file.