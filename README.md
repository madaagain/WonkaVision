# WonkaVision

## Project Overview

The goal of this project was to develop a computer vision model, named "Willy Vision," capable of efficiently detecting various models of chocolates based on shape, texture, and flavor. Deployed on the FG company's production line, this system can identify over 60 different types of chocolates from a catalog of 150, showcasing its ability to recognize a diverse array of chocolate products.

## What is Computer Vision?

Computer vision is a field within artificial intelligence that enables computers to interpret and understand the visual world. By using digital images and videos, along with deep learning models, the technology mimics human vision to identify, classify, and react to elements within visual data. This capability is pivotal in automating tasks that require visual recognition.

# The Magic behind WillyVison


### Bounding Box Prediction: 

- Each cell predicts multiple bounding boxes for objects along with their confidence scores. The confidence score reflects the accuracy of the bounding box and whether the box contains a specific object.

![Equation for Grid Devision](https://blog.paperspace.com/content/images/2018/04/Screen-Shot-2018-04-10-at-3.18.08-PM.png)


![Equation for Grid Devision](https://pub.mdpi-res.com/applsci/applsci-12-07622/article_deploy/html/images/applsci-12-07622-g009.png?1659317028)

### Grid Division: 

- YOLO divides an image into a grid (e.g., 13x13 cells). Each grid cell is responsible for detecting objects that fall within its boundaries.

![Equation for Grid Devision](https://pylessons.com/media/Tutorials/YOLO-tutorials/YOLOv3-TF2-mnist/loss_function.png)

![Equation for Grid Devision](https://miro.medium.com/v2/resize:fit:1400/1*fahR8jDZxKqArfYRPCnDjw.png)

### Class Prediction: 

- Simultaneously, each cell predicts the class probabilities for each bounding box. This step involves using softmax functions that calculate the probability of the object belonging to a specific class.

### Non-max Suppression: 

- To ensure the model does not have overlapping bounding boxes for the same object, YOLO uses a technique called non-max suppression. This step filters out bounding boxes based on the confidence score and Intersection over Union (IoU) metric, keeping only the highest scoring boxes.

### Combining Results: 

- The bounding boxes and class predictions are combined to create the final output, which includes the positions, dimensions, and class labels of all detected objects.

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