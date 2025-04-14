# Object Detection
## BDD Dataset
### Introduction
BDD100K Dataset consists of 100,000 videos and images of diverse driving scenarios in various locations in the US
The diversity of the scenarios is across the weather, time of the day, terrains and locations.
The dataset consists the scenarios that are rainy, gloomy, sunny, during the dawn or dusk, day, night. The also include urban and rural scenarios.
This would clearly make the BDD Dataset ideal for computer vision as well as autonomous driving applications.

### Data Analysis
The image data is used in this project for object detection purpose. The dataset includes training set with 70k images, 
validation set with 10k images and test set with 20k images. The resolution of the images is 1280x720 pixels. The images are labeled to represent the objects/ persons/ roads/ 
road signals found in each scenario. These labels in the form of .json files describe the scenario with objects in bounding boxes, 
along with their class. The labeled data also represents drivable areas, lanes, weather, scene type as well as time of the day.
The object classes the dataset represents are:
  - Bus
  - Traffic Light 
  - Traffic Sign
  - Person
  - Bike
  - Truck
  - Motor
  - Car
  - Train
  - Rider
<img width="428" alt="image" src="https://github.com/user-attachments/assets/ebdd5f4d-3004-4364-90e9-1e087e0115c7" />
Each image mostly have more than one class object as well as more than one of same class objects. The statistics provided above shows the number of occurences of 
all the objects in the whole dataset. The bounding boxes coordinates are represented as `x1,y1,x2,y2`. This dataset with large number of diverse scenarios 
and labels would train a robust model that fits well with the expected outputs irrespective of weather, time and unseen roads.

## Model Selection
Object detection includes classification and localization of the objects in an image.
The model requires to find all the relevant objects, classify them and place them in a bounding box
according to its estimated size. For easy and quicker model training, transfer learning is recommended. 
This suggests to use an existing model already trained on large datasets of similar data and object classes. 
This would simplify the training by already finding the right parameters for the network like optimizers, 
bias variance trade-offs, activation functions and learning rate.

The object detection models has evolved over many years, some of them are
- Viola-Jones Detector
- HOG Detector (Histogram of Oriented Gradients)
- R-CNN (Region-Convolutional Neural Network) and Fast R-CNN
- YOLO (You only Look Once)

Some models seperate the tasks into two stages whereas others combine them into one step
The two-stage detectors in first stage idenfies the region of the object using conventional
deep networks. Second stage involves object classification based on features extracted from the proposed region with 
bounding-box regression. Examples of two-stage detectors are RCNN, Faster RCNN or G-RCNN.
These models find a region of interest and use this cropped region for classification.They models achieve the highest 
detection accuracy but are typically slower. 

One-stage detectors predict bounding boxes over the images without the region proposal step. This process consumes less time 
and can therefore be used in real-time applications. They are super fast but not as good at recognizing irregularly shaped objects 
or a group of small objects. Examples of one-stage detectors are YOLO, SSD, and RetineNet.
Single-stage algorithms include generally faster detection speed and greater structural simplicity and efficiency.

The latest real-time detectors are YOLOv7 (2022), YOLOR (2021), and YOLOv4-Scaled (2020)
<img width="449" alt="image" src="https://github.com/user-attachments/assets/2a6d469b-b5e9-42b5-8136-b5c75be6d636" />
<img width="449" alt="image" src="https://github.com/user-attachments/assets/d9f97eb6-85c4-4fa8-9da6-fb7277042135" />
The first image above represents the performance of various detectors and their average precision of the bounding box
coordinates of the detected objects. The second image represents the speed of the detectors as frames per second, where 
the objects are detected. YOLOv7 has outperformed other models in both accuracy(average precision) as well as inference time.
## YOLOv8
### YOLO Introduction
YOLO uses and end-to-end neural network that predicts bounding boxes and class probabilities all at once. 
It perfroms all the predictions with a single fully connected layer and processes an image in a single iteraction.
<img width="497" alt="image" src="https://github.com/user-attachments/assets/87e2aa84-714d-4d2c-a9ca-05327c3235d9" />
The image above shows the architecture of a YOLO model. It includes 20 convolutional layers pre-trained
using ImageNet, this pre-trained model is converted to perform detection with fully connected layers 
to predict class probabilities and bounding box coordinates. YOLO uses Non-maximum suppression (NMS) as post-processing step 
to improve the accuracy and efficiency of object detection.
### YOLOv8 Features
The latest YOLOv8 has shown smaller precision and faster object detection. Hence, 
it is an ideal model to perfrom object detection.
YOLOv8 is computer vision model architecture developed by Ultralytics. It has features like 
 - Mosaic Data Augementation, mixes 4 images for context information and stops in last 10 epoch to avoid vanishing gradient
   problem.
 - Anchor-Free Dectection, directly predicts an object’s mid-point and reduces the number of bounding box predictions.
 - C2f Module, concatenates the output of all bottleneck modules instead of just one last bottlenect module(bottleneck module
   consists of bottleneck residual blocks that reduce computational costs in deep learning networks)

### Getting Started
To perfrom object detection on BDD100k Dataset, first download the BDD100k datset from [here](https://drive.google.com/file/d/1NgWX5YfEKbloAKX9l8kUVJFpWFlUO8UT/view). Place the dataset
in your home directory.
Create a directory Object_Detection_BDD
<pre> mkdir Object_Detection_BDD </pre>
Clone this repository, to your home directory too
<pre> mkdir Object_Detection_BDD </pre>
Create a new directory `data` in `Object Detection`, followed with `images` and `labels` directories in `data`.
<pre> cd Object Detection  
      mkdir data  </pre>
Copy `train` and `val` directories from images and labels directories to the `Object Detection` directory that is cloned from the repository
<pre> cp -r \home\assignment_data_bdd.zip\bdd100k_images_100k\bdd100k\images\100k\train \home\Object Detection  
      cp -r \home\assignment_data_bdd.zip\bdd100k_images_100k\bdd100k\images\100k\val \home\Object Detection  </pre>


