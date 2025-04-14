# Object Detection
## BDD Dataset
### Introduction
BDD100K Dataset consists of 100,000 videos and images of diverse driving scenarios in various locations in the US. The diversity of the scenarios is across the weather, time of the day, terrains and locations. The dataset consists scenarios that are rainy, gloomy, sunny, during the dawn or dusk, day and night. The also include urban and rural road scenarios.
This would clearly make the BDD Dataset ideal for computer vision as well as autonomous driving applications with wide range of data available under a single shed.

### Data Analysis
The image data is used in this project for object detection purpose. The BDD100k dataset has a training set with 70k images, validation set with 10k images and test set with 20k images. The resolution of the images is 1280x720 pixels. The images in this dataset are labeled to represent the vehicles like cars, buses, trucks, trains, motorcycles and bikes. The object classes also include persons, riders, traffic lights, road signs, drivable areas, crosswalks lanes, sidewalks etc. found in each scenario. These labels in the form of .json files describe the scenario with objects in bounding boxes, along with their class. The lanes, drivabable areas, road curb, cross walks, side walks are represented in the polynomial shape they exist in, unlike the bounding boxes. The label information also includes the weather, scene type and time of the day. The bounding boxes coordinates are represented as `x1,y1,x2,y2`. This dataset with large number of diverse scenarios and labels would train a robust model that fits well with the unexpected outputs irrespective of weather, time and unseen roads. 

Each image mostly has more than one class object and more than one of same class objects. The statistics provided below shows the number of occurences of all the objects in the whole dataset. This representation clearly points out that the number of occurences of cars is 3 times more than next most occured class that is Traffic Signs. Trains, motorcycles and riders are occured very few times when compared to other classes. This would effect the model training with high bias towards cars, that suggests cars would be detected correctly and the other classes like trains, riders and motorcycles might not be detected due to less occurances in the dataset.

<img width="428" alt="image" src="https://github.com/user-attachments/assets/ebdd5f4d-3004-4364-90e9-1e087e0115c7" />  
 

## Model Selection
Object detection combines both object recognition and object localization in an image. The model requires to find all the relevant objects, classify them and place them in a bounding box according to its estimated size. For easy and quicker model training, transfer learning is recommended. This enables to use an existing model already trained on large datasets of similar data and object classes. It would simplify the training by already finding the right parameters for the network like network architecture, optimizers, bias variance trade-offs, activation functions and learning rate.

The object detection models has evolved over many years, some of them are
- Viola-Jones Detector
- HOG Detector (Histogram of Oriented Gradients)
- R-CNN (Region-Convolutional Neural Network) and Fast R-CNN
- YOLO (You only Look Once)

Some models seperate the tasks into two stages whereas others combine them into one step. The two-stage detectors in first stage identify the region of the object using conventional deep networks. Second stage involves object recognition based on features extracted from the proposed region with bounding-box regression. Examples of two-stage detectors are RCNN, Faster RCNN or G-RCNN. These models find a region of interest and use this cropped region for classification. They achieve the highest detection accuracy but are typically slower. 

One-stage detectors predict bounding boxes over the images without the region proposal step. This process consumes less time 
and can therefore be used in real-time applications. They are super fast but not as good at recognizing irregularly shaped objects or a group of small objects. Examples of one-stage detectors are YOLO, SSD, and RetineNet. Single-stage algorithms include generally faster detection speed and greater structural simplicity and efficiency.

<img width="449" alt="image" src="https://github.com/user-attachments/assets/2a6d469b-b5e9-42b5-8136-b5c75be6d636" />
<img width="449" alt="image" src="https://github.com/user-attachments/assets/d9f97eb6-85c4-4fa8-9da6-fb7277042135" />  

The latest real-time detectors are YOLOv7 (2022), YOLOR (2021), and YOLOv4-Scaled (2020). The first image above represents the performance of various detectors and their average precision of the bounding box coordinates of the detected objects. The second image represents the speed of the detectors as frames per second, where the objects are detected. YOLOv7 has outperformed other models in both accuracy(average precision) as well as inference time.

## YOLOv8
### YOLO Introduction
YOLO uses an end-to-end neural network that predicts bounding boxes and class probabilities all at once. It perfroms all the predictions with a single fully connected layer and processes an image in a single iteraction.
<img width="497" alt="image" src="https://github.com/user-attachments/assets/87e2aa84-714d-4d2c-a9ca-05327c3235d9" />  

The image above shows the architecture of a YOLO model. It includes 20 convolutional layers pre-trained using ImageNet, this pre-trained model is converted to perform detection with fully connected layers to predict class probabilities and bounding box coordinates. YOLO uses Non-maximum suppression (NMS) as post-processing step to improve the accuracy and efficiency of object detection.
### YOLOv8 Features
The latest YOLOv8 has shown smaller precision and faster object detection. Hence, it is an ideal model to perfrom object detection. YOLOv8 is a computer vision model architecture developed by Ultralytics. It has features like 
 - Mosaic Data Augementation, mixes 4 images for context information and stops in last 10 epoch to avoid vanishing gradient
   problem.
 - Anchor-Free Dectection, directly predicts an object’s mid-point and reduces the number of bounding box predictions.
 - C2f Module, concatenates the output of all bottleneck modules instead of just one last bottlenect module(bottleneck module
   consists of bottleneck residual blocks that reduce computational costs in deep learning networks)

With the all the discussed models, the pre-trained model yolov8n.pt is used in this object detection task, for its efficiency, simplicity, speed and high average precision. 

## Getting Started
To perfrom object detection on BDD100k Dataset, first download the BDD100k images and labels from [here](http://bdd-data.berkeley.edu/download.html) by clicking on `100k Images` and `100k Labels`. Place the dataset in your home directory.  
Clone this repository, to your home directory too.
```
git clone https://github.com/HimaObugari/Object_detection.git
cd Object_detection
```
Copy `train` and `val` directories from the BDD Dataset `\home\bdd100k_images_100k\100k` and `\home\bdd100k_images_100k\100k` to directories `\home\Object_detection\data\images` and `\home\Object_detection\data\labels_json`
```
cp -r \home\bdd100k_images_100k\100k\train \home\Object_detection\data\images
cp -r \home\bdd100k_images_100k\100k\val \home\Object_detection\data\images
cp -r \home\bdd100k_labels\100k\train \home\Object_detection\data\labels_json
cp -r \home\bdd100k_labels\100k\val \home\Object_detection\data\labels_json
```
## Data preprocessing
To train a model to detect objects in BDD100k dataset using yolov8n.pt model, the labels are to be converted from `.json` files to `.txt` files with Class IDs of the objects. The `main.py` file converts the labels from `.json` files to `.txt` files. The `.txt` files for each label are formatted as:
```
<object1_class_id> <normalized_x_center> <normalized_y_center> <normalized_box_width> <normalized_box_height> 
<object2_class_id> <normalized_x_center> <normalized_y_center> <normalized_box_width> <normalized_box_height> </pre>
```
- x_center = (x1+x2)/2, y_center = (y1+y2)/2
- box_width = |x2-x1|, box_height = |y2-y1|
These values are normalized since the input images are normalized in YOLO model to improve the contrast among the pixels making it easier for the model to extract features for object classification. Normalized values of object centers and bounding box size would match the normalized images in the YOLO model.
Normalizing these values would look like:
- normalized_x_center = x_center/image_width, normalized_y_center = y_center/image_height
- normalized_box_width = box_width/image_width, normalized_box_height = box_height/image_height

The `.txt` files are saved in `\home\Object_detection\data\labels\train` and `\home\Object_detection\data\labels\val` folders. A `data.yaml` file is created to direct the locations of images of training and validation set along with object classes and their IDs correlating to the ones provided in the `.txt` label files.

The side-walks, crosswalk lanes, drivable areas are discarded from detectable objects in this project to completely stick to the object detection that are represented in bounding boxes, hence they are assumed as background. The object classes made to the final cut and their class IDs are: 
 - Car(class_id: 0)
 - Human(class_id: 1)
 - Bus(class_id: 2)
 - Truck(class_id: 3)
 - Bicycle(class_id: 4)
 - Rider(class_id: 5)
 - Motorcycle(class_id: 6)
 - Train(class_id: 7)
 - Traffic Light(class_id: 8)
 - Traffic Sign(class_id: 9)

Due to the availablity of a less powerful system with no GPU, the model training is performed on small part of the dataset to work easily and quickly as well as to avoid crashing the system and model development. Only 550 images are used for training the model and it is validated on 10% of the train set size which is 55 images. 

## Model Training in Docker Container

To perform object detection in a docker container, a container is to be build and run in the `Object_detection` folder. `run_docker.sh` file holds commands to build and run the docker container. Provide a simple command to build and start the container
```
./run_docker.sh
```
After the build and run, in the container run the `main.py` file to convert `.json` files to `.txt` files as well as place them in the required folders.
```
python main.py
```
The images and labels are ready in their designated folders. yolov8n.pt is used for a small dataset. The image size is modified to 640x640 to reduce the burden on the system. yolov8n.pt uses an AdamW optimzer with learning rate 0.000833. The number of classes yolov8n.pt can detect are 80, for BDD100k which is mostly road scenarios, the number of classes chosen are 10. Train the images with pre-trained model yolov8n.pt using the following command:
```
yolo task=detect mode=train model=yolov8n.pt data=/app/data.yaml epochs=10 imgsz=640
```
The resulted model from the training is saved in `/app/runs/detect/train/weights/best.pt` and other parameters and results for different batches and epochs are located in `/app/runs/detect/train` directory(Always check the name of the folder `train`, when the model is trained multiple times, `train2`, `train3` would be created to load the latest trained model).  

Validate the new model with 
```
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=/app/data.yaml
```
To perform the object detection on unseen images from the `val` folder, use the command
```
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=data/images/val
```
The detected images are stored in docker /app/runs/detect/predict. To view the results, they can be copied to the home system with:
```
cp <container_id>:/app/runs/detect/predict ./predict
```
This command saves the object detected images in `Object_detection` folder in your system. These predicted images would provide a visual understanding of the model created. One can verify what all objects are detected correctly and etc.

## Evaluation

In the field of object detection, there exists multiple performance measures and metrics to evaluate the model created. Few of them are:
- Intersection Over Union(IoU), intersection of two bounding boxes, the predicted box and the ground truth box
- Precision and Recall (P and R), Precision focuses on identifying relevant objects and recall focuses on the  model's capability to find all ground truth bounding boxes.
- Average Precision (AP), it is considered as fundamental metric of object detection. It combines the precision and recall seperately for each class in the image
- Mean Average Precision (mAP), finds the mean of AP across multiple classes within the same image
- F1 Score, is a harmonic mean of both precision and recall

A True Positive(TP) would be a correct identification of an object along with a bounding box. A True Negative(TN) would be no identification as well as no bounding box for backgroud. False Positive(FP) could be a wrong classification of the class of the object(example: car identified as traffic sign), whereas a False Negative(FN) would be no bounding box given to a relevant object (example: car in the image is not represented in a bounding box)

Road scenarios as in BDD100k Dataset hold crucial role in autonomous driving applications. Both incorrect classification as well as incorrect localization would have negative impacts, where the decisions of motion of the vehicle have higher stakes. This kind of applications cannot tolerate False Negatives and False Positives. Therefore, both Precision and Recall are suitable metrics for this model. 
 - Precision = TP/(TP+FP)
 - Recall = TP/(TP+FP)

## Result Analysis
The Precision and Recall representations of the resulted model are shown below:
![P_curve](https://github.com/user-attachments/assets/cd1652f1-7bd0-4f53-9a0e-a449c9d06c45)

The Precision is plotted with respect to the confidence of the model on the object recognized. The P-curve plot suggests that the precision is high when the confidence is high for cars. This means that the the cars are mostly recognized correctly by the model. The precision grew with confidence even for Humans. But other classes faced huge abnormalities at different confidence levels. Bicycle and Riders could not possess high confidence levels. On an average the precision of all classes has improved along with confidence of the model on that detection.

![R_curve](https://github.com/user-attachments/assets/9a2437f7-c640-47fc-a7c6-2de244486709)

The R-curve plots show a different performance, where with increase in confidence of the model prediction, the recall value reduced for all of the classes. This represents many objects in the image are not detected at all. This case is not so severe with cars, but well affected classes are traffic light and traffic sign. Average Recall of all classes is also quite low. 

## Discussion

The performance of a model highly depends on data quality like, bias towards an object that occurs more frequently. As already discussed during the Data Analysis, due to the high number of occurences of cars in the dataset, the model detected them better compared to other object classes. To improve the detection of other classes, data with almost similar occurences of other classes would avoid the bias. 

The performance of the model also depends on the data quantity, less data would underfit the model and does not perform valid detection. Dataset reduced to work with a smaller system resulted in lower occurences of many object classes. This affected the detection of most of the classes Therefore more data that is less biased would detect objects much better in unseen images. 

Powerful systems and GPUs would make it easier to process the training on large datasets. High resolutions of the images would also detect objects better. Due to the reduced resolution of the images, the traffic lights and traffic signs were not detected in some images resulting low Recall value. More time and investigation would help identifying the misplacements and incorrect recognitions, anomalies and patterns causing the wrong detections. 





