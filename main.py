from pathlib import Path
import json
import pandas
import cv2
import yaml
from ultralytics import YOLO


def run_test():
    folder_path_labels = Path(r"data/labels/json_files")
    folder_path_images_train = Path(r"data/images/train")
    folder_path_images_val = Path(r"data/images/val")
    image_files_t = list(folder_path_images_train.glob("*.jpg"))
    image_files_v = list(folder_path_images_val.glob("*.jpg"))
    images_t = [cv2.imread(str(i)) for i in image_files_t]
    images_v = [cv2.imread(str(i)) for i in image_files_v]
    images_shape_t = [i.shape for i in images_t]
    images_shape_v = [i.shape for i in images_v]
    label_files_paths = list(folder_path_labels.glob("*.json"))
    label_data = []
    for i in label_files_paths:
        with open(str(i)) as json_file:
            label_data.append(json.load(json_file))
    label_data_t, label_data_v = [],[]
    for i in range(len(label_files_paths)):
        c=0
        for j in range(len(image_files_t)):
            if label_files_paths[i].stem==image_files_t[j].stem:
                label_data_t.append(label_data[i])
                c=c+1
        if c==0:
            label_data_v.append(label_data[i])
    actual_objects_t=[0 for i in range(len(label_data_t))]
    for i in range(len(label_data_t)):
        for j in range(len(label_data_t[i]["frames"][0]["objects"])):
            if label_data_t[i]["frames"][0]["objects"][j].get("box2d"):
                actual_objects_t[i]=actual_objects_t[i]+1
    label_array_t = [[[0 for j in range(6)] for i in range(actual_objects_t[k])] for k in range(len(label_data_t))]
    for j in range(len(label_data_t)):
        for i in range(actual_objects_t[j]):
            if label_data_t[j]["frames"][0]["objects"][i].get("box2d"):
                x1 = label_data_t[j]["frames"][0]["objects"][i]["box2d"]["x1"]
                y1 = label_data_t[j]["frames"][0]["objects"][i]["box2d"]["y1"]
                x2 = label_data_t[j]["frames"][0]["objects"][i]["box2d"]["x2"]
                y2 = label_data_t[j]["frames"][0]["objects"][i]["box2d"]["y2"]
                box_width = abs(x2-x1)
                box_height = abs(y2-y1)
                x_center_norm = (x1+x2)/(2*images_shape_t[0][1])
                y_center_norm = (y1+y2)/(2*images_shape_t[0][0])
                box_width_norm = box_width/images_shape_t[0][1]
                box_height_norm = box_height/images_shape_t[0][0]
                if label_data_t[j]["frames"][0]["objects"][i]["category"]=='car':
                    class_id=0
                elif label_data_t[j]["frames"][0]["objects"][i]["category"]=='person':
                    class_id=1
                elif label_data_t[j]["frames"][0]["objects"][i]["category"]=='bus':
                    class_id=2
                elif label_data_t[j]["frames"][0]["objects"][i]["category"]=='truck':
                    class_id=3
                elif label_data_t[j]["frames"][0]["objects"][i]["category"]=='bike':
                    class_id=4
                elif label_data_t[j]["frames"][0]["objects"][i]["category"]=='motor':
                    class_id=5
                elif label_data_t[j]["frames"][0]["objects"][i]["category"]=='traffic light':
                    class_id=6
                elif label_data_t[j]["frames"][0]["objects"][i]["category"]=='traffic sign':
                    class_id=7
                label_array_t[j][i]=[class_id, x_center_norm, y_center_norm, box_width_norm, box_height_norm]
    for i in range(len(label_array_t)):
        with open(f"data/labels/train/{image_files_t[i].stem}.txt", "w") as file:
            for row in label_array_t[i]:
                file.write(" ".join(map(str, row)) + "\n")
    actual_objects_v=[0 for i in range(len(label_data_v))]
    for i in range(len(label_data_v)):
        for j in range(len(label_data_v[i]["frames"][0]["objects"])):
            if label_data_v[i]["frames"][0]["objects"][j].get("box2d"):
                actual_objects_v[i]=actual_objects_v[i]+1
    label_array_v = [[[0 for j in range(6)] for i in range(actual_objects_v[k])] for k in range(len(label_data_v))]
    for j in range(len(label_data_v)):
        for i in range(actual_objects_v[j]):
            if label_data_v[j]["frames"][0]["objects"][i].get("box2d"):
                x1 = label_data_v[j]["frames"][0]["objects"][i]["box2d"]["x1"]
                y1 = label_data_v[j]["frames"][0]["objects"][i]["box2d"]["y1"]
                x2 = label_data_v[j]["frames"][0]["objects"][i]["box2d"]["x2"]
                y2 = label_data_v[j]["frames"][0]["objects"][i]["box2d"]["y2"]
                box_width = abs(x2-x1)
                box_height = abs(y2-y1)
                x_center_norm = (x1+x2)/(2*images_shape_t[0][1])
                y_center_norm = (y1+y2)/(2*images_shape_t[0][0])
                box_width_norm = box_width/images_shape_t[0][1]
                box_height_norm = box_height/images_shape_t[0][0]
                if label_data_v[j]["frames"][0]["objects"][i]["category"]=='car':
                    class_id=0
                elif label_data_v[j]["frames"][0]["objects"][i]["category"]=='person':
                    class_id=1
                elif label_data_v[j]["frames"][0]["objects"][i]["category"]=='bus':
                    class_id=2
                elif label_data_v[j]["frames"][0]["objects"][i]["category"]=='truck':
                    class_id=3
                elif label_data_v[j]["frames"][0]["objects"][i]["category"]=='bike':
                    class_id=4
                elif label_data_v[j]["frames"][0]["objects"][i]["category"]=='motor':
                    class_id=5
                elif label_data_v[j]["frames"][0]["objects"][i]["category"]=='traffic light':
                    class_id=6
                elif label_data_v[j]["frames"][0]["objects"][i]["category"]=='traffic sign':
                    class_id=7
                label_array_v[j][i]=[class_id, x_center_norm, y_center_norm, box_width_norm, box_height_norm]
    for i in range(len(label_array_v)):
        with open(f"data/labels/val/{image_files_v[i].stem}.txt", "w") as file:
            for row in label_array_v[i]:
                file.write(" ".join(map(str, row)) + "\n")
    #model = YOLO("yolov8n.pt")
    #model.train(data=f"/app/data.yaml", epochs=5, imgsz=416, batch=4)     
      

if __name__ == "__main__":
    run_test()
