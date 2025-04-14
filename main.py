from pathlib import Path
import json
import pandas
import cv2
import yaml
from ultralytics import YOLO

def json_to_txt(set_name, label_files_paths, images_shape):
    label_data = []
    for i in label_files_paths:
        with open(str(i)) as json_file:
            label_data.append(json.load(json_file))
    actual_objects=[0 for i in range(len(label_data))]
    for i in range(len(label_data)):
        for j in range(len(label_data[i]["frames"][0]["objects"])):
            if label_data[i]["frames"][0]["objects"][j].get("box2d"):
                actual_objects[i]=actual_objects[i]+1
    label_array = [[[0 for j in range(5)] for i in range(actual_objects[k])] for k in range(len(label_data))]
    for j in range(len(label_data)):
        for i in range(actual_objects[j]):
            if label_data[j]["frames"][0]["objects"][i].get("box2d"):
                x1 = label_data[j]["frames"][0]["objects"][i]["box2d"]["x1"]
                y1 = label_data[j]["frames"][0]["objects"][i]["box2d"]["y1"]
                x2 = label_data[j]["frames"][0]["objects"][i]["box2d"]["x2"]
                y2 = label_data[j]["frames"][0]["objects"][i]["box2d"]["y2"]
                box_width = abs(x2-x1)
                box_height = abs(y2-y1)
                x_center_norm = (x1+x2)/(2*images_shape[0][1])
                y_center_norm = (y1+y2)/(2*images_shape[0][0])
                box_width_norm = box_width/images_shape[0][1]
                box_height_norm = box_height/images_shape[0][0]
                
                if label_data[j]["frames"][0]["objects"][i]["category"]=='car':
                    class_id=0
                elif label_data[j]["frames"][0]["objects"][i]["category"]=='person':
                    class_id=1
                elif label_data[j]["frames"][0]["objects"][i]["category"]=='bus':
                    class_id=2
                elif label_data[j]["frames"][0]["objects"][i]["category"]=='truck':
                    class_id=3
                elif label_data[j]["frames"][0]["objects"][i]["category"]=='bike':
                    class_id=4
                elif label_data[j]["frames"][0]["objects"][i]["category"]=='rider':
                    class_id=5
                elif label_data[j]["frames"][0]["objects"][i]["category"]=='motor':
                    class_id=6
                elif label_data[j]["frames"][0]["objects"][i]["category"]=='train':
                    class_id=7
                elif label_data[j]["frames"][0]["objects"][i]["category"]=='traffic sign':
                    class_id=8
                elif label_data[j]["frames"][0]["objects"][i]["category"]=='traffic light':
                    class_id=9
                label_array[j][i]=[class_id, x_center_norm, y_center_norm, box_width_norm, box_height_norm]
    for i in range(len(label_array)):
        with open(f"data/labels/{set_name}/{label_files_paths[i].stem}.txt", "w") as file:
            for row in label_array[i]:
                file.write(" ".join(map(str, row)) + "\n")

def run_test():
    folder_path_labels_train = Path(r"data/labels_json/train")
    folder_path_labels_val = Path(r"data/labels_json/val")
    folder_path_images_train = Path(r"data/images/train")
    folder_path_images_val = Path(r"data/images/val")
    image_files_t = list(folder_path_images_train.glob("*.jpg"))
    image_files_v = list(folder_path_images_val.glob("*.jpg"))
    images_t = [cv2.imread(str(i)) for i in image_files_t]
    images_v = [cv2.imread(str(i)) for i in image_files_v]
    images_shape_t = [i.shape for i in images_t]
    images_shape_v = [i.shape for i in images_v]
    label_files_paths_t = list(folder_path_labels_train.glob("*.json"))
    label_files_paths_v = list(folder_path_labels_val.glob("*.json"))
    set_name_train = 'train'
    set_name_val = 'val'
    label_data_t=json_to_txt(set_name_train, label_files_paths_t, images_shape_t)
    label_data_v=json_to_txt(set_name_val, label_files_paths_v, images_shape_v)
      

if __name__ == "__main__":
    run_test()
