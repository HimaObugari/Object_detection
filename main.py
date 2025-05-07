from pathlib import Path
import json
import cv2
import os
import random
import shutil

def json_to_txt(label_files_paths, images_shape):
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
    labels_dir = 'data/all_labels'
    os.makedirs(labels_dir, exist_ok=True)
    for i in range(len(label_array)):
        with open(f"data/all_labels/{label_files_paths[i].stem}.txt", "w") as file:
            for row in label_array[i]:
                file.write(" ".join(map(str, row)) + "\n")

def train_val_split(source_images,source_labels,train_images,train_labels,val_images, val_labels):
    # Create dirs if not exist
    for folder in [train_images, train_labels, val_images, val_labels]:
        os.makedirs(folder, exist_ok=True)

    # List of files
    all_files = [f for f in os.listdir(source_images) if f.endswith(('.jpg', '.png'))]
    random.shuffle(all_files)

    # Split
    split_ratio = 0.9  # 80% train, 20% val
    split_idx = int(len(all_files) * split_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    return train_files,val_files
    

def move_files(source_images, source_labels,file_list, target_img_dir, target_lbl_dir):
    for f in file_list:
        img_src = os.path.join(source_images, f)
        lbl_src = os.path.join(source_labels, f.replace('.jpg', '.txt').replace('.png', '.txt'))

        shutil.copy2(img_src, os.path.join(target_img_dir, f))
        if os.path.exists(lbl_src):
            shutil.copy2(lbl_src, os.path.join(target_lbl_dir, os.path.basename(lbl_src)))

def run_test():
    folder_path_labels = Path(r"data/all_labels_json")
    folder_path_images = Path(r"data/all_images")
    image_files = list(folder_path_images.glob("*.jpg"))
    images = [cv2.imread(str(i)) for i in image_files]
    images_shape = [i.shape for i in images]
    label_files_paths = list(folder_path_labels.glob("*.json"))
    json_to_txt(label_files_paths, images_shape)
    source_images = 'data/all_images'
    source_labels = 'data/all_labels'
    train_images = 'data/images/train'
    train_labels = 'data/labels/train'
    val_images = 'data/images/val'
    val_labels = 'data/labels/val'
    train_files,val_files=train_val_split(source_images, source_labels, train_images,train_labels,val_images,val_labels)
    move_files(source_images,source_labels,train_files, train_images, train_labels)
    move_files(source_images,source_labels,val_files, val_images, val_labels)

if __name__ == "__main__":
    run_test()
