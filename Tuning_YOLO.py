import os
import random
import shutil

def split_dataset(root_dir, images_subdir='images', train_ratio=0.8):
    """
    Reorganize a dataset of class subfolders into 'train/' and 'val/' directories
    with an 80/20 split (adjustable).

    Args:
        root_dir (str): The path to the dataset root (e.g., '/content/food41').
        images_subdir (str): The subfolder containing class folders (default 'images').
        train_ratio (float): Ratio of images to go into the 'train' set (default 0.8).

    Returns:
        None. Creates 'train/' and 'val/' directories under `root_dir`.
    """
    images_dir = os.path.join(root_dir, images_subdir)
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')

    # Create the train and val directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # List all class subfolders in images_dir
    classes = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    
    for cls in classes:
        class_dir = os.path.join(images_dir, cls)
        # Create corresponding subdirs in train and val
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

        # List all image files in this class
        all_images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        # Shuffle to ensure random distribution
        random.shuffle(all_images)

        # Compute split index
        split_index = int(len(all_images) * train_ratio)
        train_images = all_images[:split_index]
        val_images = all_images[split_index:]

        # Move or copy files to train subfolder
        for img in train_images:
            src_path = os.path.join(class_dir, img)
            dst_path = os.path.join(train_dir, cls, img)
            shutil.move(src_path, dst_path)  # Use move or copy2

        # Move or copy files to val subfolder
        for img in val_images:
            src_path = os.path.join(class_dir, img)
            dst_path = os.path.join(val_dir, cls, img)
            shutil.move(src_path, dst_path)  # Use move or copy2

    print("Dataset successfully split into 'train' and 'val' directories.")

# Example usage:
if __name__ == "__main__":
    root_directory = "/content/food41"  # Adjust to your dataset root
    split_dataset(root_dir=root_directory, images_subdir='images', train_ratio=0.8)


import os
import shutil

def move_train_val_folders(src_root, dst_root):
    """
    Move the 'train' and 'val' folders from src_root to dst_root.
    Creates dst_root if it doesn't exist.
    """

    # Create the destination root if it doesn't exist
    os.makedirs(dst_root, exist_ok=True)

    # Define source paths
    train_src = os.path.join(src_root, 'train')
    val_src = os.path.join(src_root, 'val')

    # Define destination paths
    train_dst = os.path.join(dst_root, 'train')
    val_dst = os.path.join(dst_root, 'val')

    # Move the train folder
    if os.path.isdir(train_src):
        shutil.move(train_src, train_dst)
        print(f"Moved train folder from '{train_src}' to '{train_dst}'.")
    else:
        print(f"train folder not found at '{train_src}'.")

    # Move the val folder
    if os.path.isdir(val_src):
        shutil.move(val_src, val_dst)
        print(f"Moved val folder from '{val_src}' to '{val_dst}'.")
    else:
        print(f"val folder not found at '{val_src}'.")

# Example usage:
if __name__ == "__main__":
    source_root = "/content/food41"       # Where 'train/' and 'val/' currently reside
    destination_root = "/content/new_dataset"  # Where you want to move them
    move_train_val_folders(source_root, destination_root)
from ultralytics import YOLO
import tensorflow as tf

model = YOLO("/content/yolov8n-cls.pt")


model.train(
    task="classify",
    data="/content/new_dataset", 
    epochs=10,
    imgsz=128,  # Reduced image size
    batch=64,   # Increased batch size
    cache=True, # Dataset caching
)