import os
import argparse
from tqdm import tqdm

CLASS_MAP = {
  "Car": 0,
  "Pedestrian": 1,
  "Cyclist": 2
}

# KITTI image resolution
IMG_WIDTH  = 1242
IMG_HEIGHT = 375

def convert_kitti_to_yolo(kitti_label_path, yolo_label_path):
  """
  Converts a single KITTI annotation file to the YOLO format

  Returns:
    bool: True if the conversion was successful and was written
  """
  yolo_lines = []
  with open(kitti_label_path, 'r') as f:
    kitti_lines = f.readlines()

  for line in kitti_lines:
    parts = line.strip().split()
    class_name = parts[0]

    if class_name not in CLASS_MAP:
      continue
  
    class_id = CLASS_MAP[class_name]

    # Extract 2D bounding box (left, top, right, bottom)
    # These fields are 4, 5, 6, 7 in the KITTI label format
    try:
      x1 = float(parts[4])
      y1 = float(parts[5])
      x1 = float(parts[6])
      y1 = float(parts[7])
    except (IndexError, ValueError) as e:
      print(f"Warning: Could not parse bounding box in {kitti_label_path}. Line: '{line}'. Error: {e}")
      continue

    # YOLO Format Conversion
    # 1. Calculate bounding box center
    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2
    # 2. Calculate bounding box width and height
    box_width = x2- x1
    box_height = y2- y1
    # 3. Normalize coordinates by image dimensions
    norm_center_x = box_center_x / IMG_WIDTH
    norm_center_y = box_center_y / IMG_HEIGHT
    norm_width = box_width / IMG_WIDTH
    norm_height = box_height / IMG_HEIGHT

    # Format the line for the YOLO .txt file
    # Format: class_id center_x center_y width height
    yolo_line = f"{class_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
    yolo_lines.append(yolo_line)

  # Write the converted annotations to the new file
  if yolo_lines:
      with open(yolo_label_path, 'w') as f:
          f.write("\n".join(yolo_lines))
      return True
  return False
    

def process_split(split, kitti_dir, output_dir, splits_dir):
  kitti_image_dir = os.path.join(kitti_dir, "image_2")
  kitti_label_dir = os.path.join(kitti_dir, "label_2")
  split_file = os.path.join(splits_dir, f"{split}.txt")

  yolo_image_dir = os.path.join(output_dir, "images", split)
  yolo_label_dir = os.path.join(output_dir, "labels", split)
  os.makedirs(yolo_image_dir, exist_ok=True)
  os.makedirs(yolo_label_dir, exist_ok=True)

  # Convert split file in frame ID list
  with open(split_file, 'r') as f:
    frame_ids = [line.strip() for line in f.readlines()]

  # Process each frame
  for frame_id in tqdm(frame_ids, desc=f"Converting {split} labels"):
    # Convert Labels
    kitti_label_path = os.path.join(kitti_label_dir, f"{frame_id}.txt")
    yolo_label_path = os.path.join(yolo_label_dir, f"{frame_id}.txt")
    convert_kitti_to_yolo(kitti_label_path, yolo_label_path)

    # Create symbolic link for images
    source_image_path = os.path.join(kitti_image_dir, f"{frame_id}.png")
    dest_image_path = os.path.join(yolo_image_id, f"{frame_id}.png")

    # Check if symlink already edxists to prevent errors on re-runs
    if not os.path.lexists(dest_image_path):
      os.symlink(os.path.abspath(source_image_path), dest_image_path)

    print(f"Finished processing '{split}' split.")
    print(f"Converted labels are in: {yolo_label_dir}")
    print(f"Image symlinks are in: {yolo_image_dir}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--kitti_root",
    type=str,
    default="data/raw_kitti",
    help="Path to root dir of raw KITTI dataset"
  )
  parser.add_argument(
    "--yolo_output",
    type=str,
    default="data/processed/yolo_format",
    help="Path to the output directory for YOLO-formatted data" 
  )
  parser.add_argument(
    "--splits_dir",
    type=str,
    default="data/processed/splits",
    help="Path to the directory containing train.txt and val.txt"
  )
  args = parser.parse_args()

  process_split("train", args.kitti_root, args.yolo_output, args.splits_dir)
  process_split("val", args.kitti_root, args.yolo_output, args.splits_dir)

  print("\nNext steps:")
  print("1. Create the `data.yaml` file for YOLOv5.")
  print("2. Run the YOLOv5 training script.")