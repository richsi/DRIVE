import os
import argparse
import random

def create_splits(num_frames, train_ratio, output_dir):
  """
  Generates and saves training and validation split files.

  Args:
    num_frames (int): The total number of frames in the dataset (e.g., 7481 for KITTI)
    train_ratio (float): The proportion of the dataset to be used for training
    output_dir (str): The directory where the split files will be saved
  """

  os.makedirs(output_dir, exist_ok=True)

  # Generate a list of frame IDs as zero-padded strings
  # e.g., "000000", "000001", ..., "007480"
  frame_ids = [f"{i:06d}" for i in range(num_frames)]

  random.shuffle(frame_ids)

  # Splitting the ids
  split_index = int(len(frame_ids) * train_ratio)
  train_ids = frame_ids[:split_index]
  val_ids = frame_ids[split_index:]

  train_file_path = os.path.join(output_dir, "train.txt")
  with open(train_file_path, "w") as f:
    f.write("\n".join(train_ids))
  print(f"Successfully created train.txt with {len(train_ids)} entries at {train_file_path}")

  val_file_path = os.path.join(output_dir, "val.txt")
  with open(val_file_path, 'w') as f:
      f.write("\n".join(val_ids))
  print(f"Successfully created val.txt with {len(val_ids)} entries at {val_file_path}")
  print("-" * 30)
  print(f"Total frames: {num_frames}")
  print(f"Train/Val split: {len(train_ids)}/{len(val_ids)}")




if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--num_frames",
    type=int,
    default=7481,
    help="Total number of frames in the dataset"
  )
  parser.add_argument(
    "--train_ratio",
    type=float,
    default=0.8,
    help="Ratio o fdata to be used for training (e.g., 0.8 for 80%)"
  )
  parser.add_argument(
    "--output-dir",
    type=str,
    default="data/processed/splits",
    help="Directory to save the train.txt and val.txt files"
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for shuffling to ensure reproducibility"
  )

  args = parser.parse_args()

  random.seed(args.seed)

  create_splits(args.num_frames, args.train_ratio, args.output_dir)