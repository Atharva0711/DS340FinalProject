import os
import shutil
import random
import argparse

def split_dataset(input_root, train_root, test_root, train_ratio=0.8, seed=42):
    """
    Splits a directory of track folders into train/ and test/ subfolders.

    Args:
      input_root  (str): path containing all your MoisesDB track dirs
      train_root  (str): output path for train split
      test_root   (str): output path for test split
      train_ratio (float): fraction of tracks to put in train (rest → test)
      seed        (int):   random seed for reproducibility
    """
    # 1. gather all track IDs
    tracks = [d for d in os.listdir(input_root)
              if os.path.isdir(os.path.join(input_root, d))]
    if not tracks:
        raise ValueError(f"No subdirectories found in {input_root}")

    # 2. shuffle
    random.seed(seed)
    random.shuffle(tracks)

    # 3. split
    n_train = int(len(tracks) * train_ratio)
    train_tracks = set(tracks[:n_train])
    test_tracks  = set(tracks[n_train:])

    # 4. create output dirs
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(test_root,  exist_ok=True)

    # 5. copy each track folder
    for tid in train_tracks:
        src = os.path.join(input_root, tid)
        dst = os.path.join(train_root,  tid)
        shutil.copytree(src, dst)
    for tid in test_tracks:
        src = os.path.join(input_root, tid)
        dst = os.path.join(test_root,   tid)
        shutil.copytree(src, dst)

    # 6. report
    print(f"Total tracks: {len(tracks)}")
    print(f" → Train ({len(train_tracks)}): {train_ratio*100:.0f}%")
    print(f" → Test  ({len(test_tracks)}): {(1-train_ratio)*100:.0f}%")
    print(f"Train data saved to: {train_root}")
    print(f"Test  data saved to: {test_root}")

if __name__ == "__main__":
    
    split_dataset(
        input_root  = "./processed/moisesdb",
        train_root  = "./processed/moisesdb_final/train",
        test_root   = "./processed/moisesdb_final/test",
        train_ratio = 0.83,
        seed        =  1234
    )
