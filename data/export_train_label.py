import pandas as pd
import os
import json
from collections import Counter

def grade_mode(grades):
    # Convert to int, ignore NaNs or invalid
    grades = [int(g) for g in grades if str(g).isdigit()]
    if not grades:
        return None
    return Counter(grades).most_common(1)[0][0]

def create_label1(use_mediapipe=False):
    input_csv = 'data/FEC_dataset/faceexp-comparison-data-train-public.csv'
    mapping_file = 'data/triplet_to_row_mapping.json'
    image_dir = "data/trainCrop" if use_mediapipe else "data/train"
    output_csv = 'data/labels_mediapipe.csv' if use_mediapipe else 'data/labels.csv'

    if not os.path.exists(image_dir):
        print(f"❌ Directory not found: {image_dir}")
        return

    available_files = os.listdir(image_dir)
    print(f"📂 Found {len(available_files)} files in {image_dir}")

    try:
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
            triplet_to_row = {int(k): int(v) for k, v in mapping.items()}
    except:
        triplet_to_row = {}

    try:
        dataset = pd.read_csv(input_csv, header=None, on_bad_lines='skip')
    except Exception as e:
        print(f"❌ Could not read dataset: {e}")
        return

    # Group by triplet
    triplets = {}
    for filename in available_files:
        parts = filename.split('_')
        if len(parts) >= 3:
            try:
                triplet_num = int(parts[0])
                position = int(parts[1])
                original_row = int(parts[2]) if len(parts) >= 4 else None
                if triplet_num not in triplets:
                    triplets[triplet_num] = {'files': {}, 'original_row': original_row}
                triplets[triplet_num]['files'][position] = filename
            except:
                continue

    print(f"🧩 Found {len(triplets)} triplets")

    rows = []
    for triplet_num in sorted(triplets):
        info = triplets[triplet_num]
        files = info['files']

        if not all(pos in files for pos in [1, 2, 3]):
            continue

        original_row = info['original_row']
        if original_row is None and triplet_num in triplet_to_row:
            original_row = triplet_to_row[triplet_num]
        if original_row is None:
            continue

        try:
            similarity_type = dataset.iloc[original_row, 15]  # e.g., ONE_CLASS_TRIPLET
            votes = [
                dataset.iloc[original_row, 17],
                dataset.iloc[original_row, 19],
                dataset.iloc[original_row, 21],
                dataset.iloc[original_row, 23],
                dataset.iloc[original_row, 25],
                dataset.iloc[original_row, 27],
            ]
            label = grade_mode(votes)
            if label not in [1, 2, 3]:
                continue
        except Exception as e:
            print(f"❌ Skipping triplet {triplet_num} due to error: {e}")
            continue

        name1 = os.path.join(image_dir, files[1]).replace('\\', '/')
        name2 = os.path.join(image_dir, files[2]).replace('\\', '/')
        name3 = os.path.join(image_dir, files[3]).replace('\\', '/')

        rows.append([name1, name2, name3, similarity_type, label - 1])

        if len(rows) >= 100:
            break

    if rows:
        pd.DataFrame(rows).to_csv(output_csv, index=False, header=False)
        print(f"✅ Created {output_csv} with {len(rows)} triplets")
    else:
        print("❌ No valid triplets found.")
