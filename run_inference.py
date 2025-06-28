import os
import ast
import math
import torch
import numpy as np
import pandas as pd
import cv2
import onnxruntime as ort
from numpy.linalg import norm
from pathlib import Path

# === CONFIG ===
DEVICE = torch.device("cpu")
NUM_TRIPLETS = 1592
LABELS = { # labels
    "labels": "data/labels.csv", # original cropping method
}

IMAGE_DIRS = { # images
    "ValidTriplets": "data/ValidTriplets", 
}

FERPLUS_PATH = "ferplus/emotion-ferplus-multi-output.onnx"
LAYER_OUTPUTS = [
    "ReLU384_Output_0",
    "ReLU496_Output_0",
    "ReLU636_Output_0",
    "ReLU670_Output_0",
    "Plus692_Output_0"
]

DATASET_CSV = "data/faceexp-comparison-data-train-public.csv"

# === HELPERS ===
def load_image_cv2(path): # load image from cv2
    return cv2.imread(path)

def compute_cosine(a, b): # cosine similarity
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-12)

def compute_pearson(a, b): # pearson 
    return np.corrcoef(a, b)[0, 1]

def compute_l2(a,b): #L2 Norm
    return np.linalg.norm(a-b)


""""
def parse_label(raw): # get integer representing label
    try:
        raw = str(raw).strip()
        if raw.startswith('['):
            label_list = ast.literal_eval(raw)
            if isinstance(label_list, list) and len(label_list) == 1:
                res = int(label_list[0]) - 1
                return res
        res = int(raw) - 1
        return res
    except:
        return None
        """

def extract_original_row_index(filename): # get original triplet index
    try:
        parts = filename.split('_')
        if len(parts) >= 3:
            return int(parts[2].split('.')[0])
    except:
        pass
    return None

def get_fer_8_plus_representation(img, session):
    """
    Given a cropped BGR face image, preprocesses it and runs it through the FER+ model.
    
    Parameters:
        cropped_face (np.ndarray): BGR face image as a NumPy array.
        
    Returns:
        np.ndarray: The output logits or probabilities from the FER+ model (shape: [8]).
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize to model's input size
    resized = cv2.resize(gray, (64, 64))
    # Reshape to match model input: (1, 1, 64, 64)
    blob = resized.astype(np.float32).reshape(1, 1, 64, 64)
    # Set input and perform inference
    input_name = session.get_inputs()[0].name 
    output = session.run(None, {input_name: blob})[0]
    return output[0]

def get_features_ferplus(img, session):
    # Convert RGB to grayscale (FERPlus expects grayscale)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Resize to 64x64 as expected by FERPlus
    img = cv2.resize(img, (64, 64)).astype(np.float32)
    # Normalize to [-0.5, 0.5]
    img = (img - 127.5) / 255.0
    # Reshape to (1, 1, 64, 64)
    img = img[np.newaxis, np.newaxis, :, :]
    # Get all outputs
    input_name = session.get_inputs()[0].name
    # The second output (index 1) contains the 1024-dimension features
    features = session.run([session.get_outputs()[1].name], {input_name: img})[0]
    print(features)
    # Flatten first, then normalize
    features_flat = features.flatten()
    mean = np.mean(features_flat)
    std = np.std(features_flat)
    features_norm = (features_flat - mean) / (std + 1e-12)  # Add small epsilon to prevent division by zero
    return features_norm

def get_layer_features(img, session):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    blob = resized.astype(np.float32).reshape(1, 1, 64, 64)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    return {name: out[0].flatten() for name, out in zip(LAYER_OUTPUTS, outputs)}

def computations_l2(features, row):
    sim_23 = compute_l2(features[1], features[2])
    sim_13 = compute_l2(features[0], features[2])
    sim_12 = compute_l2(features[0], features[1])
    sims = [sim_23, sim_13, sim_12]
    predicted = sims.index(min(sims))
    label = int(row[4])
    is_correct = predicted == label
    return [sim_23, sim_13, sim_12, label, is_correct]

# === MAIN FUNCTION ===
def run_inference_and_save(dataset_name):
    print("started main function")
    image_dir = IMAGE_DIRS[dataset_name]
    labels_path = LABELS[dataset_name]
    session = ort.InferenceSession(FERPLUS_PATH)
    triplets_df = pd.read_csv(labels_path, header=None).drop_duplicates().head(NUM_TRIPLETS)
    results = {layer: [] for layer in LAYER_OUTPUTS}
    for i, row in triplets_df.iterrows():
        img_paths = [os.path.join(image_dir, os.path.basename(row[j])) for j in range(3)]
        imgs = [load_image_cv2(p) for p in img_paths]
        if any(img is None for img in imgs):
            continue
        features_by_layer = [get_layer_features(img, session) for img in imgs]
        for layer in LAYER_OUTPUTS:
            features = [f[layer] for f in features_by_layer]
            res = computations_l2(features, row)
            results[layer].append([i] + res)
    for layer in LAYER_OUTPUTS:
        df = pd.DataFrame(results[layer], columns=["triplet_id", "sim_23", "sim_13", "sim_12", "label", "is_correct"])
        df.to_csv(f"results_{layer}.csv", index=False)
        print(f"Saved results for {layer}")

# === RUN ALL COMBINATIONS ===
if __name__ == "__main__":
    print("Initializing Inference")
    for dataset in ["ValidTriplets"]:
        print("Running inference on", dataset)
        run_inference_and_save(dataset)
    
    

