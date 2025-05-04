import os
import ast
import math
import torch
import numpy as np
import pandas as pd
import cv2
import onnxruntime as ort
from numpy.linalg import norm
from models.FECNet import FECNet
from pathlib import Path

# === CONFIG ===
DEVICE = torch.device("cpu")
NUM_TRIPLETS = 100
LABELS = { # labels
    "train": "data/labels.csv", # original cropping method
    "trainCrop": "data/labels_mediapipe.csv" # according to Liron's cropping method
}

IMAGE_DIRS = { # images
    "train": "data/train", # original cropping method
    "trainCrop": "data/trainCrop" # according to Liron's cropping method
}

FERPLUS_PATH = "emotion-ferplus.onnx" # Liron's model 
FECNET_PATH = "FECNet.pt" # new model
DATASET_CSV = "data/FEC_dataset/faceexp-comparison-data-train-public.csv"

# === HELPERS ===
def load_image_cv2(path): # load image from cv2
    return cv2.imread(path)

def compute_cosine(a, b): # cosine similarity
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-12)

def compute_pearson(a, b): # pearson 
    return np.corrcoef(a, b)[0, 1]

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

def get_features_fecnet(img): # returns the vector representation of the image according to original model
    from get_representation import get_FECNet_representation
    return get_FECNet_representation(img)

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

def computations(simCalc, features, row): 
    # compute similarities between each pair of images

    sim_23 = simCalc(features[1], features[2])
    sim_13 = simCalc(features[0], features[2])
    sim_12 = simCalc(features[0], features[1])
    sims = [sim_23, sim_13, sim_12]
    print(sims)
    predicted = sims.index(max(sims))  # generate model prediction
    print(predicted)
    label = int(row[4]) # get corresponding label
    print(label)
    new_score = sims[predicted] - (sum([sim_score for sim_score in sims if sim_score != sims[predicted]]) / 2) 
    is_correct = predicted == label # compare prediction to label
    exp_sims = np.exp(sims) # create array of exponents
    probs = exp_sims / np.sum(exp_sims) # divide each element by the sum of exponents
    prob_correct = probs[predicted] # the likelihood score is the value of the similarity the model predicted 
    log_likelihood = math.log(prob_correct + 1e-12) # add epsilon to avoid log(0)
    res = [round(sim_23, 4), round(sim_13, 4), round(sim_12, 4), label, is_correct, round(log_likelihood, 4), round(new_score, 4)]
    return res

# === MAIN FUNCTION ===
def run_inference_and_save(model_type, dataset_name):
    print("started main function")
    image_dir = IMAGE_DIRS[dataset_name]
    labels_path = LABELS[dataset_name]
    result_rows = []
    cos_total_log_likelihood = 0.0
    cos_total_new_score = 0.0
    cos_accuracy = 0
    pearson_total_log_likelihood = 0.0
    pearson_total_new_score = 0.0
    pearson_accuracy = 0

    try: # create data frames from csv 
        triplets_df = pd.read_csv(labels_path, header=None).drop_duplicates().head(NUM_TRIPLETS)
        class_type_df = pd.read_csv(DATASET_CSV, header=None, low_memory=False)
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return

    if model_type == "FECNet": # load FECNet model and get img vectors
        model = FECNet(pretrained=False)
        try:
            # Try with strict=True to catch any mismatches
            model.load_state_dict(torch.load(FECNET_PATH, map_location=DEVICE), strict=True)
            print("Successfully loaded weights from", FECNET_PATH)
        except Exception as e:
            print(f"Error loading weights strictly: {e}")
            try:
                # Fall back to non-strict loading if strict fails
                model.load_state_dict(torch.load(FECNET_PATH, map_location=DEVICE), strict=False)
                print("Loaded weights with some mismatches")
            except Exception as e:
                print(f"Failed to load weights: {e}")
                print("WARNING: Using randomly initialized weights!")
        
        model.to(DEVICE)
        model.eval()
        get_features = lambda img: get_features_fecnet(img)

    elif model_type == "FERPlus": # load FERPlus model and get img vectors
        session = ort.InferenceSession(FERPLUS_PATH)
        get_features = lambda img: get_fer_8_plus_representation(img, session)
    
    count = 0
    for i, row in triplets_df.iterrows():
        count+=1
        if count % 25 == 0:
            print(str(count) + " iteration") # just to keep track
        try:
            img_paths = [os.path.join(image_dir, os.path.basename(row[j])) for j in range(3)] # creates list of image paths
            imgs = [load_image_cv2(p) for p in img_paths] # loads each image in triplet
            if any(img is None for img in imgs): # if a certain error occurs just skip the triplet
                continue
            features = [get_features(img) for img in imgs]
            # compute scores with both formulas
            cos_res = computations(compute_cosine, features, row) 
            pearson_res = computations(compute_pearson, features, row)
            if not cos_res or not pearson_res:
                print("problem generating data, skipping this triplet")
                continue
            # global scores
            cos_total_log_likelihood += cos_res[5]
            cos_total_new_score += cos_res[6]
            cos_accuracy += 1 if cos_res[4] else 0
            pearson_total_log_likelihood += pearson_res[5]
            pearson_total_new_score += pearson_res[6]
            pearson_accuracy += 1 if pearson_res[4] else 0

             # Get class type from original dataset
            original_index = extract_original_row_index(os.path.basename(img_paths[0]))
            try:
                if original_index is not None and original_index < len(class_type_df):
                    class_type = class_type_df.iloc[original_index, 15] # extract class type 
                else:
                    class_type = "unknown"
            except Exception as e:
                print(f"⚠️ Could not get metadata for index {original_index}: {e}")
                class_type = "unknown"

            # Use local image paths in output
            result_rows.append([ 
                i,
                img_paths[0], img_paths[1], img_paths[2],
                cos_res[3],
                round(cos_res[0], 4),
                round(cos_res[1], 4),
                round(cos_res[2], 4),
                round(cos_res[5], 4),
                round(cos_res[6], 4),
                cos_res[4],
                round(pearson_res[0], 4),
                round(pearson_res[1], 4),
                round(pearson_res[2], 4),
                round(pearson_res[5], 4),
                round(pearson_res[6], 4),
                pearson_res[4],
                class_type
            ]) # collect data of current triplet, rounding the values to 4 digits after decimal

        except Exception as e:
            print(f"❌ Error on triplet {i}: {e}")

    # global scores computation and collection
    print("on to global calculations")
    cos_accuracy = cos_accuracy / len(result_rows) * 100 if result_rows else 0.0
    pearson_accuracy = pearson_accuracy / len(result_rows) * 100 if result_rows else 0.0
    cos_avg_log_likelihood = cos_total_log_likelihood / len(result_rows) if result_rows else 0.0
    pearson_avg_log_likelihood = pearson_total_log_likelihood / len(result_rows) if result_rows else 0.0
    cos_avg_new_score = cos_total_new_score / len(result_rows) if result_rows else 0.0
    pearson_avg_new_score = pearson_total_new_score / len(result_rows) if result_rows else 0.0
    output_name = f"results_{model_type}_{dataset_name}.csv"

    with open(output_name, "w") as f:
        f.write(f"# Method: {model_type} | Dataset: {dataset_name} | Cos: | Accuracy: "
                f"{cos_accuracy:.2f}% | Avg Log-Likelihood: {cos_avg_log_likelihood:.4f} | Avg new score: {cos_avg_new_score:.4f}  | Pearson: | Accuracy: " 
                f"{pearson_accuracy:.2f}% | Avg Log-Likelihood: {pearson_avg_log_likelihood:.4f} | Avg new score: {pearson_avg_new_score:.4f}\n")
        
        df = pd.DataFrame(result_rows, columns=[
            "triplet_id", "img1_path", "img2_path", "img3_path", "most_similar", 
            "c_comp_23", "c_comp_13", "c_comp_12", "c_log_likelihood", "c_new_score", "c_correct",
            "p_comp_23", "p_comp_13", "p_comp_12", "p_log_likelihood", "p_new_score",
            "p_correct", "class_type"
        ])
        df.to_csv(f, sep=",", index=False)
        print("file done")

    print(f"✅ Saved {output_name} with {len(result_rows)} rows")

# === RUN ALL COMBINATIONS ===
if __name__ == "__main__":
    print("Initializing Inference")
    for model in ["FECNet", "FERPlus"]:
        for dataset in ["train", "trainCrop"]:
                print("Running inference with " + str(model) + "on " + str(dataset))
                run_inference_and_save(model, dataset)
    
    
