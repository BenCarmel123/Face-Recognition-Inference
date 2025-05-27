import pandas as pd
import os
import cv2
import requests
import hashlib
import time
import json
import mediapipe as mp
from tqdm import tqdm

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def detect_and_crop_face(frame):
    """Detect and crop face using MediaPipe"""
    if frame is None:
        return None
        
    h, w, _ = frame.shape
    # Convert frame to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Extract bounding box coordinates
            bboxC = detection.location_data.relative_bounding_box
            x_min = int(bboxC.xmin * w)
            y_min = int(bboxC.ymin * h)
            x_max = int((bboxC.xmin + bboxC.width) * w)
            y_max = int((bboxC.ymin + bboxC.height) * h)

            # Crop the face
            cropped_face = frame[y_min:y_max, x_min:x_max]

            # Return cropped face if valid
            if cropped_face.size > 0:
                return cropped_face
        
    return None

def validate_url(url):
    """Check if a URL is valid (starts with http)"""
    return isinstance(url, str) and url.startswith("http")

def check_url_availability(url, verbose=False):
    """Check if a URL is available"""
    if not validate_url(url):
        if verbose:
            print(f"Invalid URL format: {url}")
        return False
        
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36',
            'Range': 'bytes=0-1024'  # Request only first 1KB
        }
        
        response = requests.get(url, headers=headers, timeout=5, stream=True)
        response.close()
        
        if response.status_code not in [200, 206]:
            if verbose:
                print(f"URL not available (status code {response.status_code}): {url}")
            return False
            
        return True
        
    except Exception as e:
        if verbose:
            print(f"Error checking URL {url}: {e}")
        return False

def download_and_process_image(url, points, temp_file_path, verbose=False):
    """Download and process a single image with both cropping methods
    Returns (download_success, traditional_crop, mediapipe_crop)
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        data = response.content
        
        # Check if image is complete
        if len(data) < 2 or data[-2:] != b'\xff\xd9':
            if verbose:
                print('Not complete image, failed to download: ' + str(url))
            return False, None, None
            
        # Save the temp image
        with open(temp_file_path, 'wb') as f:
            f.write(data)
            
        # Read the image
        image = cv2.imread(temp_file_path)
        if image is None:
            if verbose:
                print(f"Failed to read image: {url}")
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return False, None, None
        
        # Process with traditional cropping
        try:
            x = image.shape
            cropped_image = image[int(points[2] * x[0]):int(points[3] * x[0]),
                          int(points[0] * x[1]):int(points[1] * x[1])]
            
            # Traditional crop was successful
            traditional_crop = cv2.resize(cropped_image, (224, 224))
        except Exception as e:
            if verbose:
                print(f"Error with traditional cropping: {e}")
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return False, None, None
        
        # Process with MediaPipe face detection
        try:
            # Use MediaPipe for face detection
            mediapipe_cropped = detect_and_crop_face(cropped_image)
            if mediapipe_cropped is not None and mediapipe_cropped.size > 0:
                # MediaPipe crop was successful - keep original dimensions
                mediapipe_crop = mediapipe_cropped
            else:
                # MediaPipe failed to detect a face
                mediapipe_crop = None
        except Exception as e:
            if verbose:
                print(f"Error with MediaPipe cropping: {e}")
            mediapipe_crop = None
        
        # Clean up
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        return True, traditional_crop, mediapipe_crop
        
    except Exception as e:
        if verbose:
            print(f"Error with URL {url}: {e}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return False, None, None

def download_img():
    """Download images and process them with both cropping methods"""
    # Create directories
    os.makedirs("train2", exist_ok=True)
    os.makedirs("FEC_dataset", exist_ok=True)
    
    # Clean directory
    for file in os.listdir("train2"):
        try:
            os.remove(os.path.join("train2", file))
        except Exception as e:
            print(f"Could not remove {file}: {e}")
    
    # Load dataset
    try:
        dataset = pd.read_csv('FEC_dataset/faceexp-comparison-data-train-public.csv', 
                           header=None, 
                           on_bad_lines='skip',
                           low_memory=False)
        print(f"Loaded dataset with {len(dataset)} rows")
    except FileNotFoundError:
        print("Error: Dataset file not found")
        return
    
    # Tracking variables
    triplet_to_row_index = {}  # Maps valid triplet numbers to original row indices
    mediapipe_corrupted_indices = []  # Tracks row indices where MediaPipe failed
    downloaded_urls = set()  # Keeps track of already downloaded URLs
    successful_indices = []  # Tracks row indices of completely successful triplets
    valid_triplet_count = 0  # Counts valid triplets (all 3 images pass all checks)
    
    total_rows = len(dataset)
    print(f"Starting image download for {total_rows} rows...")
    
    # Initialize simple tqdm progress bar
    pbar = tqdm(total=total_rows, desc="Processing", 
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] - Row: {postfix}")
    pbar.set_postfix_str(f"0/{total_rows} - Success: 0 - MediaPipe Fail: 0")
    
    for row_idx in range(len(dataset)):
        # Get row
        row = dataset.iloc[row_idx]
        
        # Extract URLs
        urls = [row[0], row[5], row[10]]
        
        # Basic validation
        if not all(validate_url(url) for url in urls):
            pbar.set_postfix_str(f"{row_idx}/{total_rows} - Success: {valid_triplet_count} - MediaPipe Fail: {len(mediapipe_corrupted_indices)}")
            pbar.update(1)
            continue
        if any(url in downloaded_urls for url in urls):
            pbar.set_postfix_str(f"{row_idx}/{total_rows} - Success: {valid_triplet_count} - MediaPipe Fail: {len(mediapipe_corrupted_indices)}")
            pbar.update(1)
            continue
        if len(set(urls)) != 3:
            pbar.set_postfix_str(f"{row_idx}/{total_rows} - Success: {valid_triplet_count} - MediaPipe Fail: {len(mediapipe_corrupted_indices)}")
            pbar.update(1)
            continue
        
        # Check availability
        if not all(check_url_availability(url) for url in urls):
            pbar.set_postfix_str(f"{row_idx}/{total_rows} - Success: {valid_triplet_count} - MediaPipe Fail: {len(mediapipe_corrupted_indices)}")
            pbar.update(1)
            continue
            
        # Extract points
        try:
            points = [
                [float(row[1]), float(row[2]), float(row[3]), float(row[4])],
                [float(row[6]), float(row[7]), float(row[8]), float(row[9])],
                [float(row[11]), float(row[12]), float(row[13]), float(row[14])]
            ]
        except (ValueError, TypeError):
            pbar.set_postfix_str(f"{row_idx}/{total_rows} - Success: {valid_triplet_count} - MediaPipe Fail: {len(mediapipe_corrupted_indices)}")
            pbar.update(1)
            continue
        
        # Prepare for processing
        triplet_num = valid_triplet_count + 1
        
        # Process all three images
        download_results = []
        downloaded_images = []
        traditional_crops = []
        mediapipe_crops = []
        filenames = []
        
        for i, url in enumerate(urls):
            url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:10]
            filename = f"{triplet_num:03d}_{i+1}_{row_idx}_{url_hash}.jpg"
            filenames.append(os.path.join("train2", filename))
            
            # Download and process the image
            temp_file_path = f"temp_{url_hash}.jpg"
            download_success, traditional_crop, mediapipe_crop = download_and_process_image(
                urls[i], 
                points[i],
                temp_file_path
            )
            
            download_results.append(download_success)
            downloaded_images.append(url)
            traditional_crops.append(traditional_crop)
            mediapipe_crops.append(mediapipe_crop)
            
        # Check if all images were downloaded and processed successfully
        if not all(download_results):
            pbar.set_postfix_str(f"{row_idx}/{total_rows} - Success: {valid_triplet_count} - MediaPipe Fail: {len(mediapipe_corrupted_indices)}")
            pbar.update(1)
            continue
            
        # Check if all MediaPipe detections succeeded
        mediapipe_success = all(crop is not None for crop in mediapipe_crops)
        
        if not mediapipe_success:
            # At least one MediaPipe detection failed
            mediapipe_corrupted_indices.append(row_idx)
            pbar.set_postfix_str(f"{row_idx}/{total_rows} - Success: {valid_triplet_count} - MediaPipe Fail: {len(mediapipe_corrupted_indices)}")
            pbar.update(1)
            continue
            
        # If we reached here, this is a valid triplet
        valid_triplet_count += 1
        triplet_to_row_index[valid_triplet_count] = row_idx
        
        # Save MediaPipe crops to train2
        for i in range(3):
            cv2.imwrite(filenames[i], mediapipe_crops[i])
            
        # Success tracking
        successful_indices.append(row_idx)
        for url in downloaded_images:
            downloaded_urls.add(url)
        
        pbar.set_postfix_str(f"{row_idx}/{total_rows} - Success: {valid_triplet_count} - MediaPipe Fail: {len(mediapipe_corrupted_indices)}")
        pbar.update(1)
    
    pbar.close()
    
    # Save mapping of valid triplet number to original row index
    with open("triplet_to_row_mapping.json", 'w') as f:
        json.dump(triplet_to_row_index, f)
    
    # Save corrupted indices due to MediaPipe failures
    with open("mediapipe_corrupted_indices.json", 'w') as f:
        json.dump({"corrupted_indices": mediapipe_corrupted_indices}, f)
    
    # Save successful rows
    successful_dataset = dataset.iloc[successful_indices]
    output_csv = "FEC_dataset/faceexp-comparison-data-train-public-downloaded.csv"
    successful_dataset.to_csv(output_csv, header=None, index=None)
    
    print(f"Download complete. {valid_triplet_count} triplets processed successfully.")
    print(f"Train2 directory: {len(os.listdir('train2/'))} images")
    print(f"MediaPipe corrupted triplets: {len(mediapipe_corrupted_indices)}")

if __name__ == "__main__":
    download_img()
