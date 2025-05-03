# import pandas as pd
# import urllib
# import cv2, os
# import requests
# from multiprocessing import Pool

# def get_img(name_dic, verbose=True):
#     id, name_dic = name_dic
#     for key, value in name_dic.items():
#         if not key.startswith("http"):  # Check if the URL is invalid (does not start with 'http')
#             if verbose:
#                 print(f"Invalid URL, skipping: {key}")
#             return id, False
#         if not os.path.isfile("data/train/" + key.split('/')[-1]):  # Check if the image is not already downloaded
#             try:
#                 headers = {
#                     'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36'
#                 }
#                 response = requests.get(key, headers=headers)  # Send GET request to the image URL
#                 if response.status_code:  # If the request is successful
#                     data = response.content  # Get the content (image data)
#                     check_chars = data[-2:]  # Check the last two bytes to verify it's a complete image (JPEG)

#                     if check_chars != b'\xff\xd9':  # If the image is not complete, skip it
#                         if verbose:
#                             print('Not complete image, failed to download: ' + str(key))
#                         return id, False
#                     else:
#                         name = "data/train/" + key.split('/')[-1]  # Create the file path for the image
#                         with open(name, 'wb') as f:
#                             f.write(data)  # Write the image data to the file
#                         image = cv2.imread(name)  # Read the saved image using OpenCV
#                         if image is not None:  # If the image is successfully loaded
#                             x = image.shape  # Get the image's dimensions (height, width, channels)
#                             points = name_dic.get(key)  # Get the cropping coordinates from the dictionary
#                             # Crop the image using the points
#                             image = image[int(points[2] * x[0]):int(points[3] * x[0]),
#                                           int(points[0] * x[1]):int(points[1] * x[1])]
#                             res = cv2.resize(image, (224, 224))  # Resize the cropped image to 224x224
#                             cv2.imwrite(name, res)  # Save the resized image
#                             if verbose:
#                                 print("Downloaded image with URL:", key)
#             except Exception as e:  # Handle any exception that occurs
#                 if verbose:
#                     print(f"Error occurred with URL {key}: {e}")
#                 return id, False
#     return id, True  # Return True if the image was downloaded successfully

# def download_img():
#     dataset = pd.read_csv('data/FEC_dataset/faceexp-comparison-data-train-public.csv', header=None, on_bad_lines='skip')  # Load the dataset

#     steps = 20  # Number of parallel threads for downloading images

#     # Initialize the list to hold images that need to be downloaded
#     images_add = []
#     download_status = [False] * len(dataset)  # Track the download status of each image
#     triplet_count = 0  # Initialize a counter for the number of valid triplets downloaded

#     for id, data in dataset.iterrows():  # Loop through each row in the dataset
#         if not os.path.isfile("data/train/" + data[0].split('/')[-1]) or\
#            not os.path.isfile("data/train/" + data[5].split('/')[-1]) or\
#            not os.path.isfile("data/train/" + data[10].split('/')[-1]):
#             # Check if any of the images in the triplet are not already downloaded
#             images_add.append([[id, {data[0]: [data[1], data[2], data[3], data[4]],
#                                      data[5]: [data[6], data[7], data[8], data[9]],
#                                      data[10]: [data[11], data[12], data[13], data[14]],}]])

#         else:  # If the images are already downloaded, mark them as downloaded
#             download_status[id] = True

#     print("Number of all images:", len(dataset)*3)  # Total number of images (3 per row)
#     print("Number of images to download:", len(images_add)*3)  # Number of images to download
    
#     # Start parallel downloading using multiprocessing
#     with Pool(steps) as p:
#         print("Start parallel downloading")
#         outputs = p.starmap(get_img, images_add)

#     # Update the download status and count valid triplets
#     for id, status in outputs:
#         download_status[id] = status
#         if status:  # If the image was downloaded successfully, count it as a valid triplet
#             triplet_count += 1
#         if triplet_count >= 100:  # Stop if we have downloaded 100 valid triplets
#             print("Downloaded 100 triplets. Stopping the download process.")
#             break

#     print("Number of downloaded images:", len(os.listdir("data/train/")))  # Print the number of downloaded images

#     # Update the dataset with download status and save it
#     if not os.path.isfile("data/FEC_dataset/faceexp-comparison-data-train-public-downloaded_after_download.csv"):
#         dataset[download_status].to_csv("data/FEC_dataset/faceexp-comparison-data-train-public-downloaded_after_download.csv", header=None, index=None)  # Save the dataset with download status to CSV

import pandas as pd
import os
import cv2
import requests
import hashlib
import time
import json
import mediapipe as mp

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

def check_url_availability(url, verbose=True):
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

def download_and_process_image(url, points, save_path, mediapipe_save_path, verbose=True):
    """Download and process a single image with both cropping methods"""
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
            return False
            
        # Save the temp image
        temp_path = save_path + ".temp"
        with open(temp_path, 'wb') as f:
            f.write(data)
            
        # Read the image
        image = cv2.imread(temp_path)
        if image is None:
            if verbose:
                print(f"Failed to read image: {url}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
        
        # Process for train directory - traditional cropping
        try:
            x = image.shape
            cropped_image = image[int(points[2] * x[0]):int(points[3] * x[0]),
                          int(points[0] * x[1]):int(points[1] * x[1])]
            resized_image = cv2.resize(cropped_image, (224, 224))
            cv2.imwrite(save_path, resized_image)
        except Exception as e:
            print(f"Error with traditional cropping: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
        
        # Process for train2 directory - MediaPipe cropping
        try:
            # Use MediaPipe for face detection
            mediapipe_cropped = detect_and_crop_face(cropped_image)
            if mediapipe_cropped is not None and mediapipe_cropped.size > 0:
                mediapipe_resized = cv2.resize(mediapipe_cropped, (224, 224))
                cv2.imwrite(mediapipe_save_path, mediapipe_resized)
            else:
                # Fallback to traditional crop
                cv2.imwrite(mediapipe_save_path, resized_image)
        except Exception as e:
            print(f"Error with MediaPipe cropping: {e}")
            # Use traditional crop as fallback
            try:
                cv2.imwrite(mediapipe_save_path, resized_image)
            except:
                pass
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        if verbose:
            print(f"Successfully processed: {url}")
        return True
        
    except Exception as e:
        print(f"Error with URL {url}: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        if os.path.exists(mediapipe_save_path):
            os.remove(mediapipe_save_path)
        return False

def download_img():
    """Download images and process them with both cropping methods"""
    # Create directories
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/train2", exist_ok=True)
    os.makedirs("data/FEC_dataset", exist_ok=True)
    
    # Clean directories
    for directory in ["data/train", "data/train2"]:
        for file in os.listdir(directory):
            try:
                os.remove(os.path.join(directory, file))
            except Exception as e:
                print(f"Could not remove {file}: {e}")
    
    # Load dataset
    try:
        dataset = pd.read_csv('data/FEC_dataset/faceexp-comparison-data-train-public.csv', 
                           header=None, 
                           on_bad_lines='skip',
                           low_memory=False)
        print(f"Loaded dataset with {len(dataset)} rows")
    except FileNotFoundError:
        print("Error: Dataset file not found")
        return
    
    # Tracking variables
    triplet_to_row_index = {}
    downloaded_urls = set()
    successful_indices = []
    valid_triplet_count = 0
    
    print("Starting image download...")
    
    for row_idx in range(len(dataset)):
        # Stop at 100 triplets
        if valid_triplet_count >= 100:
            break
            
        # Get row
        row = dataset.iloc[row_idx]
        
        # Extract URLs
        urls = [row[0], row[5], row[10]]
        
        # Basic validation
        if not all(validate_url(url) for url in urls):
            continue
        if any(url in downloaded_urls for url in urls):
            continue
        if len(set(urls)) != 3:
            continue
        
        # Check availability
        if not all(check_url_availability(url) for url in urls):
            continue
            
        print(f"Processing row {row_idx}...")
        
        # Extract points
        try:
            points = [
                [float(row[1]), float(row[2]), float(row[3]), float(row[4])],
                [float(row[6]), float(row[7]), float(row[8]), float(row[9])],
                [float(row[11]), float(row[12]), float(row[13]), float(row[14])]
            ]
        except (ValueError, TypeError):
            continue
        
        # Create triplet number and filenames
        valid_triplet_count += 1
        triplet_to_row_index[valid_triplet_count] = row_idx
        
        # Process all three images
        filenames = []
        mediapipe_filenames = []
        
        for i, url in enumerate(urls):
            url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:10]
            filename = f"{valid_triplet_count:03d}_{i+1}_{row_idx}_{url_hash}.jpg"
            filenames.append(os.path.join("data/train", filename))
            mediapipe_filenames.append(os.path.join("data/train2", filename))
        
        # Download and process
        all_successful = True
        
        for i in range(3):
            success = download_and_process_image(
                urls[i], 
                points[i], 
                filenames[i],
                mediapipe_filenames[i]
            )
            if not success:
                all_successful = False
                break
        
        # Cleanup on failure
        if not all_successful:
            print(f"Row {row_idx}: Failed")
            for path in filenames + mediapipe_filenames:
                if os.path.exists(path):
                    os.remove(path)
            continue
        
        # Success tracking
        print(f"Row {row_idx}: Success")
        successful_indices.append(row_idx)
        for url in urls:
            downloaded_urls.add(url)
        print(f"Progress: {valid_triplet_count}/100 triplets")
        time.sleep(0.1)
    
    # Save mapping
    with open("data/triplet_to_row_mapping.json", 'w') as f:
        json.dump(triplet_to_row_index, f)
    
    # Save successful rows
    successful_dataset = dataset.iloc[successful_indices]
    output_csv = "data/FEC_dataset/faceexp-comparison-data-train-public-downloaded.csv"
    successful_dataset.to_csv(output_csv, header=None, index=None)
    
    print(f"Download complete. {valid_triplet_count} triplets processed.")
    print(f"Train directory: {len(os.listdir('data/train/'))} images")
    print(f"Train2 directory: {len(os.listdir('data/train2/'))} images")