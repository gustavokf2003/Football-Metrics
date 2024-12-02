import os
import random
import cv2
import numpy as np
from PIL import Image

def enhance_red_shirt(image):
    """Enhances the red color intensity in the shirt."""
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the ranges for the red color
    lower_red_1 = np.array([0, 70, 50])     # Lower bound for red
    upper_red_1 = np.array([10, 255, 255])  # Upper bound for red
    lower_red_2 = np.array([170, 70, 50])   # Lower bound for red
    upper_red_2 = np.array([180, 255, 255]) # Upper bound for red
    
    # Create masks for the red color
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Increase the red intensity in the original image
    img_enhanced = image.copy()
    img_enhanced[mask > 0] = img_enhanced[mask > 0] + np.array([0, 0, 50])  
    
    # Ensure values do not exceed 255
    img_enhanced = np.clip(img_enhanced, 0, 255)

    return img_enhanced

def extract_random_frames(video_folder, n_frames=5, output_folder='random_frames'):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all video files in the specified folder
    videos = [f for f in os.listdir(video_folder)]

    # Process each video found
    for video in videos:
        video_path = os.path.join(video_folder, video)
        try:
            # Capture the video using OpenCV
            video_cap = cv2.VideoCapture(video_path)

            # Get the video duration
            fps = video_cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps  # Duration in seconds
            
            # Generate n random moments (in seconds) within the video duration
            moments = sorted([random.uniform(0, duration) for _ in range(n_frames)])
            
            # Extract and save frames at the generated moments
            video_name = os.path.basename(video_path).split('.')[0]
            for i, moment in enumerate(moments):
                # Calculate the corresponding frame number for the moment
                frame_num = int(moment * fps)
                
                # Read the corresponding frame
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = video_cap.read()
                
                if not ret:
                    print(f'Could not read the frame at {moment:.2f}s in video "{video}".')
                    continue
                
                # Enhance the red color in the frame
                frame_enhanced = enhance_red_shirt(frame)
                
                # Save the frame
                frame_path = os.path.join(output_folder, f"{video_name}_frame_{i+1}.png")
                Image.fromarray(cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)).save(frame_path)
                print(f'Frame {i+1} extracted from video "{video}" at {moment:.2f}s and saved to "{frame_path}".')

            # Release the capture resource
            video_cap.release()
        except Exception as e:
            print(f'Error processing video "{video_path}": {e}')
