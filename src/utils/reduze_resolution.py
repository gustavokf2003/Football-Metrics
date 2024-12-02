import os
from moviepy.editor import VideoFileClip

def resize_videos_from_folder(folder, width, height, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate through the folder and filter video files
    videos = [os.path.join(folder, f) for f in os.listdir(folder)]
    
    for video in videos:
        try:
            # Load the video
            clip = VideoFileClip(video)
            # Resize to the desired resolution
            clip_resized = clip.resize(newsize=(width, height))
            # Define the output name
            video_name = os.path.basename(video)
            output_path = os.path.join(output_folder, video_name)
            # Save the resized video
            clip_resized.write_videofile(output_path, codec='libx264', fps=clip.fps)
            print(f'Video "{video_name}" processed and saved to "{output_path}".')
        except Exception as e:
            print(f'Error processing the video "{video}": {e}')


