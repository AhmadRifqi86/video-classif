import os
import cv2

# Specify the video path
video_path = "/app/videos/@naiwen88_video_7355136826299960583.mp4"
#video_path = "/home/arifadh/Downloads/tiktok_videos/@naiwen88_video_7355136826299960583.mp4"
print(cv2.getBuildInformation())
print(cv2.__file__)
# Check if the video file exists
try:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at {video_path}")
    
    # Attempt to open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Unable to open video file {video_path}")
    
    print("Video file opened successfully!")
    cap.release()

except Exception as e:
    print(f"Error: {e}")

#docker run --gpus all --rm  -v /home/arifadh/Downloads/tiktok_videos:./ test
#