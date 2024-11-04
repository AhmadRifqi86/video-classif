import os
import re
import cv2
import torch
import numpy as np
import argparse
from dump_lrcn import LRCN

IMG_HEIGHT, IMG_WIDTH = 64, 64  # Image dimensions
SEQUENCE_LENGTH = 60
MODEL_PATH = '/home/arifadh/Desktop/Skripsi-Magang-Proyek/best-models/best_model_seq60_batch8_hidden48_cnndensenet121_rnn512_layer4_methoduniform_outall_max250_epochs15_finetuneTrue_f10.6408.pth'  # Path to the saved model
CLASS_LABELS = ['Theft', 'Violence', 'Vandalism']

# Natural sorting function
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# Preprocess frames for inference
def preprocess_frames(frames_path, sequence_length=SEQUENCE_LENGTH):
    frames = []
    
    # Load all frame images from the directory
    frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith(('.png', '.jpg'))], key=natural_sort_key)
    
    for frame_file in frame_files:
        frame_path = os.path.join(frames_path, frame_file)
        frame = cv2.imread(frame_path)
        frame_resized = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))  # Resize frame
        frame_normalized = frame_resized / 255.0  # Normalize pixel values
        frames.append(frame_normalized.transpose(2, 0, 1))  # HWC to CHW format
    
    # If the number of frames is less than the sequence length, pad with zeros
    if len(frames) < sequence_length:
        frames += [np.zeros((3, IMG_HEIGHT, IMG_WIDTH))] * (sequence_length - len(frames))
    
    # If the number of frames exceeds the sequence length, truncate
    frames = frames[:sequence_length]
    
    # Convert to a PyTorch tensor and add the batch dimension
    return torch.tensor(frames).unsqueeze(0)

def load_model(model_path, num_classes, sequence_length, hidden_size, rnn_input_size):
    model = LRCN(num_classes=num_classes, sequence_length=sequence_length, hidden_size=hidden_size, rnn_input_size=rnn_input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Perform inference
def classify_video(model, video_tensor):
    with torch.no_grad():
        output = model(video_tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# CLI function
def main():
    parser = argparse.ArgumentParser(description='LRCN Video Classifier CLI')
    parser.add_argument('--frames', type=str, required=True, help='Path to the directory containing video frames')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to the saved model')
    args = parser.parse_args()

    # Preprocess the frames
    video_tensor = preprocess_frames(args.frames)

    # Load the model
    model = load_model(args.model, num_classes=len(CLASS_LABELS), sequence_length=SEQUENCE_LENGTH, hidden_size=56, rnn_input_size=512)

    # Perform classification
    predicted_class = classify_video(model, video_tensor)
    print(f"Predicted class: {CLASS_LABELS[predicted_class]}")

if __name__ == "__main__":
    main()