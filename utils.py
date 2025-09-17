import pickle
import gzip
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.restoration import denoise_nl_means
from skimage.filters import threshold_yen
import torch
import os
import torch.nn.functional as F
import albumentations as A

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)

def preprocess_train_data(data):
    """
    Preprocesses the training data, extracting video frames and corresponding segmentation masks.

    Args:
    - data: A list of dictionaries, where each dictionary contains the following keys:
        'name': Video name (string)
        'video': Video data (numpy array of shape (height, width, n_frames))
        'box': Bounding box for the video (not used here)
        'label': Segmentation mask for each frame (numpy array of shape (height, width, n_frames))
        'frames': List of indices (frame numbers) where the segmentation is available
        'dataset': Information about whether the video is from amateurs or experts

    Returns:
    - names: List of video names
    - video_frames: List of individual video frames (numpy arrays)
    - mask_frames: List of corresponding segmentation masks for each video frame
    """
    # Initialize empty lists to store processed video frames, mask frames, and video names
    video_frames = []
    mask_frames = []
    names = []

    IMAGE_SIZE = 128

    hist_eq = A.Compose([A.CLAHE(p=1.0),])

    # Iterate over the input data using tqdm to show a progress bar
    for item in tqdm(data):
        # Extract video and video metadata
        video = item['video']
        name = item['name']
        height, width, n_frames = video.shape  # Get the dimensions of the video

        # Initialize an empty mask array of the same shape as the video, for storing the segmentation masks
        mask = np.zeros((height, width, n_frames), dtype=bool)

        # Iterate over the frames for which segmentation data is available
        for frame in item['frames']:
            # Copy the corresponding segmentation mask into the mask array
            mask[:, :, frame] = item['label'][:, :, frame]

            # Extract the video frame and corresponding mask frame for the given index
            video_frame = video[:, :, frame].astype('float32') / 255.0
            mask_frame = mask[:, :, frame]

            # Expand the dimensions of the frames to ensure they have the shape (height, width, 1)
            video_frame = np.expand_dims(video_frame, axis=2).astype(np.float32)
            mask_frame = np.expand_dims(mask_frame, axis=2).astype(np.uint8)
            
            # Resize
            video_frame = cv2.resize(video_frame, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            mask_frame = cv2.resize(mask_frame, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            
            # Denoise
            video_frame = denoise_nl_means(video_frame)
            
            # Histogram equalization
            video_frame = hist_eq(image=video_frame)['image']

            # Add extra layers to input
            blurred_video_frame = cv2.medianBlur(video_frame, 3)
            threshold = threshold_yen(video_frame)
            thresholded_video_frame = np.array(video_frame > threshold, dtype='int')
            
            video_frame = np.stack([video_frame, blurred_video_frame, thresholded_video_frame], axis=-1)

            # Append the processed frames and name to the lists
            video_frames.append(video_frame)
            mask_frames.append(mask_frame)
            names.append(name)

    # Return the names, video frames, and mask frames
    return names, video_frames, mask_frames

def preprocess_test_data(data):
    """
    Preprocesses the test data, extracting video frames.

    Args:
    - data: A list of dictionaries, where each dictionary contains the following keys:
        'name': Video name (string)
        'video': Video data (numpy array of shape (height, width, n_frames))

    Returns:
    - names: List of video names
    - video_frames: List of individual video frames (numpy arrays)
    """
    # Initialize empty lists to store processed video frames and video names
    video_frames = []
    names = []

    IMAGE_SIZE = 128

    hist_eq = A.Compose([A.CLAHE(p=1.0),])

    # Iterate over the input data using tqdm to show a progress bar
    for item in tqdm(data):
        # Extract the video data
        video = item['video']

        # Convert the video data to float32 and transpose the dimensions to (n_frames, height, width)
        # This is often done to match the expected input shape for neural networks
        video = video.astype(np.float32).transpose((2, 0, 1))  # (n_frames, height, width)

        for frame in video:
            video_frame = frame.astype('float32') / 255.0

            video_frame = np.expand_dims(video_frame, axis=2).astype(np.float32)
            video_frame = cv2.resize(video_frame, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            video_frame = denoise_nl_means(video_frame)
            video_frame = hist_eq(image=video_frame)['image']

            blurred_video_frame = cv2.medianBlur(video_frame, 3)
            threshold = threshold_yen(video_frame)   
            thresholded_video_frame = np.array(video_frame > threshold, dtype='int')

            video_frame = np.stack([video_frame, blurred_video_frame, thresholded_video_frame], axis=-1)

            # Append all frames from the video to the video_frames list
            video_frames.append(video_frame)

        # Append the video name for each frame in the video
        names += [item['name'] for _ in video]

    # Return the video names and video frames
    return names, video_frames

def draw_mask(image, mask_generated):
    """
    Overlay a segmentation mask on the image.

    Args:
        image (numpy.ndarray): Original image.
        mask_generated (numpy.ndarray): Binary mask of the same size as the image.

    Returns:
        numpy.ndarray: Image with the segmentation mask overlay.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Ensure the mask is 2D
    if mask_generated.shape[-1] == 1:  # Remove trailing channel dimension
        mask_generated = mask_generated.squeeze(-1)

    # Apply the mask overlay (green color for segmentation)
    mask_color = np.array([0, 255, 0], dtype='uint8')  # Green mask
    colored_mask = np.where(mask_generated[..., None], mask_color, 0)
    return cv2.addWeighted(image.astype(np.uint8), 0.7, colored_mask.astype(np.uint8), 0.3, 0)

def visualize_data(video_frames, mask_frames, names, num_samples=5):
    """
    Visualize the video frames, corresponding mask frames, and overlays.

    Args:
        video_frames (list): List of video frames.
        mask_frames (list): List of mask frames.
        names (list): List of names corresponding to the frames.
        num_samples (int): Number of samples to visualize. Default is 5.
    """
    num_samples = min(num_samples, len(video_frames))  # Ensure we don't exceed available data
    indices = np.random.choice(len(video_frames), num_samples, replace=False)
    print("Indices of samples shown: ", indices)

    for idx in indices:
        video_frame = video_frames[idx][:, :, 0]  # Remove single-dimensional entries
        mask_frame = mask_frames[idx].squeeze()
        name = names[idx]

        # Ensure the mask and frame dimensions match
        if video_frame.shape[:2] != mask_frame.shape:
            raise ValueError("Video frame and mask frame dimensions do not match.")

        # Generate overlay
        overlay_frame = draw_mask(video_frame, mask_frame)

        plt.figure(figsize=(20, 5))

        # Plot video frame
        plt.subplot(1, 3, 1)
        plt.imshow(video_frame)
        plt.title(f"Video Frame ({name})")
        plt.axis('off')

        # Plot mask frame
        plt.subplot(1, 3, 2)
        plt.imshow(mask_frame, cmap='gray')
        plt.title(f"Mask Frame ({name})")
        plt.axis('off')

        # Plot overlay
        plt.subplot(1, 3, 3)
        plt.imshow(overlay_frame)
        plt.title(f"Overlay ({name})")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

def get_sequences(arr):
    first_indices, last_indices, lengths = [], [], []
    arr = [0] + list(arr) + [0]
    for index, value in enumerate(arr[:-1]):
        if arr[index+1]-arr[index] == 1:
            first_indices.append(index)
        if arr[index+1]-arr[index] == -1:
            last_indices.append(index)
    lengths = list(np.array(last_indices)-np.array(first_indices))
    return first_indices, lengths

def get_submission_ids_and_values(model, test_videos, test_names, test_data, device):
    masks_per_video = dict()

    l = len(test_videos)

    # Keep track of the current video being processed
    video_index = 0
    frame_index_in_video = 0

    for i in range(l):
        frame = torch.from_numpy(test_videos[i]).float().to(device)

        output_mask = model(frame.permute(2, 0, 1).unsqueeze(0)).detach()
        output_mask = torch.sigmoid(output_mask)

        # Get the corresponding video from test_data
        video_data = test_data[video_index]
        original_height, original_width = video_data['video'].shape[:2]  # (h, w, f)

        output_mask = F.interpolate(
            output_mask.cpu(),
            size=(original_height, original_width),
            mode='bicubic',
            align_corners=False
        ).squeeze(0).squeeze(0).numpy()

        binary_mask = (output_mask >= 0.8).astype(float)

        del frame, output_mask

        if masks_per_video.get(test_names[video_index]) is None:
            masks_per_video[test_names[video_index]] = []

        masks_per_video[test_names[video_index]].append(binary_mask)

        del binary_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Update the frame index within the current video
        frame_index_in_video += 1

        # Check if we need to move to the next video (if we've processed all frames for the current video)
        if frame_index_in_video >= video_data['video'].shape[2]:
            video_index += 1
            frame_index_in_video = 0  # Reset the frame index for the next video

    for key in masks_per_video:
        masks_per_video[key] = np.stack(masks_per_video[key], axis=0)

    # Create ids and values for sequences
    ids = []
    values = []
    for e in test_names:
        flattened_mask = masks_per_video[e].transpose(1, 2, 0).flatten()
        first_indices, lengths = get_sequences(flattened_mask)

        l = len(first_indices)

        for i in range(l):
            ids.append(f"{e}_{i}")
            values.append([first_indices[i], lengths[i]])
    
    return ids, values, masks_per_video

def create_videos(test_data, masks_per_video):
    output_dir = 'output_videos'
    os.makedirs(output_dir, exist_ok=True)

    for video_info in test_data:
        video_name = video_info['name']
        frames = masks_per_video[video_name]

        image_sequence = video_info['video']  # Shape (H, W, F)

        height, width, num_frames = image_sequence.shape
        fps = 10
        output_video_path = os.path.join(output_dir, f'{video_name}.mp4')
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for t in range(num_frames):
            image = image_sequence[:, :, t]

            # Convert grayscale image to 3 channels for blending.
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            mask = frames[t]
            mask = (mask * 255).astype(np.uint8)  # Scale mask values to 0-255

            # Paint the mask red (scale mask values into red channel).
            red_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)  # Create blank 3-channel image
            red_mask[:, :, 2] = mask  # Assign mask values to the red channel (index 2)

            # Resize red_mask to match original video dimensions.
            mask_resized = cv2.resize(red_mask, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

            # Blend the original image and the red mask.
            blended_frame = cv2.addWeighted(image_rgb, 0.5, mask_resized, 0.5, 0)

            out.write(blended_frame)

        out.release()

    print("Videos processed and saved successfully.")