import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T

from transforms import *

class EchocardiogramDataset(Dataset):
    """
    A PyTorch Dataset class to handle echocardiogram images and masks.
    This class supports dynamic data augmentation and preprocessing during data loading.
:
    Attributes:
        images (list): List of transformed image tensors.
        masks (list): List of transformed mask tensors.
        augmentation (callable): A callable function or object to apply augmentations to the images and masks.
    """

    def __init__(self, images, masks, augmentation=None):
        """
        Initializes the dataset with images, masks, and optional augmentation and preprocessing steps.

        Args:
            images (list): List of image arrays (e.g., numpy arrays).
            masks (list): List of mask arrays (e.g., numpy arrays).
            augmentation (callable, optional): Augmentation function to apply to images and masks dynamically.
        """
        self.images = images
        self.masks = masks

        self.augmentation = augmentation  # Store augmentation function if provided

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The total number of images/masks in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Fetches the image and mask at the specified index and applies augmentation and preprocessing if provided.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            Tuple[Tensor, Tensor]: The processed image and corresponding mask.
        """
        # Read images and masks for the given index
        image = self.images[idx]
        mask = self.masks[idx]

        # Dynamically apply augmentations if an augmentation function is provided
        if self.augmentation:
            # The augmentation function should take `image` and `mask` and return augmented versions.
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return self.to_tensor(image, False), self.to_tensor(mask, True)

    @staticmethod
    def to_tensor(data, is_mask=False):
        """
        Converts a NumPy array to a PyTorch tensor, with separate handling for images and masks.

        - Images: Scaled to [0, 1] and converted to `float32`.
        - Masks: Kept as integer labels without normalization (for compatibility with loss functions like CrossEntropyLoss).

        Args:
            data (numpy.ndarray): Input array (image or mask).
            is_mask (bool, optional): Whether the input is a mask. Default is False.

        Returns:
            torch.Tensor: Converted PyTorch tensor.
        """
        transform_image = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=(not is_mask))  # Scale image values to [0, 1]
        ])
        return transform_image(data.squeeze())

def get_loader(images, masks, batch_size=8, mode=None):
    """
    Builds and returns a PyTorch DataLoader for echocardiogram image segmentation.

    This function constructs a DataLoader by creating an `EchocardiogramDataset` object,
    applying preprocessing and optional data augmentation based on the specified mode.

    Args:
        images (list): List of image arrays (e.g., NumPy arrays or compatible format).
        masks (list): List of corresponding mask arrays.
        image_size (int, optional): Target size for resizing images and masks (square dimensions). Default is 224.
        batch_size (int, optional): Number of samples per batch. Default is 8.
        mode (str, optional): Mode of operation, one of 'train', 'valid', or 'test'. Determines
                              whether augmentation is applied and whether shuffling is enabled. Default is 'train'.

    Returns:
        torch.utils.data.DataLoader: DataLoader instance for batching and iterating through the dataset.

    Notes:
        - In 'train' mode, data augmentation is applied, and shuffling is enabled.
        - In 'valid' or 'test' mode, no augmentation is applied, and shuffling is disabled.
        - Preprocessing is applied regardless of the mode.
    """
    # Create preprocessing and augmentation pipelines
    augmentation_pipeline = None

    if mode == 'train':
        augmentation_pipeline = get_training_augmentation_pipeline()

    # Initialize the dataset
    dataset = EchocardiogramDataset(
        images, masks,
        augmentation=augmentation_pipeline,
    )

    # Create and return the DataLoader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=batch_size,
        shuffle=(mode == 'train'),  # Shuffle only in training mode
    )
    return data_loader