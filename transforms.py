import warnings
warnings.filterwarnings("ignore", message="Error fetching version info")
import albumentations as A


def get_training_augmentation_pipeline():
    """
    Creates an augmentation pipeline for training data with rotations and flips.
    Includes random rotations by specific angles and random horizontal/vertical flips.

    Returns:
        albumentations.Compose: A pipeline with training augmentations.
    """
    augmentation_pipeline = A.Compose([
        # Apply one of the specified fixed-angle rotations
        A.OneOf([
            A.Rotate(limit=(-30, -30), p=0.25),  # Rotate exactly -30 degrees
            A.Rotate(limit=(15, 15), p=0.25),    # Rotate exactly 15 degrees
            A.Rotate(limit=(30, 30), p=0.25),    # Rotate exactly 30 degrees
            A.Rotate(limit=(-15, -15), p=0.25)   # Rotate exactly -15 degrees
        ], p=4/7),  # Ensure one rotation is always applied

        # Apply either horizontal or vertical flip with equal probability
        # A.OneOf([
        #     A.HorizontalFlip(p=0.5),  # Flip horizontally with 50% probability
        #     A.VerticalFlip(p=0.5),    # Flip vertically with 50% probability
        # ], p=2/9),  # Ensure one flip is always applied

        A.OneOf([
            A.Affine(scale=(0.8, 1.2), balanced_scale=True, p=0.33),  # Zoom
            A.Affine(translate_px=(-10, 10), p=0.33),  # Translation
            A.Affine(shear=(-20, 20), p=0.33),  # Shearing
        ], p=3/7),

    ])

    return augmentation_pipeline
