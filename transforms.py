import albumentations as A
import cv2
import elasticdeform
import numpy as np

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
        
        # Custom elastic deformation
        # ElasticDeform(sigma=6, points=2, order=1, always_apply=False, p=1.0),
    ])

    return augmentation_pipeline

# Custom Elastic Deformation Wrapper
class ElasticDeform(A.ImageOnlyTransform):
    def __init__(self, sigma=6, points=3, order=1, always_apply=False, p=1.0):
        super(ElasticDeform, self).__init__(always_apply, p)
        self.sigma = sigma
        self.points = points
        self.order = order

    def __call__(self, image, mask=None, **kwargs):
        # Apply deformation on both image and mask simultaneously
        deformed_image, deformed_mask = elasticdeform.deform_random_grid(
            [image, mask],
            sigma=self.sigma,
            points=self.points,
            order=self.order,
            axis=(0, 1)
        )

        # Normalize the deformed image to range 0-255
        image_min = np.min(deformed_image)
        image_max = np.max(deformed_image)
        if image_max - image_min > 0:
            deformed_image = ((deformed_image - image_min) / (image_max - image_min)) * 255
        else:
            deformed_image = np.zeros_like(deformed_image)

        return {"image": deformed_image.astype(np.uint8), "mask": deformed_mask.astype(np.uint8)}
