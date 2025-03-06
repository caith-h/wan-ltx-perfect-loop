import torch

class WanInput:
    @classmethod
    def INPUT_TYPES(cls):
        """
        This node simply takes a batch of images as input.
        Each image is typically shape (B, H, W, C) with B≥1.
        We'll assume these frames form a 16fps sequence.
        """
        return {
            "required": {
                "images": ("IMAGE",),  # A batch of frames
            },
        }

    CATEGORY = "wan-ltx perfect loop"

    # We output four things:
    #  1) First frame
    #  2) Middle frame
    #  3) Last frame
    #  4) The entire batch as "Source Images"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("First", "Middle", "Last", "Source Images", "Width", "Height")
    FUNCTION = "extract_frames"

    def extract_frames(self, images):
        """
        1) Accepts a batch of frames: shape (B, H, W, C).
        2) Extract the first, middle, and last frames as separate outputs.
        3) Also return the entire batch as 'Source Images'.
        """

        # Number of frames in the batch
        batch_size = images.shape[0]
        if batch_size == 0:
            raise ValueError("No frames provided in 'images'.")

        # Indices for first, middle, last
        first_idx = 0
        mid_idx = batch_size // 2
        last_idx = batch_size - 1

        first_frame = images[first_idx].unsqueeze(0)   # (1, H, W, C)
        middle_frame = images[mid_idx].unsqueeze(0)    # (1, H, W, C)
        last_frame = images[last_idx].unsqueeze(0)     # (1, H, W, C)

        # Get dimensions from the input tensor (assuming shape is (B, H, W, C))
        height = int(images.shape[1])
        width = int(images.shape[2])

        print(images.shape[1])

        return (first_frame, middle_frame, last_frame, images, width, height)
