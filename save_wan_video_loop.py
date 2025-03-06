import torch

class SaveWanVideoLoop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # The new frames to be appended (assumed to be a batch of images)
                "images": ("IMAGE",),
                # The original batch of frames (assumed to be at 16 fps)
                "source_images": ("IMAGE",),
            }
        }

    CATEGORY = "wan-ltx perfect loop"
    # Output a batch of images that can later be saved by ComfyUI's built-in saver
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Images",)
    FUNCTION = "combine_loop"

    def combine_loop(self, images, source_images):
        """
        Combines two batches of frames (each assumed to be 16 fps):
          1) Drops every odd-indexed frame from the 'images' input (keeping frames 0, 2, 4, ...).
          2) Appends these filtered frames to the end of the 'source_images' batch.
          3) Returns the combined batch as 'Images' for further processing or saving.
        """
        # Keep only even-indexed frames from the new images
        filtered_images = images[::2]
        # Concatenate source_images and the filtered images along the batch dimension
        combined_images = torch.cat([source_images, filtered_images], dim=0)
        return (combined_images, )
