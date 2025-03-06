from .wan_input import WanInput
from .save_wan_video_loop import SaveWanVideoLoop

NODE_CLASS_MAPPINGS = {
    "WanInput": WanInput,
    "SaveWanVideoLoop": SaveWanVideoLoop
}

__all__ = ["NODE_CLASS_MAPPINGS"]