from .seggpt import SegGPT

NODE_CLASS_MAPPINGS = {
    "SegGPT": SegGPT
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SegGPT": "SegGPT Node"
}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
